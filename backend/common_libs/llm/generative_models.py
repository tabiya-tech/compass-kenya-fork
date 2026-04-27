import asyncio
import json
import logging
import traceback
from collections import OrderedDict
from typing import Any

from vertexai.generative_models import GenerativeModel, Content, Part, GenerationConfig
from vertexai.language_models import TextGenerationModel

from common_libs.llm.models_utils import LLMConfig, LLMInput, LLMResponse, BasicLLM

logger = logging.getLogger(__name__)


# Cache `GenerativeModel` instances so we don't re-construct one per turn for
# call sites that pass identical config + system instructions (data extraction
# sub-tools, transition tool, bridge LLM, first-time conversation prompts).
#
# Keyed by event-loop id so tests (which spin up a fresh asyncio loop per test)
# don't reuse models whose gRPC channels are bound to a closed loop. In
# production this collapses to a single bucket because uvicorn runs a single
# loop for the lifetime of the process.
#
# Bounded by a simple LRU so we don't leak memory when call sites pass per-turn
# varying system instructions (e.g. conversation LLM follow-up turns whose
# system_instructions include the live collected_data dump).
_GENERATIVE_MODEL_CACHE: "OrderedDict[tuple, GenerativeModel]" = OrderedDict()
_GENERATIVE_MODEL_CACHE_MAX_SIZE = 64


def _current_loop_id() -> int:
    """Return a stable id for the current asyncio loop, or 0 if there isn't one."""
    try:
        return id(asyncio.get_event_loop())
    except RuntimeError:
        return 0


def _make_generative_model_cache_key(
    *,
    model_name: str,
    system_instructions: list[str] | str | None,
    generation_config: dict,
    safety_settings: Any,
) -> tuple:
    """Build a stable, hashable cache key for a `GenerativeModel` configuration."""
    if system_instructions is None:
        si_key: Any = None
    elif isinstance(system_instructions, str):
        si_key = system_instructions
    else:
        si_key = tuple(system_instructions)
    # generation_config may contain nested dicts (response_schema) — json.dumps
    # with `default=repr` gives a stable string for hashing without requiring
    # all values to be JSON-serializable.
    gc_key = json.dumps(generation_config, sort_keys=True, default=repr)
    ss_key = repr(safety_settings)
    return (_current_loop_id(), model_name, si_key, gc_key, ss_key)


def _get_or_create_generative_model(
    *,
    model_name: str,
    system_instructions: list[str] | str | None,
    generation_config: dict,
    safety_settings: Any,
) -> GenerativeModel:
    """Return a cached `GenerativeModel` matching this config or construct one."""
    cache_key = _make_generative_model_cache_key(
        model_name=model_name,
        system_instructions=system_instructions,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    cached = _GENERATIVE_MODEL_CACHE.get(cache_key)
    if cached is not None:
        _GENERATIVE_MODEL_CACHE.move_to_end(cache_key)
        return cached

    model = GenerativeModel(
        model_name=model_name,
        system_instruction=system_instructions,
        generation_config=GenerationConfig.from_dict(generation_config),
        safety_settings=list(safety_settings),
    )
    _GENERATIVE_MODEL_CACHE[cache_key] = model
    if len(_GENERATIVE_MODEL_CACHE) > _GENERATIVE_MODEL_CACHE_MAX_SIZE:
        _GENERATIVE_MODEL_CACHE.popitem(last=False)  # evict LRU
    return model


class GeminiGenerativeLLM(BasicLLM):
    """
    A wrapper for the Gemini LLM that provides retry logic with exponential backoff and jitter for generating content.
    """

    def __init__(self, *,
                 system_instructions: list[str] | str | None = None,
                 config: LLMConfig = LLMConfig()):
        super().__init__(config=config)

        self._model = _get_or_create_generative_model(
            model_name=config.language_model_name,
            system_instructions=system_instructions,
            generation_config=config.generation_config,
            safety_settings=config.safety_settings,
        )
        # noinspection PyProtectedMember
        self._resource_name = self._model._prediction_resource_name  # pylint: disable=protected-access

    async def internal_generate_content(self, llm_input: LLMInput | str) -> LLMResponse:
        contents = llm_input if isinstance(llm_input, str) else [
            Content(role=turn.role, parts=[Part.from_text(turn.content)]) for turn in llm_input.turns]
        response = await self._model.generate_content_async(contents=contents)
        return LLMResponse(text=response.text,
                           prompt_token_count=response.usage_metadata.prompt_token_count,
                           response_token_count=response.usage_metadata.candidates_token_count)


class PalmTextGenerativeLLM(BasicLLM):
    """
    A wrapper for the Palm2 Text generation model that provides retry logic with exponential backoff and jitter for
    generating content.
    """

    def __init__(self, *, system_instructions: list[str] | str | None = None, config: LLMConfig = LLMConfig()):
        super().__init__(config=config)
        self._model = TextGenerationModel.from_pretrained("text-bison@002")
        self._params = config.generation_config
        self._system_instructions = system_instructions

    async def internal_generate_content(self, llm_input: LLMInput | str) -> LLMResponse:
        contents = llm_input if isinstance(llm_input, str) else "Current conversation:\n" + "\n".join(
            [f"{turn.role}: {turn.content}" for turn in llm_input.turns])
        prompt = contents
        if self._system_instructions:
            prompt = self._system_instructions + "\n" + contents
        response = await self._model.predict_async(prompt=prompt, **self._params)
        token_meta_data = response.raw_prediction_response.metadata['tokenMetadata']
        return LLMResponse(
            text=response.text,
            prompt_token_count=token_meta_data['inputTokenCount']['totalTokens'],
            response_token_count=token_meta_data['outputTokenCount']['totalTokens']
        )
