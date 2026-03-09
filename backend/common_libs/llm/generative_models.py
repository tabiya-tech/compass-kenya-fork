import logging
import traceback
import inspect

from vertexai.generative_models import GenerativeModel, Content, Part, GenerationConfig
from vertexai.language_models import TextGenerationModel

from common_libs.llm.models_utils import LLMConfig, LLMInput, LLMResponse, BasicLLM, LLMStreamingCallback

logger = logging.getLogger(__name__)


async def _maybe_await(callback_result):
    if inspect.isawaitable(callback_result):
        await callback_result


class GeminiGenerativeLLM(BasicLLM):
    """
    A wrapper for the Gemini LLM that provides retry logic with exponential backoff and jitter for generating content.
    """

    def __init__(self, *,
                 system_instructions: list[str] | str | None = None,
                 config: LLMConfig = LLMConfig()):
        super().__init__(config=config)

        self._model = GenerativeModel(model_name=config.language_model_name,
                                      system_instruction=system_instructions,
                                      generation_config=GenerationConfig.from_dict(config.generation_config),
                                      safety_settings=list(config.safety_settings)
                                      )
        # noinspection PyProtectedMember
        self._resource_name = self._model._prediction_resource_name  # pylint: disable=protected-access

    @staticmethod
    def _build_contents(llm_input: LLMInput | str):
        return llm_input if isinstance(llm_input, str) else [
            Content(role=turn.role, parts=[Part.from_text(turn.content)]) for turn in llm_input.turns]

    async def internal_generate_content(self, llm_input: LLMInput | str) -> LLMResponse:
        contents = self._build_contents(llm_input)
        response = await self._model.generate_content_async(contents=contents)
        return LLMResponse(text=response.text,
                           prompt_token_count=response.usage_metadata.prompt_token_count,
                           response_token_count=response.usage_metadata.candidates_token_count)

    async def internal_stream_content(
        self,
        llm_input: LLMInput | str,
        on_chunk: LLMStreamingCallback | None = None,
    ) -> LLMResponse:
        contents = self._build_contents(llm_input)
        stream = await self._model.generate_content_async(contents=contents, stream=True)

        chunks: list[str] = []
        final_response = None
        async for response in stream:
            final_response = response
            text = response.text or ""
            if not text:
                continue
            chunks.append(text)
            if on_chunk is not None:
                await _maybe_await(on_chunk(text))

        if final_response is None:
            return LLMResponse(text="", prompt_token_count=0, response_token_count=0)

        usage_metadata = final_response.usage_metadata
        return LLMResponse(
            text="".join(chunks),
            prompt_token_count=usage_metadata.prompt_token_count if usage_metadata else 0,
            response_token_count=usage_metadata.candidates_token_count if usage_metadata else 0,
        )


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

    async def internal_stream_content(
        self,
        llm_input: LLMInput | str,
        on_chunk: LLMStreamingCallback | None = None,
    ) -> LLMResponse:
        response = await self.internal_generate_content(llm_input)
        if on_chunk is not None and response.text:
            await _maybe_await(on_chunk(response.text))
        return response
