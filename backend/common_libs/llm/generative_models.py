import asyncio
import logging
from typing import Any, Type

from google import genai
from google.genai.types import GroundingChunk, GroundingChunkWeb, GroundingMetadata
from pydantic import BaseModel

from common_libs.llm.models_utils import LLMConfig, LLMInput, LLMResponse, BasicLLM, get_response_schema

logger = logging.getLogger(__name__)

_GENERATION_PARAMETERS = ("temperature", "top_p", "max_output_tokens", "seed", "stop_sequences")
"""The generation parameters supported by the Interactions API, the other keys of a generation config are ignored."""

_ROLE_TO_STEP_TYPE = {
    "user": "user_input",
    "model": "model_output",
}

_clients: dict[tuple[str, int], genai.Client] = {}

_REQUEST_TIMEOUT_MS = 180_000
"""
The timeout of a request to the API. Without it the client waits forever on a connection that was dropped silently,
the timeout error is transient, so the call is retried (see `common_libs.retry.is_retryable_error`).
"""


def get_genai_client(location: str) -> genai.Client:
    """
    Get the google-genai client for Vertex AI in the given location.
    The project and the credentials are taken from the environment (GOOGLE_APPLICATION_CREDENTIALS).

    A client is kept per location and per event loop, as its async sessions are bound to the event loop they were
    created in.
    """
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = 0
    key = (location, loop_id)
    client = _clients.get(key)
    if client is None:
        client = genai.Client(vertexai=True, location=location,
                              http_options=genai.types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS))
        _clients[key] = client
    return client


def llm_input_to_interactions_input(llm_input: LLMInput | str) -> str | list[dict[str, Any]]:
    """
    Convert the input to the input of the Interactions API: a string, or the conversation turns as steps.
    The conversation history is always sent with the input (stateless), it is not stored on the server.

    The API rejects an empty text, so an empty input is sent as a blank text, and a turn with an empty content
    as a step without content (which keeps the order of the turns).
    """
    if isinstance(llm_input, str):
        return llm_input if llm_input else " "
    steps: list[dict[str, Any]] = []
    for turn in llm_input.turns:
        step: dict[str, Any] = {"type": _ROLE_TO_STEP_TYPE.get(turn.role, "user_input")}
        if turn.content:
            step["content"] = [{"type": "text", "text": turn.content}]
        steps.append(step)
    return steps


def response_format_for(generation_config: dict, response_schema: Type[BaseModel] | None) -> dict[str, Any] | None:
    """
    The response format of the Interactions API for the structured output, or None for plain text.
    """
    if response_schema is not None:
        return {"type": "text", "mime_type": "application/json", "schema": response_schema.model_json_schema()}
    if generation_config.get("response_mime_type") == "application/json":
        return {"type": "text", "mime_type": "application/json"}
    return None


def extract_grounding_metadata(interaction) -> GroundingMetadata | None:
    """
    Build the grounding metadata of a Google Search grounded interaction:
      - the web search queries, from the google search call steps.
      - the sources, from the url citations of the output text.
    :return: The grounding metadata, or None if the interaction did not search the web.
    """
    queries: list[str] = []
    chunks: list[GroundingChunk] = []
    seen_urls: set[str] = set()
    for step in getattr(interaction, "steps", None) or []:
        if step.type == "google_search_call":
            arguments = getattr(step, "arguments", None)
            queries.extend(getattr(arguments, "queries", None) or [])
        elif step.type == "model_output":
            for content in step.content or []:
                for annotation in getattr(content, "annotations", None) or []:
                    url = getattr(annotation, "url", None)
                    if getattr(annotation, "type", None) == "url_citation" and url and url not in seen_urls:
                        seen_urls.add(url)
                        chunks.append(GroundingChunk(web=GroundingChunkWeb(uri=url, title=getattr(annotation, "title", None))))
    if not queries and not chunks:
        return None
    return GroundingMetadata(web_search_queries=queries, grounding_chunks=chunks)


class GeminiGenerativeLLM(BasicLLM):
    """
    A wrapper for the Gemini LLM, on the google-genai Interactions API of Vertex AI,
    that provides retry logic with exponential backoff and jitter for generating content.
    """

    def __init__(self, *,
                 system_instructions: list[str] | str | None = None,
                 config: LLMConfig = LLMConfig(),
                 tools: list[dict[str, Any]] | None = None):
        """
        :param system_instructions: The system instructions of the model.
        :param config: The configuration of the LLM.
        :param tools: The tools the model can use, e.g. [{"type": "google_search"}] to ground the response on a web search.
        """
        super().__init__(config=config)
        self._system_instruction = "\n".join(system_instructions) if isinstance(system_instructions, list) else system_instructions
        self._safety_settings = [dict(setting) for setting in config.safety_settings]
        self._location = config.location
        self._tools = tools
        self._resource_name = f"locations/{config.location}/publishers/google/models/{self._model_name}"

    async def internal_generate_content(self, llm_input: LLMInput | str,
                                        response_schema: Type[BaseModel] | None = None) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self._model_name,
            "input": llm_input_to_interactions_input(llm_input),
            "generation_config": {key: value for key, value in self.generation_config.items() if key in _GENERATION_PARAMETERS},
            "safety_settings": self._safety_settings,
            "store": False,
        }
        if self._system_instruction:
            request["system_instruction"] = self._system_instruction
        response_format = response_format_for(self.generation_config, get_response_schema(self.generation_config, response_schema))
        if response_format is not None:
            request["response_format"] = response_format
        if self._tools:
            request["tools"] = self._tools

        interaction = await get_genai_client(self._location).aio.interactions.create(**request)
        usage = interaction.usage
        return LLMResponse(
            text=interaction.output_text or "",
            prompt_token_count=(usage.total_input_tokens or 0) if usage else 0,
            response_token_count=(usage.total_output_tokens or 0) if usage else 0,
            grounding_metadata=extract_grounding_metadata(interaction) if self._tools else None,
        )
