"""
Bridge LLM — generates a single conversational sentence to transition the user
from one work-type bucket to the next during experience collection.

Replaces the static `messages.json:askAboutType` template (which read like a
mail-merge form: "Have you had any experiences that include: {experience_type}?")
with a freshly generated sentence that varies by user, work type, and locale.

Failure mode: returns None on any error. Callers must fall back to the existing
i18n template so the conversation never breaks if Vertex hiccups.
"""
import logging
from textwrap import dedent
from typing import Optional

from pydantic import BaseModel, Field

from app.agent.config import AgentsConfig
from app.agent.prompt_template import get_language_style
from app.agent.prompt_template.agent_prompt_template import STD_AGENT_CHARACTER
from common_libs.llm.generative_models import GeminiGenerativeLLM
from common_libs.llm.models_utils import (
    JSON_GENERATION_CONFIG,
    LLMConfig,
    MODERATE_TEMPERATURE_GENERATION_CONFIG,
)
from common_libs.llm.schema_builder import with_response_schema


class BridgeLLMResponse(BaseModel):
    bridge: str = Field(
        ...,
        description=(
            "ONE short conversational sentence that introduces the next work-type "
            "topic to the user. Must read as natural spoken language. Must NOT use "
            "colon-list phrasing such as 'experiences that include:'. Must NOT "
            "enumerate items as a list. Must be in the active conversation language."
        ),
    )


def _build_system_instructions() -> str:
    return dedent(
        """\
        You write a single conversational sentence to bridge between two topics in
        an ongoing conversation. The user has just finished talking about one type
        of work and you need to gently introduce the next type to ask about.

        Hard rules:
        - Output exactly ONE sentence in the active conversation language.
        - Do NOT use colon-list phrasing like "experiences that include:" or
          "Have you had experiences that include:".
        - Do NOT enumerate the activities as a comma-separated list parroted from
          the description. Phrase the next topic naturally.
        - Do NOT acknowledge the previous topic at length — at most a brief
          one- or two-word handoff like "Now," or "Next,".
        - Do NOT add greetings, closings, or filler.
        - Keep it warm and friendly but concise.
        """
    ) + "\n" + STD_AGENT_CHARACTER + "\n" + get_language_style(with_locale=True, for_json_output=True)


_BRIDGE_GENERATION_CONFIG = (
    MODERATE_TEMPERATURE_GENERATION_CONFIG
    | JSON_GENERATION_CONFIG
    | with_response_schema(BridgeLLMResponse)
    | {"max_output_tokens": 200}
)


async def generate_bridge_to_work_type(
    *,
    next_work_type_description: str,
    last_agent_message: str,
    logger: logging.Logger,
) -> Optional[str]:
    """
    Generate a single transition sentence to ask the user about the next work type.

    :param next_work_type_description: A localized description of the next bucket
        (e.g. "running my own business, doing freelance or contract work").
    :param last_agent_message: The reply the conversation LLM just produced this
        turn. Used to ensure the bridge tone matches.
    :param logger: agent logger.
    :return: A single sentence on success; None on any failure (caller must fall
        back to the i18n template).
    """
    try:
        llm = GeminiGenerativeLLM(
            system_instructions=_build_system_instructions(),
            config=LLMConfig(
                language_model_name=AgentsConfig.default_model,
                generation_config=_BRIDGE_GENERATION_CONFIG,
            ),
        )
        prompt = dedent(
            f"""\
            The conversation just produced this reply to the user:
                "{last_agent_message}"

            The next topic to gently introduce is described as:
                {next_work_type_description}

            Write ONE conversational sentence that hands off to that next topic.
            Remember: no colon-lists, no enumeration, no template phrasing.
            """
        )
        response = await llm.generate_content(llm_input=prompt)
        text = (response.text or "").strip()
        if not text:
            logger.warning("Bridge LLM returned empty text; falling back to template.")
            return None
        parsed = BridgeLLMResponse.model_validate_json(text)
        bridge = parsed.bridge.strip()
        if not bridge:
            logger.warning("Bridge LLM returned empty bridge field; falling back to template.")
            return None
        return bridge
    except Exception as err:
        logger.warning(
            "Bridge LLM failed (%s); falling back to askAboutType template.", err
        )
        return None
