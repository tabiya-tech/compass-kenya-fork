"""
Unit tests for the bridge LLM module. Mocks GeminiGenerativeLLM so no real
Vertex AI calls are made.
"""
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.collect_experiences_agent._bridge_llm import (
    BridgeLLMResponse,
    generate_bridge_to_work_type,
)
from app.context_vars import user_language_ctx_var
from app.i18n.types import Locale
from common_libs.llm.models_utils import LLMResponse


@pytest.fixture(autouse=True)
def _set_locale_context():
    """Production wires the locale per request via service.py:222. Unit tests
    must do the same so prompt builders that call get_i18n_manager().get_locale()
    don't raise LookupError on the missing context var."""
    token = user_language_ctx_var.set(Locale.EN_US)
    try:
        yield
    finally:
        user_language_ctx_var.reset(token)


def _make_llm_mock(returned_text: str) -> MagicMock:
    instance = MagicMock()
    instance.generate_content = AsyncMock(
        return_value=LLMResponse(
            text=returned_text,
            prompt_token_count=10,
            response_token_count=10,
        )
    )
    return instance


@pytest.mark.asyncio
async def test_bridge_llm_returns_parsed_bridge_on_success():
    expected_bridge = "Now let's talk about self-employment — have you ever run your own business?"
    payload = json.dumps({"bridge": expected_bridge})

    with patch(
        "app.agent.collect_experiences_agent._bridge_llm.GeminiGenerativeLLM",
        return_value=_make_llm_mock(payload),
    ):
        result = await generate_bridge_to_work_type(
            next_work_type_description="running my own business, doing freelance or contract work",
            last_agent_message="Got it. Anything else?",
            logger=logging.getLogger(__name__),
        )

    assert result == expected_bridge


@pytest.mark.asyncio
async def test_bridge_llm_returns_none_on_empty_response():
    with patch(
        "app.agent.collect_experiences_agent._bridge_llm.GeminiGenerativeLLM",
        return_value=_make_llm_mock(""),
    ):
        result = await generate_bridge_to_work_type(
            next_work_type_description="running my own business, doing freelance or contract work",
            last_agent_message="Got it. Anything else?",
            logger=logging.getLogger(__name__),
        )

    assert result is None


@pytest.mark.asyncio
async def test_bridge_llm_returns_none_on_invalid_json():
    with patch(
        "app.agent.collect_experiences_agent._bridge_llm.GeminiGenerativeLLM",
        return_value=_make_llm_mock("not json at all"),
    ):
        result = await generate_bridge_to_work_type(
            next_work_type_description="running my own business, doing freelance or contract work",
            last_agent_message="Got it. Anything else?",
            logger=logging.getLogger(__name__),
        )

    assert result is None


@pytest.mark.asyncio
async def test_bridge_llm_returns_none_when_bridge_field_is_blank():
    payload = json.dumps({"bridge": "   "})

    with patch(
        "app.agent.collect_experiences_agent._bridge_llm.GeminiGenerativeLLM",
        return_value=_make_llm_mock(payload),
    ):
        result = await generate_bridge_to_work_type(
            next_work_type_description="running my own business, doing freelance or contract work",
            last_agent_message="Got it.",
            logger=logging.getLogger(__name__),
        )

    assert result is None


@pytest.mark.asyncio
async def test_bridge_llm_returns_none_when_llm_raises():
    failing_llm = MagicMock()
    failing_llm.generate_content = AsyncMock(side_effect=RuntimeError("vertex went down"))

    with patch(
        "app.agent.collect_experiences_agent._bridge_llm.GeminiGenerativeLLM",
        return_value=failing_llm,
    ):
        result = await generate_bridge_to_work_type(
            next_work_type_description="running my own business, doing freelance or contract work",
            last_agent_message="Got it.",
            logger=logging.getLogger(__name__),
        )

    assert result is None


def test_bridge_response_model_rejects_missing_field():
    with pytest.raises(Exception):
        BridgeLLMResponse.model_validate_json("{}")
