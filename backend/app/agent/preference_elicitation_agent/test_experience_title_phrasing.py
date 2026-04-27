"""
Tests for the experience-title phrasing fix in PreferenceElicitationAgent.

Verifies that when experience_title is a place name (e.g. "cooking school")
the agent does NOT produce "working as cooking school" in either the
deterministic fallback (turn 1) or the LLM-generated question (turn 2).

Deterministic tests (A–E): LLM calls mocked out, safe for CI.
LLM integration test (F): marked llm_integration, skipped in CI.
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.agent.agent_types import AgentInput
from app.agent.experience import WorkType, Timeline
from app.agent.experience.experience_entity import ExperienceEntity
from app.agent.preference_elicitation_agent.agent import PreferenceElicitationAgent
from app.agent.preference_elicitation_agent.state import PreferenceElicitationAgentState
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
    ConversationTurn,
)

BAD_PHRASE = "working as cooking school"

# Patch target for the two LLM-touching methods that fire on every turn-1 call.
# _extract_user_context  — awaited directly in execute() before phase dispatch.
# _prewarm_next_vignette — spawned via asyncio.create_task in _handle_intro_phase.
_EXTRACT_CTX = "app.agent.preference_elicitation_agent.agent.PreferenceElicitationAgent._extract_user_context"
_PREWARM     = "app.agent.preference_elicitation_agent.agent.PreferenceElicitationAgent._prewarm_next_vignette"


def _make_agent(experiences: list[ExperienceEntity]) -> tuple[PreferenceElicitationAgent, PreferenceElicitationAgentState]:
    agent = PreferenceElicitationAgent()
    state = PreferenceElicitationAgentState(
        session_id=1,
        initial_experiences_snapshot=experiences,
        use_db6_for_fresh_data=False,
    )
    agent.set_state(state)
    return agent, state


def _make_context(history: ConversationHistory) -> ConversationContext:
    return ConversationContext(all_history=history, history=history, summary="")


def _cooking_school_with_company() -> list[ExperienceEntity]:
    return [ExperienceEntity(
        uuid="exp-a",
        experience_title="cooking school",
        normalized_experience_title=None,
        company="Mama Rocks Culinary Institute",
        location="Nairobi",
        timeline=Timeline(start="2021", end="2023"),
        work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT,
    )]


def _cooking_school_with_normalized_title() -> list[ExperienceEntity]:
    return [ExperienceEntity(
        uuid="exp-b",
        experience_title="cooking school",
        normalized_experience_title="Cook",
        company=None,
        location="Nairobi",
        timeline=Timeline(start="2021", end="2023"),
        work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT,
    )]


def _cooking_school_no_company_no_normalized() -> list[ExperienceEntity]:
    return [ExperienceEntity(
        uuid="exp-e",
        experience_title="cooking school",
        normalized_experience_title=None,
        company=None,
        location="Nairobi",
        timeline=Timeline(start="2021", end="2023"),
        work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT,
    )]


def _proper_job_title() -> list[ExperienceEntity]:
    return [ExperienceEntity(
        uuid="exp-c",
        experience_title="Mathematics Teacher",
        normalized_experience_title=None,
        company="Alliance High School",
        location="Kikuyu",
        timeline=Timeline(start="2018", end="2023"),
        work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT,
    )]


# ---------------------------------------------------------------------------
# Deterministic tests — LLM mocked, safe for CI
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch(_PREWARM, new_callable=AsyncMock)
@patch(_EXTRACT_CTX, new_callable=AsyncMock)
async def test_fallback_uses_company_when_title_is_place(mock_extract, mock_prewarm):
    """Fallback anchors to company name, avoiding 'working as cooking school'."""
    agent, _ = _make_agent(_cooking_school_with_company())
    out = await agent.execute(
        AgentInput(message="", is_artificial=True),
        _make_context(ConversationHistory()),
    )
    assert BAD_PHRASE not in out.message_for_user.lower()
    assert "working at Mama Rocks Culinary Institute" in out.message_for_user


@pytest.mark.asyncio
@patch(_PREWARM, new_callable=AsyncMock)
@patch(_EXTRACT_CTX, new_callable=AsyncMock)
async def test_fallback_uses_normalized_title_when_no_company(mock_extract, mock_prewarm):
    """Fallback uses normalized (ESCO) title when company is absent."""
    agent, _ = _make_agent(_cooking_school_with_normalized_title())
    out = await agent.execute(
        AgentInput(message="", is_artificial=True),
        _make_context(ConversationHistory()),
    )
    assert BAD_PHRASE not in out.message_for_user.lower()
    assert "working as Cook" in out.message_for_user


@pytest.mark.asyncio
@patch(_PREWARM, new_callable=AsyncMock)
@patch(_EXTRACT_CTX, new_callable=AsyncMock)
async def test_fallback_generic_when_no_company_and_no_normalized_title(mock_extract, mock_prewarm):
    """Degenerate case: no company, no normalized title — must not produce 'working as cooking school'."""
    agent, _ = _make_agent(_cooking_school_no_company_no_normalized())
    out = await agent.execute(
        AgentInput(message="", is_artificial=True),
        _make_context(ConversationHistory()),
    )
    assert BAD_PHRASE not in out.message_for_user.lower()
    assert "cooking school" not in out.message_for_user.lower()


@pytest.mark.asyncio
@patch(_PREWARM, new_callable=AsyncMock)
@patch(_EXTRACT_CTX, new_callable=AsyncMock)
async def test_fallback_unchanged_for_proper_job_title(mock_extract, mock_prewarm):
    """Regression: proper job title with company still produces a valid message."""
    agent, _ = _make_agent(_proper_job_title())
    out = await agent.execute(
        AgentInput(message="", is_artificial=True),
        _make_context(ConversationHistory()),
    )
    assert BAD_PHRASE not in out.message_for_user.lower()
    assert "Alliance High School" in out.message_for_user


# ---------------------------------------------------------------------------
# LLM integration test — requires real API credentials, skipped in CI
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.llm_integration
async def test_llm_turn2_does_not_produce_bad_phrasing():
    """
    LLM turn 2: the guard instruction in the prompt prevents 'working as cooking school'.
    Requires GOOGLE_API_KEY (or equivalent). Skipped in CI via -m 'not llm_integration'.
    """
    agent, state = _make_agent(_cooking_school_with_company())
    history = ConversationHistory()

    # Turn 1 (fallback — no LLM)
    out1 = await agent.execute(
        AgentInput(message="", is_artificial=True),
        _make_context(history),
    )
    history.turns.append(ConversationTurn(index=0, input=AgentInput(message=""), output=out1))

    # Turn 2 (real LLM call)
    out2 = await agent.execute(
        AgentInput(message="I really enjoyed the hands-on cooking.", is_artificial=False),
        _make_context(history),
    )
    assert BAD_PHRASE not in out2.message_for_user.lower()
