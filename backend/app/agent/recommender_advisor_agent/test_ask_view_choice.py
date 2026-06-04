"""
Tests for the "ask careers / jobs / both" flow.

At INTRO the recommender asks whether the user wants career paths, job openings, or both;
the PRESENT phase classifies that answer once and renders the chosen view. These tests cover
the INTRO prompt + flag, the keyword fallback classifier, and the PRESENT gate's routing.

Run with:
    poetry run pytest app/agent/recommender_advisor_agent/test_ask_view_choice.py -v
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import ConversationPhase, Node2VecRecommendations
from app.agent.recommender_advisor_agent.phase_handlers.intro_handler import IntroPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.present_handler import PresentPhaseHandler


def _state() -> RecommenderAdvisorAgentState:
    return RecommenderAdvisorAgentState(session_id=1, youth_id="y1")


def _present_handler() -> PresentPhaseHandler:
    # intent_classifier=None so _classify_view_choice uses the keyword fallback (no LLM).
    return PresentPhaseHandler(
        conversation_llm=MagicMock(),
        conversation_caller=MagicMock(),
        intent_classifier=None,
    )


@pytest.mark.asyncio
async def test_intro_asks_view_choice_and_sets_flag():
    state = _state()
    state.recommendations = Node2VecRecommendations(youth_id="y1")  # skip generation
    handler = IntroPhaseHandler(
        conversation_llm=MagicMock(),
        conversation_caller=MagicMock(),
        recommendation_interface=AsyncMock(),
    )
    response, _ = await handler.handle("", state, MagicMock())

    assert state.awaiting_view_choice is True
    assert state.conversation_phase == ConversationPhase.PRESENT_RECOMMENDATIONS
    msg = response.message.lower()
    assert "career" in msg and "job" in msg and "both" in msg


def test_keyword_view_choice_mapping():
    f = PresentPhaseHandler._keyword_view_choice
    assert f("show me jobs") == "jobs"
    assert f("any openings I can apply to?") == "jobs"
    assert f("career paths please") == "careers"
    assert f("the occupations") == "careers"
    assert f("both") == "both"
    assert f("everything") == "both"          # unclear -> both
    assert f("") == "both"                     # empty -> both
    assert f("jobs and career paths") == "both"  # mentions both -> both


@pytest.mark.asyncio
async def test_classify_view_choice_defaults_to_both_on_junk_intent():
    handler = _present_handler()
    bad_intent = MagicMock()
    bad_intent.intent = "nonsense"
    handler._intent_classifier = MagicMock()
    handler._intent_classifier.classify_view_choice = AsyncMock(return_value=(bad_intent, []))
    # Empty user_input -> keyword fallback -> "both"
    view, _ = await handler._classify_view_choice("", MagicMock())
    assert view == "both"


@pytest.mark.asyncio
async def test_gate_routes_to_jobs_view_and_clears_flag():
    handler = _present_handler()
    sentinel = ("JOBS_RESPONSE", [])
    handler._present_jobs_view = AsyncMock(return_value=sentinel)
    handler._present_both_view = AsyncMock(return_value=("BOTH", []))
    state = _state()
    state.recommendations = Node2VecRecommendations(youth_id="y1")
    state.awaiting_view_choice = True

    response, _ = await handler.handle("show me jobs", state, MagicMock())

    assert response == "JOBS_RESPONSE"
    assert state.awaiting_view_choice is False
    assert state.recommendation_view == "jobs"
    handler._present_jobs_view.assert_awaited_once()
    handler._present_both_view.assert_not_awaited()


@pytest.mark.asyncio
async def test_gate_routes_to_both_view():
    handler = _present_handler()
    handler._present_jobs_view = AsyncMock(return_value=("JOBS", []))
    handler._present_both_view = AsyncMock(return_value=("BOTH_RESPONSE", []))
    state = _state()
    state.recommendations = Node2VecRecommendations(youth_id="y1")
    state.awaiting_view_choice = True

    response, _ = await handler.handle("both please", state, MagicMock())

    assert response == "BOTH_RESPONSE"
    assert state.recommendation_view == "both"
    handler._present_both_view.assert_awaited_once()


@pytest.mark.asyncio
async def test_gate_careers_falls_through_to_occupation_presentation():
    handler = _present_handler()
    handler._present_jobs_view = AsyncMock()
    handler._present_both_view = AsyncMock()
    state = _state()
    # No occupations and no opportunities -> occupation path returns the no-recs message
    # without calling the LLM, so we can assert fall-through cheaply.
    state.recommendations = Node2VecRecommendations(youth_id="y1")
    state.awaiting_view_choice = True

    response, _ = await handler.handle("career paths", state, MagicMock())

    assert state.awaiting_view_choice is False
    assert state.recommendation_view == "careers"
    handler._present_jobs_view.assert_not_awaited()
    handler._present_both_view.assert_not_awaited()
    assert response.finished is False
