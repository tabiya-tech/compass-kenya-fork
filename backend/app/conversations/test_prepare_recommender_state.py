"""
Unit tests for ConversationService._prepare_recommender_state_if_needed().

These cover the runtime path the earlier BWS handoff fix missed: the matching inputs
(skills, preference vector, BWS bundle) must be populated only once the conversation
reaches the recommender stage, and refreshed from source on each prep — NOT latched on an
early turn. The previous fix's unit tests covered only the pure helper, so this gating /
refresh path went unverified and shipped broken (the recommender forwarded empty BWS,
empty skills, and a stale default preference vector to the matching service).

Run with:
    poetry run pytest app/conversations/test_prepare_recommender_state.py -v
"""

import logging
from unittest.mock import AsyncMock

import pytest

from app.application_state import ApplicationState
from app.agent.agent_director.abstract_agent_director import CounselingSubPhase
from app.conversations.phase_state_machine import JourneyPhase
from app.conversations.service import ConversationService
from app.agent.preference_elicitation_agent.types import PreferenceVector


def _hb_entry(mean: float) -> dict:
    return {"mean": mean, "sd": 0.3, "ci_low": mean - 0.6, "ci_high": mean + 0.6, "rank": 1}


def _service() -> ConversationService:
    # Bypass the heavy __init__; the method under test only needs _logger and, on the
    # step-skip path, _user_recommendations_service.
    svc = ConversationService.__new__(ConversationService)
    svc._logger = logging.getLogger("test_prepare_recommender_state")
    svc._user_recommendations_service = AsyncMock()
    svc._user_recommendations_service.get_by_user_id = AsyncMock(return_value=None)
    return svc


def _state(sub_phase: CounselingSubPhase) -> ApplicationState:
    state = ApplicationState.new_state(session_id=1)
    state.agent_director_state.counseling_sub_phase = sub_phase
    # Pre-set a non-empty skills_vector so these tests isolate the prefs/BWS path from
    # the skills extractor (skills extraction is exercised live).
    state.recommender_advisor_agent_state.skills_vector = {"skills": [{"id": "x"}], "total_experiences": 1}
    pref = state.preference_elicitation_agent_state
    pref.hb_scores = {"4.A.2.b.1": _hb_entry(2.1), "4.A.3.a.1": _hb_entry(-1.4)}
    pref.hb_ranking = ["4.A.2.b.1", "4.A.3.a.1"]
    pref.preference_vector = PreferenceVector(confidence_score=0.64, work_environment_importance=0.72)
    return state


@pytest.mark.asyncio
async def test_gate_closed_before_recommender_stage_leaves_inputs_empty():
    # During preference elicitation the recommender inputs must NOT be populated yet.
    state = _state(CounselingSubPhase.PREFERENCE_ELICITATION)
    rec = state.recommender_advisor_agent_state
    await _service()._prepare_recommender_state_if_needed(state, "youth_1")
    assert rec.bws_scores is None
    assert rec.top_10_bws is None
    assert rec.preference_vector is None


@pytest.mark.asyncio
async def test_recommender_stage_forwards_hb_means_and_refreshes_prefs():
    state = _state(CounselingSubPhase.RECOMMENDER_ADVISOR)
    rec = state.recommender_advisor_agent_state
    await _service()._prepare_recommender_state_if_needed(state, "youth_1")
    # continuous HB posterior means under the bws_scores name (not integer counts)
    assert rec.bws_scores == {"4.A.2.b.1": 2.1, "4.A.3.a.1": -1.4}
    assert rec.top_10_bws == ["4.A.2.b.1", "4.A.3.a.1"]
    # preference vector refreshed from the preference-elicitation agent
    assert rec.preference_vector is not None
    assert rec.preference_vector.confidence_score == 0.64


@pytest.mark.asyncio
async def test_refresh_overwrites_stale_counts_and_default_prefs():
    # Simulate a session the old latch corrupted: RAAS already holds integer counts and a
    # stale default preference vector. A prep in the recommender stage must replace both.
    state = _state(CounselingSubPhase.RECOMMENDER_ADVISOR)
    rec = state.recommender_advisor_agent_state
    rec.bws_scores = {"4.A.2.b.1": 1.0, "4.A.3.a.1": -1.0}  # integer counts
    rec.top_10_bws = ["4.A.2.b.1"]
    rec.preference_vector = PreferenceVector(confidence_score=0.0)  # stale default
    await _service()._prepare_recommender_state_if_needed(state, "youth_1")
    assert rec.bws_scores == {"4.A.2.b.1": 2.1, "4.A.3.a.1": -1.4}  # replaced by HB means
    assert rec.preference_vector.confidence_score == 0.64  # refreshed to final


@pytest.mark.asyncio
async def test_step_skip_to_recommendation_opens_gate():
    # The returning-user step-skip path must also populate the inputs.
    state = _state(CounselingSubPhase.PREFERENCE_ELICITATION)
    state.agent_director_state.skip_to_phase = JourneyPhase.RECOMMENDATION
    rec = state.recommender_advisor_agent_state
    await _service()._prepare_recommender_state_if_needed(state, "youth_1")
    assert rec.bws_scores == {"4.A.2.b.1": 2.1, "4.A.3.a.1": -1.4}
