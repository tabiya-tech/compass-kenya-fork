"""
Tests for opportunity (job opening) parity with occupations.

After the jobs/both view, a user follow-up about a specific opening should make that
opportunity the current focus and flow through the same explore → concerns → action
machinery as occupations. These tests cover the jobs-followup routing gate, the
opportunity exploration branch, the jobs-followup classifier prompt, and the
show_opportunities mis-route fix.

Run with:
    poetry run pytest app/agent/recommender_advisor_agent/test_opportunity_parity.py -v
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    Node2VecRecommendations,
    OpportunityRecommendation,
    OccupationRecommendation,
)
from app.agent.recommender_advisor_agent.phase_handlers.present_handler import PresentPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.exploration_handler import ExplorationPhaseHandler
from app.agent.recommender_advisor_agent.llm_response_models import ConversationResponse
from app.conversation_memory.conversation_memory_types import ConversationContext


def _context() -> ConversationContext:
    """A real, empty conversation context for handlers that hit the prompt formatter."""
    return ConversationContext()


def _opp(uuid: str, title: str, rank: int = 1) -> OpportunityRecommendation:
    return OpportunityRecommendation(
        uuid=uuid,
        originUuid=uuid,
        rank=rank,
        opportunity_title=title,
        location="Nairobi",
    )


def _recs(opps=None, occs=None) -> Node2VecRecommendations:
    return Node2VecRecommendations(
        user_id="y1",
        opportunity_recommendations=opps or [],
        occupation_recommendations=occs or [],
    )


def _state_after_jobs_view(opps) -> RecommenderAdvisorAgentState:
    state = RecommenderAdvisorAgentState(session_id=1, youth_id="y1")
    state.recommendations = _recs(opps=opps)
    state.recommendation_view = "jobs"
    state.awaiting_view_choice = False
    state.presented_opportunities = [o.uuid for o in opps]
    return state


@pytest.mark.asyncio
async def test_jobs_followup_explore_opportunity_sets_focus_and_explores():
    """After a jobs view, 'tell me about #1' focuses that opening and goes to EXPLORATION."""
    opps = [_opp("opp_glovo", "Glovo Rider"), _opp("opp_safcom", "Safaricom Intern", rank=2)]
    state = _state_after_jobs_view(opps)

    intent = MagicMock()
    intent.intent = "explore_opportunity"
    intent.target_occupation_index = 1
    intent.target_recommendation_id = "opp_glovo"

    classifier = MagicMock()
    classifier.classify_jobs_followup = AsyncMock(return_value=(intent, []))

    exploration = MagicMock()
    exploration.handle = AsyncMock(return_value=("EXPLORE_RESPONSE", []))

    handler = PresentPhaseHandler(
        conversation_llm=MagicMock(),
        conversation_caller=MagicMock(),
        intent_classifier=classifier,
        exploration_handler=exploration,
    )

    response, _ = await handler.handle("tell me about the Glovo one", state, MagicMock())

    assert response == "EXPLORE_RESPONSE"
    assert state.current_focus_id == "opp_glovo"
    assert state.current_recommendation_type == "opportunity"
    assert state.conversation_phase == ConversationPhase.CAREER_EXPLORATION
    exploration.handle.assert_awaited_once()


def _opp_full(uuid="opp_glovo") -> OpportunityRecommendation:
    return OpportunityRecommendation(
        uuid=uuid,
        originUuid=uuid,
        rank=1,
        opportunity_title="Glovo Rider",
        location="Nairobi",
        contract_type="part_time",
        employer="Glovo",
        posting_url="https://jobs.example/glovo",
        justification="Matches your delivery experience",
    )


def test_build_opportunity_summary_uses_real_fields_no_fabrication():
    handler = ExplorationPhaseHandler(
        conversation_llm=MagicMock(), conversation_caller=MagicMock(),
    )
    summary = handler._build_opportunity_summary(_opp_full())
    assert "Glovo Rider" in summary
    assert "Nairobi" in summary
    assert "part_time" in summary or "part time" in summary.lower()
    assert "https://jobs.example/glovo" in summary
    # Must NOT instruct the LLM to invent day-to-day tasks the posting doesn't have.
    assert "generate realistic tasks" not in summary.lower()


@pytest.mark.asyncio
async def test_exploration_handler_explores_opportunity_focus():
    """An opportunity focus is explored via the shared machinery, not rejected as 'not an occupation'."""
    opp = _opp_full()
    state = RecommenderAdvisorAgentState(session_id=1, youth_id="y1")
    state.recommendations = _recs(opps=[opp])
    state.current_focus_id = opp.uuid
    state.current_recommendation_type = "opportunity"

    handler = ExplorationPhaseHandler(conversation_llm=MagicMock(), conversation_caller=MagicMock())
    handler._call_llm = AsyncMock(
        return_value=(ConversationResponse(reasoning="r", message="About Glovo Rider", finished=False), [])
    )

    response, _ = await handler.handle("tell me about this one", state, MagicMock())

    assert "couldn't find" not in response.message.lower()
    assert opp.uuid in state.explored_items
    handler._call_llm.assert_awaited_once()
    # The opportunity (not occupation) prompt path was used.
    prompt_arg = handler._call_llm.call_args.args[0]
    assert "JOB OPENING EXPLORATION" in prompt_arg


def test_jobs_followup_prompt_lists_openings_and_offers_explore_opportunity():
    from app.agent.recommender_advisor_agent.intent_classifier import IntentClassifier

    classifier = IntentClassifier(intent_caller=MagicMock())
    state = RecommenderAdvisorAgentState(session_id=1, youth_id="y1")
    opps = [_opp("opp_glovo", "Glovo Rider"), _opp("opp_safcom", "Safaricom Intern", rank=2)]
    state.recommendations = _recs(opps=opps)
    state.presented_opportunities = [o.uuid for o in opps]
    state.recommendation_view = "jobs"

    prompt = classifier._build_jobs_followup_phase_prompt("tell me about the Glovo one", state)

    assert "Glovo Rider" in prompt
    assert "opp_glovo" in prompt
    assert "explore_opportunity" in prompt


def test_jobs_followup_prompt_includes_careers_when_view_both():
    from app.agent.recommender_advisor_agent.intent_classifier import IntentClassifier

    classifier = IntentClassifier(intent_caller=MagicMock())
    state = RecommenderAdvisorAgentState(session_id=1, youth_id="y1")
    opps = [_opp("opp_glovo", "Glovo Rider")]
    occ = OccupationRecommendation(
        uuid="occ_da", originUuid="occ_da", rank=1,
        occupation_id="2511", occupation_code="2511", occupation="Data Analyst",
    )
    state.recommendations = _recs(opps=opps, occs=[occ])
    state.presented_opportunities = ["opp_glovo"]
    state.presented_occupations = ["occ_da"]
    state.recommendation_view = "both"

    prompt = classifier._build_jobs_followup_phase_prompt("the data analyst path", state)

    assert "Data Analyst" in prompt
    assert "explore_occupation" in prompt


@pytest.mark.asyncio
async def test_concerns_handler_records_opportunity_concern():
    """A concern about a job opening is recorded with item_type='opportunity'."""
    from app.agent.recommender_advisor_agent.phase_handlers.concerns_handler import ConcernsPhaseHandler
    from app.agent.recommender_advisor_agent.llm_response_models import ResistanceClassification

    opp = _opp_full()
    state = RecommenderAdvisorAgentState(session_id=1, youth_id="y1")
    state.recommendations = _recs(opps=[opp])
    state.current_focus_id = opp.uuid
    state.current_recommendation_type = "opportunity"

    resistance_caller = MagicMock()
    resistance_caller.call_llm = AsyncMock(return_value=(
        ResistanceClassification(reasoning="r", resistance_type="circumstantial",
                                 concern_summary="Location is far"), []))
    conv_caller = MagicMock()
    conv_caller.call_llm = AsyncMock(return_value=(
        ConversationResponse(reasoning="r", message="Let's talk about the commute", finished=False), []))

    handler = ConcernsPhaseHandler(
        conversation_llm=MagicMock(),
        conversation_caller=conv_caller,
        resistance_caller=resistance_caller,
        intent_classifier=None,  # skip routing; exercise core concern-recording path
    )

    response, _ = await handler.handle("the location is too far for me", state, _context())

    assert len(state.concerns_raised) == 1
    assert state.concerns_raised[0].item_type == "opportunity"
    assert state.concerns_raised[0].item_id == opp.uuid
    assert response.finished is False


@pytest.mark.asyncio
async def test_action_handler_records_opportunity_commitment():
    """Committing to apply to a job opening records an opportunity commitment with the job title."""
    from app.agent.recommender_advisor_agent.phase_handlers.action_handler import ActionPhaseHandler
    from app.agent.recommender_advisor_agent.llm_response_models import ActionExtractionResult

    opp = _opp_full()
    state = RecommenderAdvisorAgentState(session_id=1, youth_id="y1")
    state.recommendations = _recs(opps=[opp])
    state.current_focus_id = opp.uuid
    state.current_recommendation_type = "opportunity"
    state.conversation_phase = ConversationPhase.ACTION_PLANNING

    action_caller = MagicMock()
    action_caller.call_llm = AsyncMock(return_value=(
        ActionExtractionResult(reasoning="r", has_commitment=True, action_type="apply_to_job",
                               commitment_level="will_do_this_week", barriers_mentioned=[]), []))

    handler = ActionPhaseHandler(
        conversation_llm=MagicMock(),
        conversation_caller=MagicMock(),
        action_caller=action_caller,
        intent_classifier=None,   # skip menu routing; exercise extraction path
        wrapup_handler=None,      # fall back to acknowledgment
    )

    response, _ = await handler.handle("yes, I'll apply this week", state, _context())

    assert state.action_commitment is not None
    assert state.action_commitment.recommendation_type == "opportunity"
    assert state.action_commitment.recommendation_title == "Glovo Rider"
    assert state.conversation_phase == ConversationPhase.WRAPUP


@pytest.mark.asyncio
async def test_followup_show_opportunities_sets_jobs_view():
    """show_opportunities in FOLLOW_UP routes to the jobs view, not the occupation dump."""
    from app.agent.recommender_advisor_agent.phase_handlers.followup_handler import FollowupPhaseHandler

    state = RecommenderAdvisorAgentState(session_id=1, youth_id="y1")
    state.recommendations = _recs(opps=[_opp("opp_glovo", "Glovo Rider")])
    state.conversation_phase = ConversationPhase.FOLLOW_UP

    intent = MagicMock()
    intent.intent = "show_opportunities"
    intent.target_occupation_index = None
    intent.target_recommendation_id = None
    intent.requested_occupation_name = None

    classifier = MagicMock()
    classifier.classify_intent = AsyncMock(return_value=(intent, []))

    handler = FollowupPhaseHandler(
        conversation_llm=MagicMock(), conversation_caller=MagicMock(), intent_classifier=classifier,
    )

    response, _ = await handler.handle("show me the jobs", state, MagicMock())

    assert state.recommendation_view == "jobs"
    assert state.conversation_phase == ConversationPhase.PRESENT_RECOMMENDATIONS
    assert response.finished is False


@pytest.mark.asyncio
async def test_present_renders_jobs_when_view_jobs_and_none_presented():
    """PRESENT with recommendation_view='jobs' (e.g. routed from FOLLOW_UP) renders the jobs view."""
    state = RecommenderAdvisorAgentState(session_id=1, youth_id="y1")
    state.recommendations = _recs(
        opps=[_opp("opp_glovo", "Glovo Rider")],
        occs=[OccupationRecommendation(uuid="occ_da", originUuid="occ_da", rank=1,
                                       occupation_id="2511", occupation_code="2511", occupation="Data Analyst")],
    )
    state.recommendation_view = "jobs"
    state.awaiting_view_choice = False
    # nothing presented yet

    handler = PresentPhaseHandler(
        conversation_llm=MagicMock(), conversation_caller=MagicMock(), intent_classifier=MagicMock(),
    )
    handler._present_jobs_view = AsyncMock(return_value=("JOBS_RESPONSE", []))

    response, _ = await handler.handle("show me the jobs", state, MagicMock())

    assert response == "JOBS_RESPONSE"
    handler._present_jobs_view.assert_awaited_once()


@pytest.mark.asyncio
async def test_jobs_followup_explore_occupation_routes_to_occupation():
    """In a 'both' view, a follow-up about a career path still routes to occupation exploration."""
    occ = OccupationRecommendation(uuid="occ_da", originUuid="occ_da", rank=1,
                                   occupation_id="2511", occupation_code="2511", occupation="Data Analyst")
    state = RecommenderAdvisorAgentState(session_id=1, youth_id="y1")
    state.recommendations = _recs(opps=[_opp("opp_glovo", "Glovo Rider")], occs=[occ])
    state.recommendation_view = "both"
    state.awaiting_view_choice = False
    state.presented_opportunities = ["opp_glovo"]
    state.presented_occupations = ["occ_da"]

    intent = MagicMock()
    intent.intent = "explore_occupation"
    intent.target_occupation_index = 1
    intent.target_recommendation_id = "occ_da"

    classifier = MagicMock()
    classifier.classify_jobs_followup = AsyncMock(return_value=(intent, []))
    exploration = MagicMock()
    exploration.handle = AsyncMock(return_value=("EXPLORE_OCC", []))

    handler = PresentPhaseHandler(
        conversation_llm=MagicMock(), conversation_caller=MagicMock(),
        intent_classifier=classifier, exploration_handler=exploration,
    )

    response, _ = await handler.handle("tell me about the Data Analyst path", state, MagicMock())

    assert response == "EXPLORE_OCC"
    assert state.current_focus_id == "occ_da"
    assert state.current_recommendation_type == "occupation"
    assert state.conversation_phase == ConversationPhase.CAREER_EXPLORATION


@pytest.mark.asyncio
async def test_reject_during_opportunity_exploration_clears_view(monkeypatch):
    """M1: rejecting a job clears recommendation_view so the next turn isn't re-routed to jobs."""
    opp = _opp_full()
    state = RecommenderAdvisorAgentState(session_id=1, youth_id="y1")
    state.recommendations = _recs(opps=[opp])
    state.current_focus_id = opp.uuid
    state.current_recommendation_type = "opportunity"
    state.recommendation_view = "jobs"
    state.explored_items = [opp.uuid]  # not initial -> intent classification runs

    intent = MagicMock(); intent.intent = "reject"
    intent.target_recommendation_id = None; intent.target_occupation_index = None
    classifier = MagicMock(); classifier.classify_intent = AsyncMock(return_value=(intent, []))

    handler = ExplorationPhaseHandler(conversation_llm=MagicMock(), conversation_caller=MagicMock(),
                                      intent_classifier=classifier)

    _, _ = await handler.handle("not this one", state, _context())

    assert state.recommendation_view is None
    assert state.conversation_phase == ConversationPhase.PRESENT_RECOMMENDATIONS


@pytest.mark.asyncio
async def test_explore_different_from_opportunity_indexes_opportunities():
    """M2: 'tell me about the other one' from a job focus selects another opportunity, not an occupation."""
    opp1, opp2 = _opp("opp_a", "Job A"), _opp("opp_b", "Job B", rank=2)
    occ = OccupationRecommendation(uuid="occ_x", originUuid="occ_x", rank=1,
                                   occupation_id="1", occupation_code="1", occupation="Some Career")
    state = RecommenderAdvisorAgentState(session_id=1, youth_id="y1")
    state.recommendations = _recs(opps=[opp1, opp2], occs=[occ])
    state.current_focus_id = opp1.uuid
    state.current_recommendation_type = "opportunity"
    state.explored_items = [opp1.uuid]

    intent = MagicMock(); intent.intent = "explore_different"
    intent.target_recommendation_id = None; intent.target_occupation_index = 2  # second opportunity
    classifier = MagicMock(); classifier.classify_intent = AsyncMock(return_value=(intent, []))

    handler = ExplorationPhaseHandler(conversation_llm=MagicMock(), conversation_caller=MagicMock(),
                                      intent_classifier=classifier)
    handler._call_llm = AsyncMock(
        return_value=(ConversationResponse(reasoning="r", message="about Job B", finished=False), []))

    await handler.handle("tell me about the other one", state, _context())

    assert state.current_focus_id == "opp_b"
    assert state.current_recommendation_type == "opportunity"
