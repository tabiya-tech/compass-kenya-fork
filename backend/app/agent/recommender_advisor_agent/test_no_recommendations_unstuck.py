"""
Tests for the "user stuck when no recommendations are found" fix.

When the matching service returns an empty result (no career paths and no job
openings), the agent must NOT cache that empty result. Caching it froze the user:
the fetch guard (`recommendations is None`) treated the empty object as "already
fetched", so the matching service was never called again and the user looped on a
"no recommendations" message forever.

The fix: don't cache empty results, tell the user to try again, and on the next
turn re-call the matching service.

Covers:
1. Node2VecRecommendations.is_empty()
2. INTRO handler — empty result is not cached, phase stays INTRO, retry message
3. INTRO handler — non-empty result IS cached and transitions to PRESENT
4. INTRO handler — empty-then-populated across two turns re-calls the engine
5. PRESENT handler — empty dead-end clears cache and routes back to INTRO
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    Node2VecRecommendations,
    OccupationRecommendation,
    ScoreBreakdown,
    SkillComponent,
)
from app.agent.recommender_advisor_agent.phase_handlers.intro_handler import IntroPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.present_handler import PresentPhaseHandler
from app.conversation_memory.conversation_memory_manager import ConversationContext
from app.countries import Country


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_occupation(uuid: str, name: str) -> OccupationRecommendation:
    return OccupationRecommendation(
        uuid=uuid,
        originUuid=f"{uuid}_origin",
        rank=1,
        occupation_id=f"OCC_{uuid}",
        occupation_code="1234",
        occupation=name,
        final_score=0.75,
        score_breakdown=ScoreBreakdown(
            total_skill_utility=0.70,
            skill_components=SkillComponent(loc=0.75, ess=0.68, opt=0.70, grp=0.72),
            skill_penalty_applied=0.0,
            preference_score=0.75,
            demand_score=0.70,
            demand_label="Moderate Demand",
        ),
    )


def _empty_recommendations() -> Node2VecRecommendations:
    return Node2VecRecommendations(
        youth_id="test_user",
        generated_at="2026-01-01T00:00:00Z",
        recommended_by=["Algorithm"],
        occupation_recommendations=[],
        opportunity_recommendations=[],
        skillstraining_recommendations=[],
    )


def _populated_recommendations() -> Node2VecRecommendations:
    return Node2VecRecommendations(
        youth_id="test_user",
        generated_at="2026-01-01T00:00:00Z",
        recommended_by=["Algorithm"],
        occupation_recommendations=[_make_occupation("uuid_a", "Electrician")],
        opportunity_recommendations=[],
        skillstraining_recommendations=[],
    )


def _make_state(phase: ConversationPhase = ConversationPhase.INTRO) -> RecommenderAdvisorAgentState:
    return RecommenderAdvisorAgentState(
        session_id=1,
        youth_id="test_user",
        country_of_user=Country.KENYA,
        discuss_recommendations=True,
        conversation_phase=phase,
    )


def _make_intro_handler(recommendation_interface) -> IntroPhaseHandler:
    return IntroPhaseHandler(
        conversation_llm=MagicMock(),
        conversation_caller=MagicMock(),
        recommendation_interface=recommendation_interface,
    )


def _make_present_handler() -> PresentPhaseHandler:
    return PresentPhaseHandler(
        conversation_llm=MagicMock(),
        conversation_caller=MagicMock(),
    )


@pytest.fixture
def mock_context():
    return MagicMock(spec=ConversationContext)


# ---------------------------------------------------------------------------
# 1. is_empty()
# ---------------------------------------------------------------------------

class TestIsEmpty:
    def test_empty_when_no_occupations_and_no_opportunities(self):
        assert _empty_recommendations().is_empty() is True

    def test_not_empty_with_occupations(self):
        assert _populated_recommendations().is_empty() is False


# ---------------------------------------------------------------------------
# 2-4. INTRO handler retry behaviour
# ---------------------------------------------------------------------------

class TestIntroEmptyRetry:
    @pytest.mark.asyncio
    async def test_empty_result_is_not_cached(self, mock_context):
        # GIVEN the matching engine returns nothing
        rec_interface = MagicMock()
        rec_interface.generate_recommendations = AsyncMock(return_value=_empty_recommendations())
        handler = _make_intro_handler(rec_interface)
        state = _make_state()

        # WHEN the intro handler runs
        response, _ = await handler.handle("", state, mock_context)

        # THEN the empty result is NOT cached (so the next turn retries)
        assert state.recommendations is None
        # AND we stay in INTRO rather than advancing to a dead-end PRESENT
        assert state.conversation_phase == ConversationPhase.INTRO
        # AND the user is invited to try again, conversation stays open
        assert "try again" in response.message.lower()
        assert response.finished is False

    @pytest.mark.asyncio
    async def test_populated_result_is_cached_and_advances(self, mock_context):
        # GIVEN the engine returns real recommendations
        rec_interface = MagicMock()
        rec_interface.generate_recommendations = AsyncMock(return_value=_populated_recommendations())
        handler = _make_intro_handler(rec_interface)
        state = _make_state()

        # WHEN the intro handler runs
        response, _ = await handler.handle("", state, mock_context)

        # THEN recommendations are cached and we advance to PRESENT
        assert state.recommendations is not None
        assert state.conversation_phase == ConversationPhase.PRESENT_RECOMMENDATIONS
        assert state.awaiting_view_choice is True
        assert response.finished is False

    @pytest.mark.asyncio
    async def test_empty_then_populated_recalls_engine(self, mock_context):
        # GIVEN the engine first returns nothing, then real recommendations
        rec_interface = MagicMock()
        rec_interface.generate_recommendations = AsyncMock(
            side_effect=[_empty_recommendations(), _populated_recommendations()]
        )
        handler = _make_intro_handler(rec_interface)
        state = _make_state()

        # WHEN the first turn runs (empty) and then a second turn runs
        await handler.handle("", state, mock_context)
        assert state.recommendations is None  # not cached
        await handler.handle("try again", state, mock_context)

        # THEN the engine was called twice (the retry actually fired)
        assert rec_interface.generate_recommendations.call_count == 2
        # AND the second, populated result is now cached
        assert state.recommendations is not None
        assert state.conversation_phase == ConversationPhase.PRESENT_RECOMMENDATIONS


# ---------------------------------------------------------------------------
# 5. PRESENT handler dead-end recovery
# ---------------------------------------------------------------------------

class TestPresentEmptyRecovery:
    @pytest.mark.asyncio
    async def test_empty_recommendations_routes_back_to_intro(self, mock_context):
        # GIVEN we somehow reach PRESENT with empty (non-None) recommendations
        handler = _make_present_handler()
        state = _make_state(phase=ConversationPhase.PRESENT_RECOMMENDATIONS)
        state.recommendations = _empty_recommendations()

        # WHEN the present handler runs
        response, _ = await handler.handle("", state, mock_context)

        # THEN the empty cache is cleared and we route back to INTRO so the next turn retries
        assert state.recommendations is None
        assert state.conversation_phase == ConversationPhase.INTRO
        # AND the user is not frozen — conversation stays open with a retry message
        assert "try again" in response.message.lower()
        assert response.finished is False
