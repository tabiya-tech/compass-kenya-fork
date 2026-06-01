"""
Regression tests for ConversationService._prepare_recommender_state_if_needed.

History:
- The original lock-in bug guarded the sync with ``if rec_state.preference_vector
  is None``. Because PEAS's preference_vector defaults to a non-None default
  instance, the first turn copied those defaults into RAAS and locked RAAS at
  uniform 0.5 weights forever.
- A first fix attempt phase-gated on ``current_phase == ConversationPhase.COUNSELING``.
  Reviewer flagged that COUNSELING is too coarse — it covers EXPLORE_EXPERIENCES,
  PREFERENCE_ELICITATION and RECOMMENDER_ADVISOR sub-phases. The gate would open
  during EXPLORE_EXPERIENCES, *before* any vignette, with the same defaults trap.
- The current implementation gates on the sub-phase ``RECOMMENDER_ADVISOR`` and
  also opens for post-COUNSELING phases (CHECKOUT/ENDED) to self-heal stuck
  production sessions. Value comparison uses ``content_equals`` to ignore the
  fresh-on-every-instance ``last_updated`` timestamp.

These tests pin all of the above.
"""

import logging
import time
from unittest.mock import MagicMock

import pytest

from app.agent.agent_director.abstract_agent_director import (
    AgentDirectorState,
    ConversationPhase,
    CounselingSubPhase,
)
from app.agent.collect_experiences_agent import CollectExperiencesAgentState
from app.agent.explore_experiences_agent_director import (
    ExploreExperiencesAgentDirectorState,
)
from app.agent.preference_elicitation_agent import PreferenceElicitationAgentState
from app.agent.preference_elicitation_agent.types import PreferenceVector
from app.agent.recommender_advisor_agent import RecommenderAdvisorAgentState
from app.agent.skill_explorer_agent import SkillsExplorerAgentState
from app.agent.welcome_agent import WelcomeAgentState
from app.application_state import ApplicationState
from app.conversation_memory.conversation_memory_types import (
    ConversationMemoryManagerState,
)
from app.conversations.phase_state_machine import JourneyPhase
from app.conversations.service import ConversationService


SESSION_ID = 999_111_222


def _build_state(
    *,
    current_phase: ConversationPhase = ConversationPhase.INTRO,
    sub_phase: CounselingSubPhase = CounselingSubPhase.EXPLORE_EXPERIENCES,
    skip_to_phase=None,
) -> ApplicationState:
    """Construct an ApplicationState at a specific (phase, sub-phase) point."""
    director = AgentDirectorState(session_id=SESSION_ID)
    director.current_phase = current_phase
    director.counseling_sub_phase = sub_phase
    if skip_to_phase is not None:
        director.skip_to_phase = skip_to_phase
    return ApplicationState(
        session_id=SESSION_ID,
        agent_director_state=director,
        welcome_agent_state=WelcomeAgentState(session_id=SESSION_ID),
        explore_experiences_director_state=ExploreExperiencesAgentDirectorState(
            session_id=SESSION_ID
        ),
        conversation_memory_manager_state=ConversationMemoryManagerState(
            session_id=SESSION_ID
        ),
        collect_experience_state=CollectExperiencesAgentState(session_id=SESSION_ID),
        skills_explorer_agent_state=SkillsExplorerAgentState(session_id=SESSION_ID),
        preference_elicitation_agent_state=PreferenceElicitationAgentState(
            session_id=SESSION_ID
        ),
        recommender_advisor_agent_state=RecommenderAdvisorAgentState(
            session_id=SESSION_ID
        ),
    )


def _populate_peas(state: ApplicationState) -> None:
    """Populate PEAS as if the user just finished vignettes + BWS."""
    peas = state.preference_elicitation_agent_state
    peas.preference_vector = PreferenceVector(
        n_vignettes_completed=6,
        confidence_score=0.7,
        financial_importance=0.82,
        career_advancement_importance=0.61,
    )
    peas.bws_scores = {"WA_1": 1.2, "WA_2": 0.4}
    peas.top_10_bws = ["WA_1", "WA_2"]


def _service_stub() -> ConversationService:
    svc = ConversationService.__new__(ConversationService)
    svc._logger = logging.getLogger("test_recommender_state_sync")
    svc._user_recommendations_service = MagicMock()
    return svc


@pytest.fixture
def service() -> ConversationService:
    return _service_stub()


class TestSubPhaseGate:
    """Sub-phase RECOMMENDER_ADVISOR is what unlocks the sync, not COUNSELING."""

    @pytest.mark.asyncio
    async def test_intro_phase_does_not_sync(self, service):
        state = _build_state(current_phase=ConversationPhase.INTRO)
        _populate_peas(state)

        await service._prepare_recommender_state_if_needed(state, "u")

        raas = state.recommender_advisor_agent_state
        assert raas.preference_vector is None
        assert raas.bws_scores is None
        assert raas.top_10_bws is None

    @pytest.mark.asyncio
    async def test_counseling_explore_experiences_does_not_sync(self, service):
        """Regression check for the first-fix defect: COUNSELING starts in the
        EXPLORE_EXPERIENCES sub-phase. If we gated on current_phase alone,
        PEAS's default-shaped instance would get copied into RAAS here.
        """
        state = _build_state(
            current_phase=ConversationPhase.COUNSELING,
            sub_phase=CounselingSubPhase.EXPLORE_EXPERIENCES,
        )
        assert state.preference_elicitation_agent_state.preference_vector.n_vignettes_completed == 0

        await service._prepare_recommender_state_if_needed(state, "u")

        assert state.recommender_advisor_agent_state.preference_vector is None

    @pytest.mark.asyncio
    async def test_counseling_preference_elicitation_does_not_sync(self, service):
        """Mid-vignettes: PEAS PV may have partial values but the recommender
        isn't running yet, so RAAS should remain untouched.
        """
        state = _build_state(
            current_phase=ConversationPhase.COUNSELING,
            sub_phase=CounselingSubPhase.PREFERENCE_ELICITATION,
        )
        state.preference_elicitation_agent_state.preference_vector.n_vignettes_completed = 3

        await service._prepare_recommender_state_if_needed(state, "u")

        assert state.recommender_advisor_agent_state.preference_vector is None
        assert state.recommender_advisor_agent_state.bws_scores is None

    @pytest.mark.asyncio
    async def test_counseling_recommender_advisor_does_sync(self, service):
        state = _build_state(
            current_phase=ConversationPhase.COUNSELING,
            sub_phase=CounselingSubPhase.RECOMMENDER_ADVISOR,
        )
        _populate_peas(state)

        await service._prepare_recommender_state_if_needed(state, "u")

        raas = state.recommender_advisor_agent_state
        assert raas.preference_vector is not None
        assert raas.preference_vector.n_vignettes_completed == 6
        assert raas.bws_scores == {"WA_1": 1.2, "WA_2": 0.4}
        assert raas.top_10_bws == ["WA_1", "WA_2"]


class TestPostCounselingSelfHeal:
    """CHECKOUT and ENDED open the gate so stuck production sessions self-heal."""

    @pytest.mark.asyncio
    async def test_checkout_with_stuck_raas_defaults_heals(self, service):
        state = _build_state(current_phase=ConversationPhase.CHECKOUT)
        state.recommender_advisor_agent_state.preference_vector = PreferenceVector()
        _populate_peas(state)

        await service._prepare_recommender_state_if_needed(state, "u")

        raas_pv = state.recommender_advisor_agent_state.preference_vector
        assert raas_pv.n_vignettes_completed == 6
        assert raas_pv.financial_importance == pytest.approx(0.82)

    @pytest.mark.asyncio
    async def test_ended_with_stuck_raas_defaults_heals(self, service):
        state = _build_state(current_phase=ConversationPhase.ENDED)
        state.recommender_advisor_agent_state.preference_vector = PreferenceVector()
        _populate_peas(state)

        await service._prepare_recommender_state_if_needed(state, "u")

        assert state.recommender_advisor_agent_state.preference_vector.n_vignettes_completed == 6

    @pytest.mark.asyncio
    async def test_checkout_without_peas_data_does_not_seed_raas_with_defaults(self, service):
        """Edge case: a session lands in CHECKOUT without going through vignettes.
        Without the n_vignettes_completed > 0 precondition, the self-heal path
        would copy PEAS's default instance into RAAS — exactly the defaults trap
        we're trying to avoid.
        """
        state = _build_state(current_phase=ConversationPhase.CHECKOUT)
        assert state.preference_elicitation_agent_state.preference_vector.n_vignettes_completed == 0

        await service._prepare_recommender_state_if_needed(state, "u")

        assert state.recommender_advisor_agent_state.preference_vector is None


class TestContentEqualityIgnoresLastUpdated:
    """Pydantic's default == includes last_updated, which is fresh on every
    default instance. The sync must use content_equals to avoid spurious
    diffs and the resulting Mongo writes on every load.
    """

    def test_default_instances_content_equal_despite_timestamp_drift(self):
        a = PreferenceVector()
        time.sleep(0.001)
        b = PreferenceVector()
        assert a != b  # baseline: Pydantic __eq__ does NOT match
        assert a.content_equals(b)  # but content_equals does

    @pytest.mark.asyncio
    async def test_idempotent_no_resync_when_content_matches(self, service):
        state = _build_state(
            current_phase=ConversationPhase.COUNSELING,
            sub_phase=CounselingSubPhase.RECOMMENDER_ADVISOR,
        )
        _populate_peas(state)

        await service._prepare_recommender_state_if_needed(state, "u")
        first_ref = state.recommender_advisor_agent_state.preference_vector

        # Simulate Mongo round-trip: clone RAAS PV by content so it's a distinct
        # instance with the same data. Without content_equals this would trigger
        # a spurious re-sync.
        state.recommender_advisor_agent_state.preference_vector = PreferenceVector(
            **first_ref.model_dump()
        )

        await service._prepare_recommender_state_if_needed(state, "u")

        # PV ref should not have been overwritten back to PEAS's instance since
        # the existing one is content-equal. (Identity comparison documents the
        # no-resync intent.)
        assert state.recommender_advisor_agent_state.preference_vector is not first_ref
        # And the data is still correct
        assert state.recommender_advisor_agent_state.preference_vector.n_vignettes_completed == 6


class TestBwsBundle:
    @pytest.mark.asyncio
    async def test_hb_path_overrides_fallback(self, service):
        state = _build_state(
            current_phase=ConversationPhase.COUNSELING,
            sub_phase=CounselingSubPhase.RECOMMENDER_ADVISOR,
        )
        _populate_peas(state)
        peas = state.preference_elicitation_agent_state
        peas.hb_scores = {
            "WA_A": {"mean": 2.5, "sd": 0.1},
            "WA_B": {"mean": 1.1, "sd": 0.2},
        }
        peas.hb_ranking = ["WA_A", "WA_B"]

        await service._prepare_recommender_state_if_needed(state, "u")

        raas = state.recommender_advisor_agent_state
        assert raas.bws_scores == {"WA_A": 2.5, "WA_B": 1.1}
        assert raas.top_10_bws == ["WA_A", "WA_B"]


class TestStepSkip:
    @pytest.mark.asyncio
    async def test_skip_to_recommendation_opens_gate(self, service):
        state = _build_state(
            current_phase=ConversationPhase.INTRO,
            skip_to_phase=JourneyPhase.RECOMMENDATION,
        )
        _populate_peas(state)

        async def _none(*_a, **_k):
            return None
        service._user_recommendations_service.get_by_user_id = _none

        await service._prepare_recommender_state_if_needed(state, "u")

        raas = state.recommender_advisor_agent_state
        assert raas.preference_vector is not None
        assert raas.bws_scores == {"WA_1": 1.2, "WA_2": 0.4}
