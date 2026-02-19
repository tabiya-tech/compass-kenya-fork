"""
Builds PhaseDataStatus from application state and user_recommendations.

PhaseDataStatus drives determine_start_phase; no override or external store.
"""
from app.application_state import ApplicationState
from app.conversations.phase_state_machine import JourneyPhase, PhaseDataStatus
from app.agent.explore_experiences_agent_director import DiveInPhase
from app.agent.agent_director.abstract_agent_director import ConversationPhase
from app.user_recommendations.services.service import IUserRecommendationsService


def conversation_phase_for_entry(journey_phase: JourneyPhase) -> ConversationPhase | None:
    """
    Map journey phase to conversation phase when step-skipping.
    None means stay in INTRO; otherwise transition to that phase.
    """
    if journey_phase == JourneyPhase.SKILLS_ELICITATION:
        return None
    return ConversationPhase.COUNSELING


def apply_entry_phase(state: ApplicationState, entry_phase: JourneyPhase) -> bool:
    """
    Apply entry phase to state. Returns True if transitioned to COUNSELING.
    """
    target = conversation_phase_for_entry(entry_phase)
    if target is None:
        return False
    state.agent_director_state.skip_to_phase = entry_phase
    state.agent_director_state.current_phase = target
    return True


async def build_phase_data_status_from_state(
    state: ApplicationState,
    user_id: str,
    user_recommendations_service: IUserRecommendationsService,
) -> PhaseDataStatus:
    has_skills = _has_skills_elicitation(state)
    has_preference = _has_preference_elicitation(state)
    has_recommendation = await user_recommendations_service.has_recommendations(
        user_id
    )
    return PhaseDataStatus(
        has_skills_elicitation=has_skills,
        has_preference_elicitation=has_preference,
        has_matching=False,
        has_recommendation=has_recommendation,
    )


def _has_skills_elicitation(state: ApplicationState) -> bool:
    exp_state = state.explore_experiences_director_state
    explored = exp_state.explored_experiences or []
    for exp in explored:
        top = exp.top_skills
        if not top:
            continue
        if isinstance(top[0], tuple):
            skills = [s for _, s in top]
        else:
            skills = top
        if skills:
            return True
    for exp_state_val in exp_state.experiences_state.values():
        if (
            exp_state_val.dive_in_phase == DiveInPhase.PROCESSED
            and exp_state_val.experience.top_skills
        ):
            return True
    return False


def _has_preference_elicitation(state: ApplicationState) -> bool:
    pv = state.preference_elicitation_agent_state.preference_vector
    return pv is not None and pv.confidence_score > 0.0
