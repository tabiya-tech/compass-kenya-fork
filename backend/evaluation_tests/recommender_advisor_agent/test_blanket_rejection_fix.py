"""
Evaluation test for the blanket rejection fix.

Replays the exact 7-turn transcript from the bug report and asserts:
  - Turn 3 ("No I don't like any of them") does NOT transition to ACTION_PLANNING
  - finished=False on every turn
  - All 7 turns complete without the session being terminated

Requires GCP credentials. Marked evaluation_test to exclude from CI.

Bug reference: recommender agent abruptly ends conversation when user rejects
all recommendations in ADDRESS_CONCERNS phase.
"""

import asyncio
import logging
from pathlib import Path

import pytest

from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    Node2VecRecommendations,
    OccupationRecommendation,
    SkillsTrainingRecommendation,
    ScoreBreakdown,
    SkillComponent,
)
from app.agent.recommender_advisor_agent.phase_handlers.intro_handler import IntroPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.present_handler import PresentPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.exploration_handler import ExplorationPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.concerns_handler import ConcernsPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.tradeoffs_handler import TradeoffsPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.followup_handler import FollowupPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.skills_pivot_handler import SkillsPivotPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.action_handler import ActionPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.wrapup_handler import WrapupPhaseHandler
from app.agent.recommender_advisor_agent.intent_classifier import IntentClassifier
from app.agent.recommender_advisor_agent.recommendation_interface import RecommendationInterface
from app.agent.recommender_advisor_agent.llm_response_models import (
    ConversationResponse,
    ResistanceClassification,
    UserIntentClassification,
    ActionExtractionResult,
)
from app.agent.llm_caller import LLMCaller
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
    ConversationTurn,
)
from app.agent.agent_types import AgentInput, AgentOutput
from app.agent.preference_elicitation_agent.types import PreferenceVector
from common_libs.llm.generative_models import GeminiGenerativeLLM
from common_libs.llm.models_utils import LLMConfig, LOW_TEMPERATURE_GENERATION_CONFIG, JSON_GENERATION_CONFIG
from app.countries import Country

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transcript from the bug report (no training recs — prevents pivot fallback)
# ---------------------------------------------------------------------------

TRANSCRIPT_TURNS = [
    "",                                                                        # Turn 0 – initial
    "yeah",                                                                    # Turn 1
    "I don't think I have the skills to become a Web Developer",               # Turn 2 – concern
    "No I don't like any of them",                                             # Turn 3 ← was bug trigger
    "No I want to keep talking",                                               # Turn 4
    "More careers",                                                            # Turn 5
    "No don't end. What about jobs, do you have any recommendations here?",    # Turn 6
]

BLANKET_REJECTION_TURN = 3


def _build_recommendations() -> Node2VecRecommendations:
    """5 occupations from the bug transcript. No training recs so pivot-to-training can't mask the bug."""
    def _occ(uuid, code, name, demand_label, score):
        return OccupationRecommendation(
            uuid=uuid,
            originUuid=f"{uuid}_origin",
            rank=1,
            occupation_id=f"KESCO_{code}",
            occupation_code=code,
            occupation=name,
            final_score=score,
            score_breakdown=ScoreBreakdown(
                total_skill_utility=score - 0.05,
                skill_components=SkillComponent(loc=score, ess=score - 0.07, opt=score - 0.05, grp=score - 0.03),
                skill_penalty_applied=0.0,
                preference_score=score + 0.02,
                demand_score=score - 0.08,
                demand_label=demand_label,
            ),
            salary_range="KES 25,000-40,000/month",
            justification="Good fit based on your profile.",
        )

    return Node2VecRecommendations(
        youth_id="bug_repro_user",
        generated_at="2026-04-26T10:00:00Z",
        recommended_by=["Algorithm"],
        occupation_recommendations=[
            _occ("occ_borehole", "8121", "Borehole Driller", "Moderate Demand", 0.80),
            _occ("occ_webdev", "2512", "Web Developer", "Moderate Demand", 0.78),
            _occ("occ_truck", "8322", "Truck Driver", "Moderate Demand", 0.75),
            _occ("occ_elec", "7411", "Electrician", "High Demand", 0.82),
            _occ("occ_port", "9333", "Port Cargo Handler", "High Demand", 0.72),
        ],
        opportunity_recommendations=[],
        skillstraining_recommendations=[],   # intentionally empty
    )


async def _build_handlers(llm: GeminiGenerativeLLM) -> dict:
    conversation_caller = LLMCaller[ConversationResponse](model_response_type=ConversationResponse)
    resistance_caller = LLMCaller[ResistanceClassification](model_response_type=ResistanceClassification)
    intent_caller = LLMCaller[UserIntentClassification](model_response_type=UserIntentClassification)
    action_caller = LLMCaller[ActionExtractionResult](model_response_type=ActionExtractionResult)

    intent_classifier = IntentClassifier(intent_caller=intent_caller)
    recommendation_interface = RecommendationInterface(node2vec_client=None)

    action_handler = ActionPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        action_caller=action_caller,
        intent_classifier=intent_classifier,
    )
    concerns_handler = ConcernsPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        resistance_caller=resistance_caller,
        intent_classifier=intent_classifier,
        action_handler=action_handler,
        recommendation_interface=recommendation_interface,
    )
    tradeoffs_handler = TradeoffsPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
    )
    wrapup_handler = WrapupPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        db6_client=None,
    )
    followup_handler = FollowupPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
    )
    skills_pivot_handler = SkillsPivotPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
    )
    exploration_handler = ExplorationPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
        concerns_handler=concerns_handler,
        tradeoffs_handler=tradeoffs_handler,
        occupation_search_service=None,
    )
    intro_handler = IntroPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        recommendation_interface=recommendation_interface,
        occupation_search_service=None,
    )
    present_handler = PresentPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
        exploration_handler=exploration_handler,
        concerns_handler=concerns_handler,
        tradeoffs_handler=tradeoffs_handler,
        occupation_search_service=None,
    )

    exploration_handler._action_handler = action_handler
    exploration_handler._skills_pivot_handler = skills_pivot_handler
    present_handler._skills_pivot_handler = skills_pivot_handler
    action_handler._present_handler = present_handler
    action_handler._concerns_handler = concerns_handler
    action_handler._wrapup_handler = wrapup_handler
    skills_pivot_handler._exploration_handler = exploration_handler
    skills_pivot_handler._concerns_handler = concerns_handler
    skills_pivot_handler._action_planning_handler = action_handler
    skills_pivot_handler._present_handler = present_handler

    return {
        ConversationPhase.INTRO: intro_handler,
        ConversationPhase.PRESENT_RECOMMENDATIONS: present_handler,
        ConversationPhase.CAREER_EXPLORATION: exploration_handler,
        ConversationPhase.ADDRESS_CONCERNS: concerns_handler,
        ConversationPhase.DISCUSS_TRADEOFFS: tradeoffs_handler,
        ConversationPhase.FOLLOW_UP: followup_handler,
        ConversationPhase.SKILLS_UPGRADE_PIVOT: skills_pivot_handler,
        ConversationPhase.ACTION_PLANNING: action_handler,
        ConversationPhase.WRAPUP: wrapup_handler,
        ConversationPhase.COMPLETE: wrapup_handler,
    }


@pytest.mark.asyncio
@pytest.mark.evaluation_test
async def test_blanket_rejection_does_not_terminate_session():
    """
    Replay the 7-turn bug transcript.

    Before the fix: Turn 3 misrouted through the resistance classifier as
    'none' resistance, transitioned to ACTION_PLANNING, and the LLM
    intermittently set finished=True, ending the conversation.

    After the fix:
    - Turn 3 is classified as 'reject' intent
    - _handle_blanket_rejection is called, engine re-called, stubs return
      same UUIDs so graceful "check back" message is returned
    - finished=False on every turn
    - Session runs all 7 turns
    """
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "blanket_rejection_fix_test.log"

    llm = GeminiGenerativeLLM(
        system_instructions=(
            "You are a career advisor. Always respond with valid JSON matching the "
            "ConversationResponse schema. Only set finished=true when the user has "
            "explicitly said goodbye and you are wrapping up."
        ),
        config=LLMConfig(generation_config=LOW_TEMPERATURE_GENERATION_CONFIG | JSON_GENERATION_CONFIG),
    )

    handlers = await _build_handlers(llm)

    state = RecommenderAdvisorAgentState(
        session_id=99001,
        youth_id="bug_repro_user",
        country_of_user=Country.KENYA,
        conversation_phase=ConversationPhase.PRESENT_RECOMMENDATIONS,
        recommendations=_build_recommendations(),
        preference_vector=PreferenceVector(
            financial_importance=0.85,
            work_environment_importance=0.70,
            career_advancement_importance=0.55,
            work_life_balance_importance=0.55,
            job_security_importance=0.50,
            task_preference_importance=0.50,
            social_impact_importance=0.45,
        ),
        discuss_recommendations=True,
    )

    history = ConversationHistory()

    with open(log_file, "w") as f:
        f.write("=== Blanket Rejection Fix Evaluation ===\n\n")

    for turn_idx, user_msg in enumerate(TRANSCRIPT_TURNS):
        context = ConversationContext(all_history=history, history=history, summary="")
        phase_before = state.conversation_phase
        handler = handlers[state.conversation_phase]

        if state.conversation_phase == ConversationPhase.COMPLETE:
            response, llm_stats = await handler.handle_complete(user_msg, state, context)
        else:
            response, llm_stats = await handler.handle(user_msg, state, context)

        phase_after = state.conversation_phase

        with open(log_file, "a") as f:
            f.write(f"Turn {turn_idx}: [{phase_before.value} → {phase_after.value}]\n")
            f.write(f"  user: {user_msg!r}\n")
            f.write(f"  agent: {response.message[:120]}\n")
            f.write(f"  finished: {response.finished}\n\n")

        logger.info("Turn %d | %s → %s | finished=%s", turn_idx, phase_before.value, phase_after.value, response.finished)

        # Core assertion: session must never terminate before COMPLETE
        assert response.finished is False, (
            f"Turn {turn_idx}: finished=True in phase {phase_after.value}. "
            f"User said: {user_msg!r}. Agent said: {response.message[:200]}"
        )

        # Turn 3 specific: blanket rejection must NOT transition to ACTION_PLANNING
        if turn_idx == BLANKET_REJECTION_TURN:
            assert phase_after != ConversationPhase.ACTION_PLANNING, (
                f"Turn 3 (blanket rejection) incorrectly transitioned to ACTION_PLANNING. "
                f"This is the bug path — the reject intent was not classified correctly."
            )

        history.turns.append(ConversationTurn(
            index=turn_idx,
            input=AgentInput(message=user_msg, is_artificial=(turn_idx == 0)),
            output=AgentOutput(
                message_for_user=response.message,
                finished=response.finished,
                llm_stats=llm_stats,
                agent_response_time_in_sec=0.0,
            ),
        ))

    with open(log_file, "a") as f:
        f.write("=== ALL 7 TURNS COMPLETED — TEST PASSED ===\n")

    logger.info("All %d turns completed without premature termination.", len(TRANSCRIPT_TURNS))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_blanket_rejection_does_not_terminate_session())
