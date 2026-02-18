"""
Test for Action Planning Intent Classification Fix

This test simulates the exact conversation that caused the error:
- User expresses interest in DJing (outside recommendations)
- User accepts returning to original recommendations
- User shows interest in marine equipment fundi
- User commits to enrolling in training

Tests the fixes:
1. Null check for failed intent classification
2. ACTION_PLANNING-specific prompt in IntentClassifier
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import pytest

from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import ConversationPhase
from app.agent.recommender_advisor_agent.phase_handlers.intro_handler import IntroPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.present_handler import PresentPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.exploration_handler import ExplorationPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.concerns_handler import ConcernsPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.action_handler import ActionPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.tradeoffs_handler import TradeoffsPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.followup_handler import FollowupPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.skills_pivot_handler import SkillsPivotPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.wrapup_handler import WrapupPhaseHandler
from app.agent.recommender_advisor_agent.intent_classifier import IntentClassifier
from app.agent.recommender_advisor_agent.llm_response_models import (
    ConversationResponse,
    ResistanceClassification,
    UserIntentClassification,
    ActionExtractionResult
)
from app.agent.recommender_advisor_agent.recommendation_interface import RecommendationInterface
from app.agent.llm_caller import LLMCaller
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
    ConversationTurn
)
from app.agent.agent_types import AgentInput, AgentOutput
from common_libs.llm.generative_models import GeminiGenerativeLLM
from common_libs.llm.models_utils import LLMConfig, LOW_TEMPERATURE_GENERATION_CONFIG, JSON_GENERATION_CONFIG
from app.countries import Country

# Import sample data creation functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from test_recommender_agent_interactive import (
    create_sample_recommendations,
    create_sample_skills_vector,
    create_sample_preference_vector
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationLogger:
    """Logs conversation to a file."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Clear existing log
        with open(self.log_file, 'w') as f:
            f.write(f"=== Conversation Log - {datetime.now().isoformat()} ===\n\n")

    def log_turn(self, turn_number: int, phase: str, user_input: str, agent_response: str,
                 intent_classification: dict = None, llm_stats: list = None):
        """Log a conversation turn."""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"TURN {turn_number} - Phase: {phase}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"USER: {user_input}\n\n")

            if intent_classification:
                f.write("INTENT CLASSIFICATION:\n")
                f.write(f"  Intent: {intent_classification.get('intent', 'N/A')}\n")
                f.write(f"  Reasoning: {intent_classification.get('reasoning', 'N/A')}\n")
                f.write(f"  Target Recommendation ID: {intent_classification.get('target_recommendation_id', 'N/A')}\n")
                f.write(f"  Target Occupation Index: {intent_classification.get('target_occupation_index', 'N/A')}\n")
                f.write(f"  Requested Occupation: {intent_classification.get('requested_occupation_name', 'N/A')}\n\n")

            f.write(f"AGENT: {agent_response}\n\n")

            if llm_stats:
                total_input = sum(s.prompt_token_count for s in llm_stats)
                total_output = sum(s.response_token_count for s in llm_stats)
                total_latency = sum(s.response_time_in_sec for s in llm_stats)

                f.write("LLM STATS:\n")
                f.write(f"  Input Tokens: {total_input:,}\n")
                f.write(f"  Output Tokens: {total_output:,}\n")
                f.write(f"  Latency: {total_latency:.2f}s\n")

    def log_error(self, turn_number: int, error: Exception):
        """Log an error."""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ERROR AT TURN {turn_number}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Error Type: {type(error).__name__}\n")
            f.write(f"Error Message: {str(error)}\n\n")

    def log_success(self):
        """Log successful completion."""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write("TEST PASSED SUCCESSFULLY\n")
            f.write(f"{'='*80}\n")


async def initialize_handlers():
    """Initialize all phase handlers."""
    # Initialize LLM
    llm_config = LLMConfig(
        generation_config=LOW_TEMPERATURE_GENERATION_CONFIG | JSON_GENERATION_CONFIG
    )

    llm = GeminiGenerativeLLM(
        system_instructions="",
        config=llm_config
    )

    # Create LLM callers
    conversation_caller = LLMCaller[ConversationResponse](
        model_response_type=ConversationResponse
    )

    resistance_caller = LLMCaller[ResistanceClassification](
        model_response_type=ResistanceClassification
    )

    intent_caller = LLMCaller[UserIntentClassification](
        model_response_type=UserIntentClassification
    )

    action_caller = LLMCaller[ActionExtractionResult](
        model_response_type=ActionExtractionResult
    )

    # Initialize recommendation interface
    recommendation_interface = RecommendationInterface(node2vec_client=None)

    # Initialize IntentClassifier
    intent_classifier = IntentClassifier(intent_caller=intent_caller)

    # Initialize handlers
    concerns_handler = ConcernsPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        resistance_caller=resistance_caller,
        intent_classifier=intent_classifier,
        occupation_search_service=None
    )

    intro_handler = IntroPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        recommendation_interface=recommendation_interface,
        occupation_search_service=None
    )

    tradeoffs_handler = TradeoffsPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller
    )

    skills_pivot_handler = SkillsPivotPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier
    )

    wrapup_handler = WrapupPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        db6_client=None
    )

    followup_handler = FollowupPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier
    )

    exploration_handler = ExplorationPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
        concerns_handler=concerns_handler,
        tradeoffs_handler=tradeoffs_handler,
        occupation_search_service=None
    )

    present_handler = PresentPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
        exploration_handler=exploration_handler,
        concerns_handler=concerns_handler,
        tradeoffs_handler=tradeoffs_handler,
        occupation_search_service=None
    )

    action_handler = ActionPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        action_caller=action_caller,
        intent_classifier=intent_classifier
    )

    # Set up delegation chains
    exploration_handler._action_handler = action_handler
    exploration_handler._tradeoffs_handler = tradeoffs_handler
    exploration_handler._skills_pivot_handler = skills_pivot_handler
    present_handler._skills_pivot_handler = skills_pivot_handler
    action_handler._present_handler = present_handler
    action_handler._concerns_handler = concerns_handler
    action_handler._wrapup_handler = wrapup_handler
    concerns_handler._action_handler = action_handler
    skills_pivot_handler._exploration_handler = exploration_handler
    skills_pivot_handler._concerns_handler = concerns_handler
    skills_pivot_handler._action_planning_handler = action_handler
    skills_pivot_handler._present_handler = present_handler

    return {
        ConversationPhase.INTRO: intro_handler,
        ConversationPhase.PRESENT_RECOMMENDATIONS: present_handler,
        ConversationPhase.CAREER_EXPLORATION: exploration_handler,
        ConversationPhase.ADDRESS_CONCERNS: concerns_handler,
        ConversationPhase.ACTION_PLANNING: action_handler,
        ConversationPhase.DISCUSS_TRADEOFFS: tradeoffs_handler,
        ConversationPhase.FOLLOW_UP: followup_handler,
        ConversationPhase.SKILLS_UPGRADE_PIVOT: skills_pivot_handler,
        ConversationPhase.WRAPUP: wrapup_handler,
        ConversationPhase.COMPLETE: wrapup_handler
    }


@pytest.mark.asyncio
@pytest.mark.llm_integration
@pytest.mark.evaluation_test
async def test_action_planning_intent_classification():
    """
    Test the exact conversation flow that caused the error.

    Conversation flow:
    1. Intro
    2. User: "okay" → Present recommendations
    3. User: "none of these seem interesting, I want to transition into DJing" → Guardrail
    4. User: "I would still like to explore what it takes to pursue DJing" → Skills pivot
    5. User: "Okay maybe I want to look at careers that build more directly on your current strengths" → Back to present
    6. User: "show them to me again" → Re-present
    7. User: "maybe the Equipment fundi" → Career exploration
    8. User: "I would love to explore how becoming a general electrician first..." → Exploration with tradeoff
    9. User: "Well, electrical work, idk, arent there electrical hazards..." → Address concerns
    10. User: "yes that clears it" → Action planning
    11. User: "yes I want to enroll in Training" → THIS CAUSED THE ERROR
    """

    # Set up logging
    log_file = Path(__file__).parent / "logs" / "action_planning_fix_test.log"
    conv_logger = ConversationLogger(log_file)

    logger.info(f"Starting test - log file: {log_file}")

    try:
        # Initialize handlers
        handlers = await initialize_handlers()

        # Create initial state
        state = RecommenderAdvisorAgentState(
            session_id=12345,
            youth_id="test_user_123",
            country_of_user=Country.KENYA,
            conversation_phase=ConversationPhase.INTRO,
            recommendations=create_sample_recommendations(),
            skills_vector=create_sample_skills_vector(),
            preference_vector=create_sample_preference_vector()
        )

        # Create conversation context
        conversation_history = ConversationHistory()

        # Define conversation turns (simulating the exact user inputs)
        conversation_turns = [
            ("", "INTRO - auto"),  # Turn 0: Intro
            ("okay", "Present recommendations"),
            ("Uhm none of these seem interesting to me, I want to transition into Djiing", "Guardrail trigger"),
            ("I would still like to explore what it takes to pursue DJing", "Persist on DJing"),
            ("Okay maybe I want to look at careers that build more directly on your current strengths", "Return to recommendations"),
            ("show them to me again", "Re-present"),
            ("maybe the Equipment fundi", "Interest in marine fundi"),
            ("I would love to explore how becoming a general electrician first might open doors and potentially lead to specialized marine work", "Explore electrician path"),
            ("Well, electrical work, idk, arent there electrical hazards that can harm me and when they do what insures me for medical bills thereafter. Plus, sijui kama inaleta dooh poa", "Express concerns"),
            ("yes that clears it", "Accept concern resolution"),
            ("yes I want to enroll in Training", "THE PROBLEMATIC INPUT - Commit to training"),
        ]

        # Execute conversation
        for turn_index, (user_input, description) in enumerate(conversation_turns):
            logger.info(f"\n{'='*80}")
            logger.info(f"Turn {turn_index}: {description}")
            logger.info(f"Phase: {state.conversation_phase}")
            logger.info(f"User input: '{user_input}'")

            # Get current handler
            current_handler = handlers.get(state.conversation_phase)
            assert current_handler is not None, f"No handler for phase: {state.conversation_phase}"

            # Create context
            context = ConversationContext(
                all_history=conversation_history,
                history=conversation_history,
                summary=""
            )

            # Execute handler
            try:
                response, llm_stats = await current_handler.handle(user_input, state, context)

                logger.info(f"Agent response: {response.message[:100]}...")
                logger.info(f"New phase: {state.conversation_phase}")

                # Log to file
                conv_logger.log_turn(
                    turn_number=turn_index,
                    phase=str(state.conversation_phase),
                    user_input=user_input or "(auto)",
                    agent_response=response.message,
                    llm_stats=llm_stats
                )

                # Update conversation history
                conversation_turn = ConversationTurn(
                    index=turn_index,
                    input=AgentInput(message=user_input, is_artificial=(turn_index == 0)),
                    output=AgentOutput(
                        message_for_user=response.message,
                        finished=response.finished,
                        llm_stats=llm_stats,
                        agent_response_time_in_sec=sum(s.response_time_in_sec for s in llm_stats) if llm_stats else 0.0
                    )
                )
                conversation_history.turns.append(conversation_turn)

                # CRITICAL TEST: Turn 10 (index 10) should NOT crash
                if turn_index == 10:
                    logger.info("✓ CRITICAL TEST PASSED: Turn 10 did not crash!")
                    logger.info(f"  Phase after turn: {state.conversation_phase}")
                    logger.info(f"  Response generated successfully")

                    # Verify we're in the right phase (should be ACTION_PLANNING or WRAPUP)
                    assert state.conversation_phase in [
                        ConversationPhase.ACTION_PLANNING,
                        ConversationPhase.WRAPUP
                    ], f"Expected ACTION_PLANNING or WRAPUP, got {state.conversation_phase}"

                # Check if conversation finished
                if response.finished:
                    logger.info("Conversation marked as finished")
                    break

            except Exception as e:
                logger.error(f"Error at turn {turn_index}: {type(e).__name__}: {e}")
                conv_logger.log_error(turn_index, e)

                # If error is at turn 10, this is the bug we're testing for
                if turn_index == 10:
                    raise AssertionError(
                        f"CRITICAL TEST FAILED: Turn 10 crashed with error: {type(e).__name__}: {e}"
                    ) from e
                else:
                    raise

        # Log success
        conv_logger.log_success()
        logger.info(f"\n{'='*80}")
        logger.info("TEST PASSED - All turns completed successfully!")
        logger.info(f"Log saved to: {log_file}")
        logger.info(f"{'='*80}\n")

    except Exception as e:
        logger.error(f"Test failed: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    # Run the test directly
    asyncio.run(test_action_planning_intent_classification())
