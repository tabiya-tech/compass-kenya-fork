#!/usr/bin/env python3
"""
Debug script to test occupation switching bug.

This script runs a minimal conversation with specific inputs that trigger
the occupation switch bug, captures the intent classifier output, and
checks if the current_focus_id updates correctly.

Usage:
    poetry run python scripts/debug_occupation_switch.py
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import ConversationPhase
from app.agent.agent_types import AgentInput
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
    ConversationTurn,
)
from app.conversation_memory.conversation_formatter import ConversationHistoryFormatter
from app.agent.agent_types import AgentOutput
from app.countries import Country

# Import handlers directly
from app.agent.recommender_advisor_agent.phase_handlers.intro_handler import IntroPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.present_handler import PresentPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.exploration_handler import ExplorationPhaseHandler
from app.agent.recommender_advisor_agent.intent_classifier import IntentClassifier
from app.agent.recommender_advisor_agent.llm_response_models import (
    ConversationResponse,
    UserIntentClassification,
)
from app.agent.llm_caller import LLMCaller
from app.agent.recommender_advisor_agent.recommendation_interface import RecommendationInterface
from common_libs.llm.generative_models import GeminiGenerativeLLM
from common_libs.llm.models_utils import LLMConfig, LOW_TEMPERATURE_GENERATION_CONFIG, JSON_GENERATION_CONFIG

# Import sample data creation functions
from test_recommender_agent_interactive import (
    create_sample_recommendations,
    create_sample_skills_vector,
    create_sample_preference_vector,
)


def print_separator():
    print("\n" + "=" * 100 + "\n")


def print_step(step_num: int, description: str):
    print(f"\n{'#' * 100}")
    print(f"# STEP {step_num}: {description}")
    print(f"{'#' * 100}\n")


async def main():
    print_separator()
    print("DEBUG SCRIPT: Testing Occupation Switch Bug")
    print_separator()

    # Initialize handlers (like the test script does)
    print_step(1, "Initializing Phase Handlers")

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

    intent_caller = LLMCaller[UserIntentClassification](
        model_response_type=UserIntentClassification
    )

    # Initialize intent classifier
    intent_classifier = IntentClassifier(intent_caller=intent_caller)

    # Initialize recommendation interface
    recommendation_interface = RecommendationInterface(node2vec_client=None)

    # Initialize handlers
    intro_handler = IntroPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        recommendation_interface=recommendation_interface,
        occupation_search_service=None
    )

    present_handler = PresentPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
        occupation_search_service=None
    )

    exploration_handler = ExplorationPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
        occupation_search_service=None
    )

    handlers = {
        ConversationPhase.INTRO: intro_handler,
        ConversationPhase.PRESENT_RECOMMENDATIONS: present_handler,
        ConversationPhase.CAREER_EXPLORATION: exploration_handler,
    }

    print("✓ Handlers initialized")

    # Create initial state
    print_step(2, "Creating initial state with sample data")
    recommendations = create_sample_recommendations()
    skills_vector = create_sample_skills_vector()
    preference_vector = create_sample_preference_vector()

    state = RecommenderAdvisorAgentState(
        session_id="test_session_debug",
        youth_id="test_user_debug",
        country_of_user=Country.KENYA,
        conversation_phase=ConversationPhase.INTRO,
        recommendations=recommendations,
        skills_vector=skills_vector,
        preference_vector=preference_vector
    )

    print(f"✓ State created")
    print(f"  - Session ID: {state.session_id}")
    print(f"  - Youth ID: {state.youth_id}")
    print(f"  - Recommendations loaded: {len(state.recommendations.occupation_recommendations)} occupations")
    print(f"  - Initial phase: {state.conversation_phase}")

    # Print available occupations for reference
    print("\n📋 Available Occupations:")
    for i, occ in enumerate(state.recommendations.occupation_recommendations[:5], 1):
        print(f"  {i}. {occ.occupation} (uuid: {occ.uuid})")

    # Create conversation context
    conversation_history = ConversationHistory()
    context = ConversationContext(
        all_history=conversation_history,
        history=conversation_history,
        summary="",
    )

    # Test sequence
    test_inputs = [
        {
            "turn": 0,
            "input": "",
            "description": "INTRO - Agent introduces recommendations",
            "expected_phase": ConversationPhase.PRESENT_RECOMMENDATIONS,
            "expected_focus": None,
        },
        {
            "turn": 1,
            "input": "Okay",
            "description": "User acknowledges intro",
            "expected_phase": ConversationPhase.PRESENT_RECOMMENDATIONS,
            "expected_focus": None,
        },
        {
            "turn": 2,
            "input": "tell me about 4",
            "description": "User asks about occupation #4 (Boat/Marine Equipment Fundi)",
            "expected_phase": ConversationPhase.CAREER_EXPLORATION,
            "expected_focus": "occ_004_uuid",  # Boat Fundi
        },
        {
            "turn": 3,
            "input": "okay then fundi wa stima inakaa kua poa",
            "description": "🔴 BUG TEST: User switches to Electrician (occupation #1)",
            "expected_phase": ConversationPhase.CAREER_EXPLORATION,
            "expected_focus": "occ_001_uuid",  # Should switch to Electrician
        },
    ]

    bug_detected = False

    for test in test_inputs:
        print_step(test["turn"] + 3, test["description"])

        # Show current state before execution
        print(f"📊 State BEFORE execution:")
        print(f"  - Phase: {state.conversation_phase}")
        print(f"  - Current Focus ID: {state.current_focus_id}")
        if state.current_focus_id:
            rec = state.get_recommendation_by_id(state.current_focus_id)
            if rec:
                print(f"  - Current Focus Name: {rec.occupation}")

        # Execute handler
        print(f"\n💬 User Input: \"{test['input']}\"")
        print(f"\n⚙️  Executing handler...")

        # Get the appropriate handler for current phase
        handler = handlers.get(state.conversation_phase)
        if not handler:
            print(f"❌ No handler for phase: {state.conversation_phase}")
            continue

        # Execute handler
        response, llm_stats = await handler.handle(test["input"], state, context)

        # Create output object for compatibility
        output = type('Output', (), {
            'message_for_user': response.message,
            'finished': response.finished
        })()

        # Show agent response
        print(f"\n🤖 Agent Response:")
        print(f"  {output.message_for_user[:200]}..." if len(output.message_for_user) > 200 else f"  {output.message_for_user}")

        # Show state after execution
        print(f"\n📊 State AFTER execution:")
        print(f"  - Phase: {state.conversation_phase}")
        print(f"  - Current Focus ID: {state.current_focus_id}")
        if state.current_focus_id:
            rec = state.get_recommendation_by_id(state.current_focus_id)
            if rec:
                print(f"  - Current Focus Name: {rec.occupation}")

        # Verify expectations
        print(f"\n✅ Verification:")
        phase_match = state.conversation_phase == test["expected_phase"]
        focus_match = state.current_focus_id == test["expected_focus"]

        print(f"  - Expected Phase: {test['expected_phase']}")
        print(f"  - Actual Phase: {state.conversation_phase}")
        print(f"  - Phase Match: {'✓ PASS' if phase_match else '✗ FAIL'}")

        print(f"  - Expected Focus: {test['expected_focus']}")
        print(f"  - Actual Focus: {state.current_focus_id}")
        print(f"  - Focus Match: {'✓ PASS' if focus_match else '✗ FAIL'}")

        # Check for bug in turn 3
        if test["turn"] == 3:
            if not focus_match:
                bug_detected = True
                print(f"\n🔴 BUG DETECTED!")
                print(f"  User said: '{test['input']}'")
                print(f"  Agent generated response about: Electrician")
                print(f"  But current_focus_id is still: {state.current_focus_id}")

                # Show which occupation the state thinks we're focused on
                if state.current_focus_id:
                    rec = state.get_recommendation_by_id(state.current_focus_id)
                    if rec:
                        print(f"  State thinks we're focused on: {rec.occupation}")

                print(f"\n  Expected: occ_001_uuid (Fundi wa Stima/Electrician)")
                print(f"  This means the intent classifier failed to:")
                print(f"    1. Detect occupation switch intent, OR")
                print(f"    2. Extract target_recommendation_id/target_occupation_index")
            else:
                print(f"\n✅ BUG NOT DETECTED - Focus correctly updated to Electrician!")

        print_separator()

    # Final summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    if bug_detected:
        print("\n🔴 BUG CONFIRMED")
        print("The occupation switch from Boat Fundi to Electrician DID NOT update current_focus_id")
        print("\nCheck the debug output above (marked with [INTENT CLASSIFIER DEBUG]) to see:")
        print("  1. What intent was classified")
        print("  2. Whether target_recommendation_id was extracted")
        print("  3. Whether target_occupation_index was extracted")
        return 1
    else:
        print("\n✅ NO BUG DETECTED")
        print("The occupation switch worked correctly!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
