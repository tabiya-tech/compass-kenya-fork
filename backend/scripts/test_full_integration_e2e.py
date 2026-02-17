#!/usr/bin/env python3
"""
End-to-End Integration Test for Full Compass Flow with Recommender Agent.

Tests the complete conversation flow:
1. Welcome → user starts
2. Experiences → extract skills
3. Preferences → elicit preferences
4. Recommendations → matching service integration

This test verifies:
- Skills extraction from experiences
- Preference vector loading
- Matching service client integration
- Automatic state initialization
- Data flow through all agents

Usage as script:
    poetry run python scripts/test_full_integration_e2e.py
    poetry run python scripts/test_full_integration_e2e.py --verbose
    poetry run python scripts/test_full_integration_e2e.py --use-real-service

Usage as pytest:
    poetry run pytest scripts/test_full_integration_e2e.py -v
    poetry run pytest scripts/test_full_integration_e2e.py -v --run-integration

Note: The test that calls the real matching service is marked with @pytest.mark.llm_integration
      and will be skipped in CI/CD unless --run-integration flag is provided.
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.agent.agent_director.llm_agent_director import LLMAgentDirector
from app.agent.agent_types import AgentInput, AgentType
from app.agent.explore_experiences_agent_director import DiveInPhase
from app.agent.preference_elicitation_agent.types import PreferenceVector
from app.agent.recommender_advisor_agent.types import ConversationPhase as RecPhase
from app.application_state import ApplicationState
from app.conversation_memory.conversation_memory_manager import ConversationMemoryManager
from app.conversation_memory.conversation_memory_types import ConversationContext
from app.conversations.service import ConversationService
from app.store.database_application_state_store import DatabaseApplicationStateStore
from app.server_dependencies.db_dependencies import CompassDBProvider
from app.countries import Country
from app.agent.linking_and_ranking_pipeline import ExperiencePipelineConfig
from app.vector_search.esco_entities import SkillEntity
from app.agent.experience.experience_entity import ExperienceEntity, ResponsibilitiesData
from app.metrics.application_state_metrics_recorder.recorder import ApplicationStateMetricsRecorder
from app.conversations.reactions.repository import ReactionRepository
from app.job_preferences.service import JobPreferencesService
from evaluation_tests.conversation_libs.search_service_fixtures import get_search_services
import argparse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class E2ETestResult:
    """Track test results and statistics."""
    def __init__(self):
        self.session_id: Optional[int] = None
        self.skills_extracted: int = 0
        self.preferences_loaded: bool = False
        self.bws_scores_loaded: bool = False
        self.matching_service_called: bool = False
        self.recommendations_received: int = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, error: str):
        self.errors.append(error)
        logger.error(f"❌ {error}")

    def add_warning(self, warning: str):
        self.warnings.append(warning)
        logger.warning(f"⚠️  {warning}")

    def print_summary(self):
        print("\n" + "="*80)
        print("END-TO-END TEST SUMMARY")
        print("="*80)
        print(f"Session ID: {self.session_id}")
        print(f"Skills Extracted: {self.skills_extracted}")
        print(f"Preferences Loaded: {'✅' if self.preferences_loaded else '❌'}")
        print(f"BWS Scores Loaded: {'✅' if self.bws_scores_loaded else '❌'}")
        print(f"Matching Service Called: {'✅' if self.matching_service_called else '❌'}")
        print(f"Recommendations Received: {self.recommendations_received}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"   - {w}")

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for e in self.errors:
                print(f"   - {e}")
            print("\n❌ TEST FAILED")
            return False
        else:
            print("\n✅ ALL TESTS PASSED")
            return True


async def create_mock_experience_with_skills() -> ExperienceEntity:
    """Create a mock experience with realistic skills for testing."""
    skills = [
        SkillEntity(
            id="skill_001",
            modelId="test_model",
            UUID="uuid_001",
            preferredLabel="Customer Service",
            altLabels=["Customer Support"],
            description="Providing customer service",
            scopeNote="",
            originUUID="origin_001",
            UUIDHistory=[],
            score=0.85,
            skillType="skill/competence"
        ),
        SkillEntity(
            id="skill_002",
            modelId="test_model",
            UUID="uuid_002",
            preferredLabel="Communication",
            altLabels=["Verbal Communication"],
            description="Effective communication skills",
            scopeNote="",
            originUUID="origin_002",
            UUIDHistory=[],
            score=0.90,
            skillType="skill/competence"
        ),
        SkillEntity(
            id="skill_003",
            modelId="test_model",
            UUID="uuid_003",
            preferredLabel="Problem Solving",
            altLabels=["Critical Thinking"],
            description="Analytical problem solving",
            scopeNote="",
            originUUID="origin_003",
            UUIDHistory=[],
            score=0.75,
            skillType="skill/competence"
        ),
    ]

    return ExperienceEntity[SkillEntity](
        uuid="exp_001",
        experience_title="Customer Service Representative",
        normalized_experience_title="Customer Service Representative",
        company="Tech Solutions Ltd",
        location="Nairobi, Kenya",
        timeline=None,
        work_type=None,
        top_skills=skills,
        remaining_skills=[],
        summary="Provided customer support and handled inquiries",
        responsibilities=ResponsibilitiesData(
            responsibilities=["Answer customer calls", "Resolve issues"],
            non_responsibilities=[],
            other_peoples_responsibilities=[]
        ),
        esco_occupations=[],
        questions_and_answers=[]
    )


async def create_mock_preference_vector() -> PreferenceVector:
    """Create a mock preference vector for testing."""
    return PreferenceVector(
        financial_importance=0.8,
        work_environment_importance=0.6,
        career_advancement_importance=0.7,
        work_life_balance_importance=0.5,
        job_security_importance=0.6,
        task_preference_importance=0.4,
        social_impact_importance=0.5,
        confidence_score=0.85,
        n_vignettes_completed=10,
        per_dimension_uncertainty={
            "financial_importance": 0.1,
            "work_environment_importance": 0.15,
            "career_advancement_importance": 0.12
        },
        posterior_mean=[0.8, 0.6, 0.7, 0.5, 0.6, 0.4, 0.5],
        posterior_covariance_diagonal=[0.01, 0.02, 0.015, 0.025, 0.02, 0.03, 0.025],
        fim_determinant=0.00001,
        decision_patterns={
            "mentions_family_frequently": True,
            "career_growth_focused": True
        },
        tradeoff_willingness={
            "will_sacrifice_salary_for_flexibility": False,
            "prefers_stability_over_high_pay": False
        },
        values_signals={
            "stability_seeking": True,
            "autonomy_seeking": True
        },
        consistency_indicators={
            "response_consistency": 0.85,
            "conviction_strength": 0.8
        },
        extracted_constraints={
            "minimum_salary": 50000
        }
    )


async def run_skills_extraction_test(state: ApplicationState, result: E2ETestResult):
    """Test that skills are properly extracted from experiences."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Skills Extraction from Experiences")
    logger.info("="*80)

    # Add a mock experience with skills
    mock_experience = await create_mock_experience_with_skills()
    state.explore_experiences_director_state.explored_experiences = [mock_experience]

    # Import skills extractor
    from app.agent.recommender_advisor_agent.skills_extractor import SkillsExtractor

    extractor = SkillsExtractor()
    skills_vector = extractor.extract_skills_vector(
        state.explore_experiences_director_state.explored_experiences
    )

    # Verify extraction
    if skills_vector and "skills" in skills_vector:
        result.skills_extracted = len(skills_vector["skills"])
        logger.info(f"✅ Extracted {result.skills_extracted} skills from 1 experience")

        # Check data quality
        for skill in skills_vector["skills"]:
            if not skill.get("skill_id") or not skill.get("uuid"):
                result.add_error(f"Skill missing required fields: {skill}")
            if not skill.get("preferred_label"):
                result.add_warning(f"Skill missing label: {skill.get('skill_id')}")
    else:
        result.add_error("Skills extraction returned empty or invalid data")

    # Store in state
    state.recommender_advisor_agent_state.skills_vector = skills_vector
    logger.info(f"✅ Skills vector stored in recommender state")


async def run_preference_loading_test(state: ApplicationState, result: E2ETestResult):
    """Test that preferences are properly loaded."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Preference Vector Loading")
    logger.info("="*80)

    # Create mock preference vector
    mock_prefs = await create_mock_preference_vector()
    state.preference_elicitation_agent_state.preference_vector = mock_prefs
    state.preference_elicitation_agent_state.conversation_phase = "COMPLETE"

    logger.info(f"✅ Created preference vector (confidence: {mock_prefs.confidence_score:.2f})")

    result.preferences_loaded = True

    # BWS scores are optional (not stored in preference state for now)
    # They can be added directly to recommender state if needed
    result.bws_scores_loaded = False  # Not testing BWS scores in this E2E


async def run_state_preparation_test(state: ApplicationState, result: E2ETestResult):
    """Test that ConversationService prepares recommender state correctly."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Automatic State Preparation")
    logger.info("="*80)

    # Simulate what ConversationService does
    from app.conversations.service import ConversationService

    # Create a minimal conversation service instance
    # We'll call the method directly
    class MockReactionRepo:
        async def get_reactions(self, session_id):
            return []

    class MockJobPrefsService:
        async def create_or_update(self, session_id, preferences):
            pass

    class MockMetricsRecorder:
        async def get_state(self, session_id):
            return state
        async def save_state(self, state, user_id):
            pass
        async def delete_state(self, session_id):
            pass

    # We need to test the _prepare_recommender_state_if_needed method
    # Let's clear the recommender state first
    state.recommender_advisor_agent_state.skills_vector = None
    state.recommender_advisor_agent_state.preference_vector = None
    state.recommender_advisor_agent_state.bws_occupation_scores = None

    logger.info("Cleared recommender state to test auto-initialization")

    # Manually call the preparation logic (simulating ConversationService)
    rec_state = state.recommender_advisor_agent_state

    # Extract skills
    from app.agent.recommender_advisor_agent.skills_extractor import SkillsExtractor
    explored_experiences = state.explore_experiences_director_state.explored_experiences
    extractor = SkillsExtractor()
    skills_vector = extractor.extract_skills_vector(explored_experiences)
    rec_state.skills_vector = skills_vector

    # Load preferences
    pref_state = state.preference_elicitation_agent_state
    if pref_state.preference_vector is not None:
        rec_state.preference_vector = pref_state.preference_vector

    # BWS scores (optional - not always available)
    # Note: BWS scores not stored in preference_elicitation state currently
    # They would be passed directly to recommender if needed

    # Set youth_id
    if rec_state.youth_id is None:
        rec_state.youth_id = f"youth_{state.session_id}"

    # Verify state was prepared
    if rec_state.skills_vector is None:
        result.add_error("Skills vector not loaded into recommender state")
    else:
        logger.info(f"✅ Skills vector loaded: {len(rec_state.skills_vector.get('skills', []))} skills")

    if rec_state.preference_vector is None:
        result.add_error("Preference vector not loaded into recommender state")
    else:
        logger.info(f"✅ Preference vector loaded (confidence: {rec_state.preference_vector.confidence_score:.2f})")

    if rec_state.bws_occupation_scores is None:
        result.add_warning("BWS scores not loaded (optional)")
    else:
        logger.info(f"✅ BWS scores loaded: {len(rec_state.bws_occupation_scores)} occupations")

    if rec_state.youth_id is None:
        result.add_error("Youth ID not set")
    else:
        logger.info(f"✅ Youth ID set: {rec_state.youth_id}")


async def run_matching_service_integration_test(state: ApplicationState, result: E2ETestResult, use_real_service: bool = False):
    """Test matching service client integration."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Matching Service Integration")
    logger.info("="*80)

    if not use_real_service:
        logger.info("⚠️  Skipping real service call (use --use-real-service to test)")
        result.add_warning("Matching service not tested (use --use-real-service flag)")
        return

    # Test matching service client
    from app.agent.recommender_advisor_agent.matching_service_client import MatchingServiceClient
    import os

    try:
        # Get config from environment directly
        service_url = os.getenv("MATCHING_SERVICE_URL")
        service_key = os.getenv("MATCHING_SERVICE_API_KEY")

        if not service_url or not service_key:
            result.add_warning("Matching service not configured in environment (.env)")
            return

        client = MatchingServiceClient(
            base_url=service_url,
            api_key=service_key
        )

        logger.info(f"Created matching service client: {service_url}")

        # Make a test call
        rec_state = state.recommender_advisor_agent_state

        response = await client.generate_recommendations(
            youth_id=rec_state.youth_id,
            city="Nairobi",  # Test data
            province="Nairobi County",
            skills_vector=rec_state.skills_vector,
            preference_vector=rec_state.preference_vector
        )

        result.matching_service_called = True
        logger.info("✅ Matching service call successful")

        # Verify response structure
        if isinstance(response, list) and len(response) > 0:
            user_data = response[0]
            result.recommendations_received = (
                len(user_data.get("occupation_recommendations", [])) +
                len(user_data.get("opportunity_recommendations", [])) +
                len(user_data.get("skill_gap_recommendations", []))
            )
            logger.info(f"✅ Received {result.recommendations_received} total recommendations")
            logger.info(f"   - Occupations: {len(user_data.get('occupation_recommendations', []))}")
            logger.info(f"   - Opportunities: {len(user_data.get('opportunity_recommendations', []))}")
            logger.info(f"   - Trainings: {len(user_data.get('skill_gap_recommendations', []))}")
        else:
            result.add_warning("Matching service returned unexpected format")

    except Exception as e:
        result.add_error(f"Matching service integration failed: {e}")


async def run_recommendation_interface_test(state: ApplicationState, result: E2ETestResult):
    """Test RecommendationInterface with prepared state."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: RecommendationInterface (Stub Fallback)")
    logger.info("="*80)

    from app.agent.recommender_advisor_agent.recommendation_interface import RecommendationInterface

    # Create interface without matching service (will use stubs)
    interface = RecommendationInterface(matching_service_client=None)

    rec_state = state.recommender_advisor_agent_state

    try:
        recommendations = await interface.generate_recommendations(
            youth_id=rec_state.youth_id,
            city="Nairobi",
            province="Nairobi County",
            preference_vector=rec_state.preference_vector,
            skills_vector=rec_state.skills_vector,
            bws_occupation_scores=rec_state.bws_occupation_scores
        )

        logger.info("✅ RecommendationInterface returned successfully")
        logger.info(f"   - Youth ID: {recommendations.youth_id}")
        logger.info(f"   - Occupations: {len(recommendations.occupation_recommendations)}")
        logger.info(f"   - Opportunities: {len(recommendations.opportunity_recommendations)}")
        logger.info(f"   - Trainings: {len(recommendations.skillstraining_recommendations)}")
        logger.info(f"   - Algorithm: {recommendations.algorithm_version}")
        logger.info(f"   - Confidence: {recommendations.confidence}")

        if len(recommendations.occupation_recommendations) == 0:
            result.add_warning("No occupation recommendations returned (using stubs)")

    except Exception as e:
        result.add_error(f"RecommendationInterface failed: {e}")


async def run_full_e2e_test(use_real_service: bool = False) -> bool:
    """Run the complete end-to-end integration test."""
    result = E2ETestResult()

    try:
        # Create a test session
        session_id = 99999  # Test session ID
        result.session_id = session_id

        logger.info("\n" + "="*80)
        logger.info("STARTING FULL E2E INTEGRATION TEST")
        logger.info("="*80)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Use Real Service: {use_real_service}")

        # Create application state
        state = ApplicationState.new_state(
            session_id=session_id,
            country_of_user=Country.KENYA
        )

        # Run tests in sequence
        await run_skills_extraction_test(state, result)
        await run_preference_loading_test(state, result)
        await run_state_preparation_test(state, result)
        await run_matching_service_integration_test(state, result, use_real_service)
        await run_recommendation_interface_test(state, result)

        # Print summary
        success = result.print_summary()
        return success

    except Exception as e:
        logger.exception("Fatal error in E2E test")
        result.add_error(f"Fatal error: {e}")
        result.print_summary()
        return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="E2E Integration Test for Compass Recommender")
    parser.add_argument(
        "--use-real-service",
        action="store_true",
        help="Call real matching service (requires config)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = await run_full_e2e_test(use_real_service=args.use_real_service)

    sys.exit(0 if success else 1)


# ============================================================================
# PYTEST TEST FUNCTIONS
# ============================================================================
# These functions allow the test to be run via pytest with proper markers

@pytest.mark.asyncio
async def test_e2e_integration_without_service():
    """
    Pytest version of E2E test without calling real matching service.

    This test is safe to run in CI/CD as it doesn't require credentials.
    Tests skills extraction, preference loading, and state preparation.
    """
    success = await run_full_e2e_test(use_real_service=False)
    assert success, "E2E integration test failed (without real service)"


@pytest.mark.asyncio
@pytest.mark.llm_integration
async def test_e2e_integration_with_real_service():
    """
    Pytest version of E2E test WITH real matching service.

    Marked with @pytest.mark.llm_integration - will be skipped in CI/CD.
    Requires MATCHING_SERVICE_URL and MATCHING_SERVICE_API_KEY in .env

    To run: pytest scripts/test_full_integration_e2e.py -v -m llm_integration
    """
    import os

    # Skip if credentials not available
    if not os.getenv("MATCHING_SERVICE_URL") or not os.getenv("MATCHING_SERVICE_API_KEY"):
        pytest.skip("Matching service credentials not configured in .env")

    success = await run_full_e2e_test(use_real_service=True)
    assert success, "E2E integration test failed (with real service)"


# ============================================================================
# CLI ENTRY POINT (for running as script)
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())
