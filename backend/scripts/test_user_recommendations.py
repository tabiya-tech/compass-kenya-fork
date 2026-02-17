#!/usr/bin/env python3
"""
Test recommendations for a specific user.

Fetches user data from database and generates recommendations.

Usage:
    poetry run python scripts/test_user_recommendations.py MaaUmBy38fOVo19Zqf9mqiTjWi52
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from motor.motor_asyncio import AsyncIOMotorClient
from app.app_config import get_application_config
from app.agent.recommender_advisor_agent.matching_service_client import MatchingServiceClient
from app.agent.recommender_advisor_agent.recommendation_interface import RecommendationInterface
from app.agent.recommender_advisor_agent.skills_extractor import SkillsExtractor
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def get_user_data(user_id: str):
    """Fetch user data from MongoDB."""
    logger.info(f"Fetching data for user: {user_id}")

    # Get MongoDB connection
    mongo_uri = os.getenv("APPLICATION_MONGODB_URI")
    if not mongo_uri:
        logger.error("APPLICATION_MONGODB_URI not set in environment")
        return None

    client = AsyncIOMotorClient(mongo_uri)

    try:
        # Get application database
        db_name = os.getenv("APPLICATION_DATABASE_NAME", "compass-kenya-application-local")
        db = client[db_name]
        logger.info(f"Connected to database: {db_name}")

        # Find user's latest session
        application_state_collection = db["applicationState"]
        user_states = await application_state_collection.find(
            {"user_id": user_id}
        ).sort("session_id", -1).to_list(length=10)

        if not user_states:
            logger.error(f"No sessions found for user {user_id}")
            return None

        logger.info(f"Found {len(user_states)} sessions for user")

        # Get the latest session
        latest_state = user_states[0]
        session_id = latest_state.get("session_id")

        logger.info(f"Latest session: {session_id}")
        logger.info(f"State keys: {list(latest_state.keys())}")

        # Extract relevant data
        user_data = {
            "session_id": session_id,
            "user_id": user_id,
            "experiences": [],
            "preferences": None,
            "skills_vector": None
        }

        # Get experiences
        explore_exp_state = latest_state.get("explore_experiences_director_state", {})
        explored_experiences = explore_exp_state.get("explored_experiences", [])

        logger.info(f"Found {len(explored_experiences)} explored experiences")

        if explored_experiences:
            user_data["experiences"] = explored_experiences

            # Extract skills from experiences
            from app.agent.experience.experience_entity import ExperienceEntity
            from app.vector_search.esco_entities import SkillEntity

            # Convert to ExperienceEntity objects
            experience_objects = []
            for exp_data in explored_experiences:
                try:
                    exp = ExperienceEntity[SkillEntity].model_validate(exp_data)
                    experience_objects.append(exp)
                except Exception as e:
                    logger.warning(f"Failed to parse experience: {e}")

            if experience_objects:
                extractor = SkillsExtractor()
                user_data["skills_vector"] = extractor.extract_skills_vector(experience_objects)
                logger.info(f"Extracted skills: {len(user_data['skills_vector'].get('skills', []))} skills")

        # Get preferences
        pref_state = latest_state.get("preference_elicitation_agent_state", {})
        pref_vector = pref_state.get("preference_vector")

        if pref_vector:
            user_data["preferences"] = pref_vector
            logger.info(f"Found preference vector with confidence: {pref_vector.get('confidence_score', 'N/A')}")
        else:
            logger.warning("No preference vector found")

        return user_data

    finally:
        client.close()


async def test_recommendations(user_id: str):
    """Test generating recommendations for a user."""

    # Get user data
    user_data = await get_user_data(user_id)

    if not user_data:
        logger.error("Failed to fetch user data")
        return

    print("\n" + "="*80)
    print("USER DATA SUMMARY")
    print("="*80)
    print(f"User ID: {user_data['user_id']}")
    print(f"Session ID: {user_data['session_id']}")
    print(f"Experiences: {len(user_data['experiences'])}")

    if user_data['skills_vector']:
        skills = user_data['skills_vector'].get('skills', [])
        print(f"Skills Extracted: {len(skills)}")
        if skills:
            print(f"\nTop 5 Skills:")
            for i, skill in enumerate(skills[:5], 1):
                print(f"  {i}. {skill.get('preferred_label')} (proficiency: {skill.get('proficiency', 'N/A'):.2f})")
    else:
        print("Skills Extracted: 0 (no experiences)")

    if user_data['preferences']:
        prefs = user_data['preferences']
        print(f"\nPreferences:")
        print(f"  - Financial Importance: {prefs.get('financial_importance', 'N/A')}")
        print(f"  - Work Environment: {prefs.get('work_environment_importance', 'N/A')}")
        print(f"  - Career Advancement: {prefs.get('career_advancement_importance', 'N/A')}")
        print(f"  - Work-Life Balance: {prefs.get('work_life_balance_importance', 'N/A')}")
        print(f"  - Job Security: {prefs.get('job_security_importance', 'N/A')}")
        print(f"  - Confidence Score: {prefs.get('confidence_score', 'N/A')}")
    else:
        print("\nPreferences: Not available")

    # Test matching service
    print("\n" + "="*80)
    print("TESTING MATCHING SERVICE")
    print("="*80)

    service_url = os.getenv("MATCHING_SERVICE_URL")
    service_key = os.getenv("MATCHING_SERVICE_API_KEY")

    if not service_url or not service_key:
        print("❌ Matching service not configured (missing URL or API key)")
        return

    # Create matching service client
    client = MatchingServiceClient(
        base_url=service_url,
        api_key=service_key
    )

    # Create recommendation interface
    interface = RecommendationInterface(matching_service_client=client)

    try:
        # Generate recommendations
        recommendations = await interface.generate_recommendations(
            youth_id=user_data['user_id'],
            city="Nairobi",  # Default for now
            province="Nairobi County",
            skills_vector=user_data['skills_vector'],
            preference_vector=user_data['preferences'],
            bws_occupation_scores=None
        )

        print("\n✅ Recommendations Generated Successfully!")
        print(f"\nYouth ID: {recommendations.youth_id}")
        print(f"Algorithm: {recommendations.algorithm_version}")
        print(f"Confidence: {recommendations.confidence}")

        print(f"\n📊 Recommendations Summary:")
        print(f"  - Occupations: {len(recommendations.occupation_recommendations)}")
        print(f"  - Opportunities: {len(recommendations.opportunity_recommendations)}")
        print(f"  - Trainings: {len(recommendations.skillstraining_recommendations)}")

        if recommendations.occupation_recommendations:
            print(f"\n🎯 Top 5 Occupation Recommendations:")
            for i, occ in enumerate(recommendations.occupation_recommendations[:5], 1):
                print(f"\n  {i}. {occ.occupation}")
                print(f"     Rank: {occ.rank}")
                print(f"     Eligible: {occ.is_eligible}")
                if occ.final_score:
                    print(f"     Score: {occ.final_score:.3f}")
                if occ.justification:
                    print(f"     Why: {occ.justification[:100]}...")

        if recommendations.opportunity_recommendations:
            print(f"\n💼 Top 3 Job Opportunities:")
            for i, opp in enumerate(recommendations.opportunity_recommendations[:3], 1):
                print(f"\n  {i}. {opp.opportunity_title}")
                if opp.employer:
                    print(f"     Employer: {opp.employer}")
                if opp.location:
                    print(f"     Location: {opp.location}")
                if opp.salary_range:
                    print(f"     Salary: {opp.salary_range}")

        if recommendations.skillstraining_recommendations:
            print(f"\n📚 Training Recommendations:")
            for i, training in enumerate(recommendations.skillstraining_recommendations[:3], 1):
                print(f"\n  {i}. {training.training_title}")
                if training.skill_to_learn:
                    print(f"     Skill: {training.skill_to_learn}")
                if training.provider:
                    print(f"     Provider: {training.provider}")

        print("\n" + "="*80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: poetry run python scripts/test_user_recommendations.py <user_id>")
        print("Example: poetry run python scripts/test_user_recommendations.py MaaUmBy38fOVo19Zqf9mqiTjWi52")
        sys.exit(1)

    user_id = sys.argv[1]
    await test_recommendations(user_id)


if __name__ == "__main__":
    asyncio.run(main())
