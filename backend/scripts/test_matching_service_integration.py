"""
Test script for Matching Service integration.

Tests the full integration between the Recommender Advisor Agent
and the deployed matching service.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.agent.recommender_advisor_agent.matching_service_client import (
    MatchingServiceClient,
    MatchingServiceError
)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_matching_service_connection():
    """Test basic HTTP connection to matching service."""
    print("\n" + "="*80)
    print("TEST 1: Basic Connection Test")
    print("="*80)

    # Initialize client with dev credentials
    client = MatchingServiceClient(
        base_url="https://matching-gateway-dev-9baomanq.uc.gateway.dev/match",
        api_key="AIzaSyAdirPom5z8sXSzkmSHs2xXFkF8j7O9yKY",
        timeout=30.0
    )

    # Minimal test payload
    try:
        result = await client.generate_recommendations(
            youth_id="test_user_001",
            city="Johannesburg",
            province="Gauteng",
            skills_vector={
                "skills": [],
                "total_experiences": 0,
                "extraction_metadata": {}
            },
            preference_vector=None
        )

        print("\n✅ Connection successful!")
        print(f"Response type: {type(result)}")
        print(f"Response keys: {result[0].keys() if isinstance(result, list) and len(result) > 0 else 'N/A'}")
        print(f"\nSample response:")
        import json
        print(json.dumps(result, indent=2)[:500] + "...")

        return True

    except MatchingServiceError as e:
        print(f"\n❌ Matching service error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_skills_transformation():
    """Test skills vector transformation."""
    print("\n" + "="*80)
    print("TEST 2: Skills Vector Transformation")
    print("="*80)

    client = MatchingServiceClient(
        base_url="https://matching-gateway-dev-9baomanq.uc.gateway.dev/match",
        api_key="AIzaSyAdirPom5z8sXSzkmSHs2xXFkF8j7O9yKY"
    )

    # Sample skills vector from SkillsExtractor
    skills_vector = {
        "skills": [
            {
                "skill_id": "skill_001",
                "uuid": "uuid_001",
                "preferred_label": "Python Programming",
                "skill_type": "skill/competence",
                "proficiency": 0.85,
                "frequency": 2,
                "avg_score": 0.80
            },
            {
                "skill_id": "skill_002",
                "uuid": "uuid_002",
                "preferred_label": "Git Version Control",
                "skill_type": "skill/competence",
                "proficiency": 0.70,
                "frequency": 1,
                "avg_score": 0.70
            }
        ],
        "total_experiences": 2,
        "extraction_metadata": {
            "top_skills_processed": 2,
            "remaining_skills_processed": 0,
            "unique_skills": 2
        }
    }

    # Test transformation
    transformed = client._transform_skills_vector(skills_vector)
    print(f"\n✅ Transformed skills vector:")
    print(f"  - top_skills count: {len(transformed.get('top_skills', []))}")
    print(f"  - Sample skill: {transformed['top_skills'][0] if transformed['top_skills'] else 'None'}")

    # Test with matching service
    try:
        result = await client.generate_recommendations(
            youth_id="test_user_002",
            city="Nairobi",
            province="Nairobi County",
            skills_vector=skills_vector,
            preference_vector={
                "earnings_per_month": 0.8,
                "task_content": 0.6,
                "physical_demand": 0.3,
                "work_flexibility": 0.7,
                "social_interaction": 0.5,
                "career_growth": 0.9,
                "social_meaning": 0.6
            }
        )

        print(f"\n✅ Matching service accepted transformed skills!")
        print(f"  - Occupation recommendations: {len(result[0].get('occupation_recommendations', []))}")
        print(f"  - Opportunity recommendations: {len(result[0].get('opportunity_recommendations', []))}")
        print(f"  - Skill gap recommendations: {len(result[0].get('skill_gap_recommendations', []))}")

        return True

    except Exception as e:
        print(f"\n❌ Error with skills transformation: {e}")
        return False


async def test_preference_transformation():
    """Test preference vector transformation."""
    print("\n" + "="*80)
    print("TEST 3: Preference Vector Transformation")
    print("="*80)

    client = MatchingServiceClient(
        base_url="https://matching-gateway-dev-9baomanq.uc.gateway.dev/match",
        api_key="AIzaSyAdirPom5z8sXSzkmSHs2xXFkF8j7O9yKY"
    )

    # Test with Pydantic model (mock)
    class MockPreferenceVector:
        def model_dump(self):
            return {
                "earnings_per_month": 0.7,
                "task_content": 0.5,
                "physical_demand": 0.4,
                "work_flexibility": 0.8,
                "social_interaction": 0.6,
                "career_growth": 0.9,
                "social_meaning": 0.7
            }

    pref_obj = MockPreferenceVector()
    transformed = client._transform_preference_vector(pref_obj)

    print(f"\n✅ Transformed preference vector:")
    for key, value in transformed.items():
        print(f"  - {key}: {value}")

    # Test with None
    default_prefs = client._transform_preference_vector(None)
    print(f"\n✅ Default preferences (when None):")
    print(f"  - All values set to: {default_prefs['earnings_per_month']}")

    return True


async def test_end_to_end_flow():
    """Test complete end-to-end recommendation flow."""
    print("\n" + "="*80)
    print("TEST 4: End-to-End Recommendation Flow")
    print("="*80)

    from app.agent.recommender_advisor_agent.recommendation_interface import RecommendationInterface
    from app.agent.recommender_advisor_agent.types import Node2VecRecommendations

    # Create matching service client
    client = MatchingServiceClient(
        base_url="https://matching-gateway-dev-9baomanq.uc.gateway.dev/match",
        api_key="AIzaSyAdirPom5z8sXSzkmSHs2xXFkF8j7O9yKY"
    )

    # Create recommendation interface
    interface = RecommendationInterface(matching_service_client=client)

    # Generate recommendations
    try:
        recommendations = await interface.generate_recommendations(
            youth_id="test_user_e2e",
            city="Mombasa",
            province="Mombasa County",
            skills_vector={
                "skills": [
                    {
                        "skill_id": "test_skill_001",
                        "uuid": "test_uuid_001",
                        "preferred_label": "Customer Service",
                        "skill_type": "skill/competence",
                        "proficiency": 0.75,
                        "avg_score": 0.70
                    }
                ],
                "total_experiences": 1,
                "extraction_metadata": {}
            },
            preference_vector={
                "earnings_per_month": 0.6,
                "task_content": 0.5,
                "physical_demand": 0.4,
                "work_flexibility": 0.7,
                "social_interaction": 0.8,
                "career_growth": 0.6,
                "social_meaning": 0.5
            }
        )

        print(f"\n✅ Recommendations generated successfully!")
        print(f"  - Type: {type(recommendations)}")
        print(f"  - Youth ID: {recommendations.youth_id}")
        print(f"  - Occupation recommendations: {len(recommendations.occupation_recommendations)}")
        print(f"  - Opportunity recommendations: {len(recommendations.opportunity_recommendations)}")
        print(f"  - Training recommendations: {len(recommendations.skillstraining_recommendations)}")
        print(f"  - Algorithm version: {recommendations.algorithm_version}")
        print(f"  - Confidence: {recommendations.confidence}")

        # Check if occupations have expected fields
        if recommendations.occupation_recommendations:
            occ = recommendations.occupation_recommendations[0]
            print(f"\n  Sample occupation:")
            print(f"    - UUID: {occ.uuid}")
            print(f"    - Rank: {occ.rank}")
            print(f"    - Title: {occ.occupation}")
            print(f"    - Eligible: {occ.is_eligible}")
            print(f"    - Final Score: {occ.final_score}")

        return True

    except Exception as e:
        print(f"\n❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("MATCHING SERVICE INTEGRATION TESTS")
    print("="*80)

    results = {}

    # Run tests
    results['connection'] = await test_matching_service_connection()
    results['skills'] = await test_skills_transformation()
    results['preferences'] = await test_preference_transformation()
    results['end_to_end'] = await test_end_to_end_flow()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.ljust(20)}: {status}")

    total_passed = sum(1 for p in results.values() if p)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! Integration is complete.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
