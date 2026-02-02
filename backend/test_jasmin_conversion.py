"""
Test script to verify Jasmin's Node2Vec output converts correctly to agent format.

Run with: python backend/test_jasmin_conversion.py
"""

import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.agent.recommender_advisor_agent.types import Node2VecRecommendations


def test_jasmin_conversion():
    """Test conversion of Jasmin's actual Node2Vec output."""

    # Load Jasmin's output
    jasmin_file = Path(__file__).parent.parent.parent / "project-workspace/shared documents/node2vec-output.json"

    if not jasmin_file.exists():
        print(f"ERROR: Jasmin's output file not found at {jasmin_file}")
        return False

    with open(jasmin_file, 'r') as f:
        jasmin_data = json.load(f)

    print(f"✓ Loaded Jasmin's output: {len(jasmin_data)} user(s)")

    # Convert to agent format
    try:
        recommendations = Node2VecRecommendations.from_jasmin_output(jasmin_data)
        print(f"✓ Converted to agent format successfully")
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify structure
    print(f"\n=== Converted Recommendations ===")
    print(f"Youth ID: {recommendations.youth_id}")
    print(f"Occupation recommendations: {len(recommendations.occupation_recommendations)}")
    print(f"Opportunity recommendations: {len(recommendations.opportunity_recommendations)}")
    print(f"Training recommendations: {len(recommendations.skillstraining_recommendations)}")
    print(f"Raw skill gaps: {len(recommendations.skill_gap_recommendations)}")

    # Test first opportunity
    if recommendations.opportunity_recommendations:
        opp = recommendations.opportunity_recommendations[0]
        print(f"\n=== First Opportunity ===")
        print(f"UUID: {opp.uuid}")
        print(f"Title: {opp.opportunity_title}")
        print(f"Location: {opp.location}")
        print(f"Eligible: {opp.is_eligible}")
        print(f"Final score: {opp.final_score:.4f}")
        print(f"Score breakdown:")
        print(f"  - Skill utility: {opp.score_breakdown.total_skill_utility:.4f}")
        print(f"  - Preference score: {opp.score_breakdown.preference_score:.4f}")
        print(f"  - Demand score: {opp.score_breakdown.demand_score:.4f}")
        print(f"  - Demand label: {opp.score_breakdown.demand_label}")
        print(f"Matched skills: {len(opp.matched_skills.essential_skill_matches)} essential")
        print(f"Matched preferences: {len(opp.matched_preferences)} attributes")

        # Test backward-compatible property access
        print(f"\n=== Backward Compatibility (via @property) ===")
        print(f"Essential skills: {opp.essential_skills[:3]}...")  # First 3
        print(f"Property access works: ✓")

    # Test training conversion
    if recommendations.skillstraining_recommendations:
        training = recommendations.skillstraining_recommendations[0]
        print(f"\n=== First Training (converted from skill gap) ===")
        print(f"UUID: {training.uuid}")
        print(f"Skill: {training.skill}")
        print(f"Training title: {training.training_title}")
        print(f"Provider: {training.provider}")
        print(f"Cost: {training.cost}")

    print(f"\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_jasmin_conversion()
    sys.exit(0 if success else 1)
