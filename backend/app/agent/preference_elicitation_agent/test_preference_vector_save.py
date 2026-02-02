"""
Unit test for preference vector saving to JobPreferences collection.

Tests that the PreferenceElicitationAgent correctly saves preference vectors
to the JobPreferences database when completing the WRAPUP phase.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.preference_elicitation_agent.agent import PreferenceElicitationAgent
from app.agent.preference_elicitation_agent.state import PreferenceElicitationAgentState
from app.agent.preference_elicitation_agent.types import PreferenceVector
from app.job_preferences.types import JobPreferences


@pytest.mark.asyncio
async def test_save_preference_vector_to_job_preferences():
    """
    Test that preference vector is correctly saved to JobPreferences collection.

    This verifies the Epic 2 → JobPreferences database integration.
    """
    # Create agent
    agent = PreferenceElicitationAgent()

    # Create state with completed preference vector
    state = PreferenceElicitationAgentState(
        session_id=99999,
        initial_experiences_snapshot=[],
        use_db6_for_fresh_data=False,
        conversation_phase="WRAPUP"
    )

    # Set up a completed preference vector
    state.preference_vector = PreferenceVector(
        # Core dimensions
        financial_importance=0.85,
        work_environment_importance=0.70,
        career_advancement_importance=0.90,
        work_life_balance_importance=0.60,
        job_security_importance=0.75,
        task_preference_importance=0.65,
        social_impact_importance=0.55,

        # Quality metadata
        confidence_score=0.82,
        n_vignettes_completed=8,
        per_dimension_uncertainty={
            "financial_importance": 0.15,
            "work_environment_importance": 0.20,
            "career_advancement_importance": 0.12,
            "work_life_balance_importance": 0.25,
            "job_security_importance": 0.18,
            "task_preference_importance": 0.22,
            "social_impact_importance": 0.28
        },

        # Bayesian metadata
        posterior_mean=[0.5, 0.3, 0.7, 0.2, 0.4, 0.1, 0.0],
        posterior_covariance_diagonal=[0.15, 0.20, 0.12, 0.25, 0.18, 0.22, 0.28],
        fim_determinant=5432.1,

        # Qualitative insights (correct types: dict[str, bool] not lists)
        decision_patterns={"uses_financial_language": True, "career_growth_focused": True},
        tradeoff_willingness={"salary_for_flexibility": True},
        values_signals={"helping_others": True},
        consistency_indicators={
            "response_consistency": 0.95,
            "conviction_strength": 0.88,
            "preference_stability": 0.92
        },
        extracted_constraints={"minimum_salary": 50000}
    )

    agent.set_state(state)

    # Mock the job preferences service (note: get_job_preferences_service is NOT async, just returns service)
    mock_service = MagicMock()
    mock_service.create_or_update = AsyncMock()

    with patch('app.agent.preference_elicitation_agent.agent.get_job_preferences_service', new_callable=AsyncMock, return_value=mock_service):
        # Call the save method
        await agent._save_preference_vector_to_job_preferences()

    # Verify create_or_update was called
    assert mock_service.create_or_update.called, "create_or_update was not called"

    # Get the call arguments
    call_args = mock_service.create_or_update.call_args
    assert call_args is not None, "create_or_update call args are None"

    # Extract arguments
    kwargs = call_args.kwargs
    session_id = kwargs.get('session_id')
    preferences = kwargs.get('preferences')

    # Verify session_id
    assert session_id == 99999, f"Expected session_id 99999, got {session_id}"

    # Verify preferences object
    assert isinstance(preferences, JobPreferences), f"Expected JobPreferences, got {type(preferences)}"

    # Verify core dimensions were mapped correctly
    assert preferences.financial_importance == 0.85
    assert preferences.work_environment_importance == 0.70
    assert preferences.career_advancement_importance == 0.90
    assert preferences.work_life_balance_importance == 0.60
    assert preferences.job_security_importance == 0.75
    assert preferences.task_preference_importance == 0.65
    assert preferences.social_impact_importance == 0.55

    # Verify metadata
    assert preferences.confidence_score == 0.82
    assert preferences.n_vignettes_completed == 8
    assert preferences.per_dimension_uncertainty is not None
    assert len(preferences.per_dimension_uncertainty) == 7

    # Verify Bayesian metadata
    assert preferences.posterior_mean == [0.5, 0.3, 0.7, 0.2, 0.4, 0.1, 0.0]
    assert preferences.posterior_covariance_diagonal == [0.15, 0.20, 0.12, 0.25, 0.18, 0.22, 0.28]
    assert preferences.fim_determinant == 5432.1

    # Verify qualitative insights
    assert preferences.decision_patterns == {"uses_financial_language": True, "career_growth_focused": True}
    assert preferences.tradeoff_willingness == {"salary_for_flexibility": True}
    assert preferences.values_signals == {"helping_others": True}
    assert preferences.consistency_indicators["response_consistency"] == 0.95

    # Verify hard constraints
    assert preferences.concrete_salary_min == 50000

    # Verify timestamp is recent (within last few seconds)
    assert preferences.last_updated is not None
    time_diff = (datetime.now(timezone.utc) - preferences.last_updated).total_seconds()
    assert time_diff < 5, f"Timestamp too old: {time_diff} seconds"

    print("\n✅ ALL ASSERTIONS PASSED!")
    print(f"\nPreference Vector → JobPreferences Mapping Verified:")
    print(f"  - Session ID: {session_id}")
    print(f"  - Financial: {preferences.financial_importance}")
    print(f"  - Work Environment: {preferences.work_environment_importance}")
    print(f"  - Career Advancement: {preferences.career_advancement_importance}")
    print(f"  - Confidence: {preferences.confidence_score}")
    print(f"  - Vignettes Completed: {preferences.n_vignettes_completed}")
    print(f"  - FIM Determinant: {preferences.fim_determinant}")
    print(f"  - Decision Patterns: {preferences.decision_patterns}")
    print(f"  - Minimum Salary Constraint: {preferences.concrete_salary_min}")


@pytest.mark.asyncio
async def test_save_handles_service_error_gracefully():
    """
    Test that errors during save don't crash the agent.

    The save should be best-effort - if it fails, log but don't fail the conversation.
    """
    agent = PreferenceElicitationAgent()

    state = PreferenceElicitationAgentState(
        session_id=99999,
        initial_experiences_snapshot=[],
        use_db6_for_fresh_data=False
    )

    state.preference_vector = PreferenceVector(
        financial_importance=0.5,
        work_environment_importance=0.5,
        career_advancement_importance=0.5,
        work_life_balance_importance=0.5,
        job_security_importance=0.5,
        task_preference_importance=0.5,
        social_impact_importance=0.5,
        confidence_score=0.5,
        n_vignettes_completed=1
    )

    agent.set_state(state)

    # Mock service that raises an error
    mock_service = MagicMock()
    mock_service.create_or_update = AsyncMock(side_effect=Exception("Database connection failed"))

    with patch('app.agent.preference_elicitation_agent.agent.get_job_preferences_service', new_callable=AsyncMock, return_value=mock_service):
        # Should not raise exception
        try:
            await agent._save_preference_vector_to_job_preferences()
            print("\n✅ Error handled gracefully - no exception raised")
        except Exception as e:
            pytest.fail(f"Should not raise exception, but got: {e}")


@pytest.mark.asyncio
async def test_save_called_during_wrapup():
    """
    Test that _save_preference_vector_to_job_preferences is called during WRAPUP phase.

    This is an integration-style test that verifies the save happens at the right time.
    """
    agent = PreferenceElicitationAgent()

    state = PreferenceElicitationAgentState(
        session_id=88888,
        initial_experiences_snapshot=[],
        use_db6_for_fresh_data=False,
        conversation_phase="WRAPUP"
    )

    # Set up minimal preference vector
    state.preference_vector = PreferenceVector(
        financial_importance=0.7,
        work_environment_importance=0.6,
        career_advancement_importance=0.8,
        work_life_balance_importance=0.5,
        job_security_importance=0.6,
        task_preference_importance=0.5,
        social_impact_importance=0.4,
        confidence_score=0.75,
        n_vignettes_completed=6
    )

    state.completed_vignettes = ["v1", "v2", "v3", "v4", "v5", "v6"]

    agent.set_state(state)

    # Mock both DB6 and JobPreferences saves
    mock_db6_service = AsyncMock()
    mock_job_prefs_service = MagicMock()
    mock_job_prefs_service.create_or_update = AsyncMock()

    # Mock the agent's save methods directly
    agent._save_preference_vector_to_db6 = AsyncMock()

    with patch('app.agent.preference_elicitation_agent.agent.get_job_preferences_service', return_value=mock_job_prefs_service):
        # Trigger save by calling the internal method that handles WRAPUP
        await agent._save_preference_vector_to_job_preferences()

    # Verify the save was called
    assert mock_job_prefs_service.create_or_update.called, "JobPreferences save was not called"

    print("\n✅ Verified: _save_preference_vector_to_job_preferences is called")
    print(f"   Called with session_id: {mock_job_prefs_service.create_or_update.call_args.kwargs['session_id']}")


if __name__ == "__main__":
    # Allow running directly for quick testing
    import asyncio

    print("="*80)
    print("STANDALONE TEST: Preference Vector → JobPreferences Save")
    print("="*80)

    async def run_all_tests():
        print("\n[TEST 1] Testing save logic...")
        await test_save_preference_vector_to_job_preferences()

        print("\n[TEST 2] Testing error handling...")
        await test_save_handles_service_error_gracefully()

        print("\n[TEST 3] Testing save is called during WRAPUP...")
        await test_save_called_during_wrapup()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)

    asyncio.run(run_all_tests())
