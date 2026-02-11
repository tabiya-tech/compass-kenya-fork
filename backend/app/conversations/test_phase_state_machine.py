import pytest

from app.conversations.phase_state_machine import (
    JourneyPhase,
    PhaseDataStatus,
    determine_start_phase,
)


def test_default_start_phase_when_no_data():
    status = PhaseDataStatus()

    phase = determine_start_phase(status)

    assert phase is JourneyPhase.SKILLS_ELICITATION


def test_start_at_preference_when_only_skills_exist():
    status = PhaseDataStatus(has_skills_elicitation=True)

    phase = determine_start_phase(status)

    assert phase is JourneyPhase.PREFERENCE_ELICITATION


def test_start_at_matching_when_skills_and_preferences_exist():
    status = PhaseDataStatus(
        has_skills_elicitation=True,
        has_preference_elicitation=True,
    )

    phase = determine_start_phase(status)

    assert phase is JourneyPhase.MATCHING


def test_start_at_matching_when_matching_results_already_exist():
    status = PhaseDataStatus(has_matching=True)

    phase = determine_start_phase(status)

    assert phase is JourneyPhase.MATCHING


def test_start_at_recommendation_when_recommendations_exist():
    status = PhaseDataStatus(has_recommendation=True)

    phase = determine_start_phase(status)

    assert phase is JourneyPhase.RECOMMENDATION


def test_respects_allowed_phases_subset():
    status = PhaseDataStatus(
        has_skills_elicitation=True,
        has_preference_elicitation=True,
        has_matching=True,
        has_recommendation=True,
    )

    phase = determine_start_phase(
        status,
        allowed_phases=[JourneyPhase.SKILLS_ELICITATION, JourneyPhase.PREFERENCE_ELICITATION],
    )

    assert phase is JourneyPhase.PREFERENCE_ELICITATION


def test_raises_when_allowed_phases_empty_after_filter():
    status = PhaseDataStatus()

    with pytest.raises(ValueError):
        determine_start_phase(status, allowed_phases=[])

