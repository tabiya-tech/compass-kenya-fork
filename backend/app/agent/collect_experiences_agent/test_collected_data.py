import pytest
from app.agent.collect_experiences_agent._types import CollectedData


def test_responsibilities_defaults_to_empty_list():
    data = CollectedData(
        index=0,
        experience_title="Software Engineer",
        company=None,
        location=None,
        start_date=None,
        end_date=None,
        paid_work=None,
        work_type=None,
    )
    assert data.responsibilities == []


def test_responsibilities_accepts_list_of_strings():
    data = CollectedData(
        index=0,
        experience_title="Software Engineer",
        company="Acme",
        location=None,
        start_date=None,
        end_date=None,
        paid_work=None,
        work_type=None,
        responsibilities=["Built REST APIs", "Led code reviews"],
    )
    assert data.responsibilities == ["Built REST APIs", "Led code reviews"]


def test_existing_document_without_field_deserializes_cleanly():
    """Simulate a stored MongoDB document that predates the responsibilities field."""
    raw = {
        "uuid": "abc-123",
        "index": 0,
        "experience_title": "Teacher",
        "company": "School",
        "location": "Nairobi",
        "start_date": "2019-01",
        "end_date": "2021-12",
        "paid_work": True,
        "work_type": "FORMAL_SECTOR_WAGED_EMPLOYMENT",
        "source": "cv",
    }
    data = CollectedData(**raw)
    assert data.responsibilities == []


def test_responsibilities_serializes_to_json():
    data = CollectedData(
        index=0,
        experience_title="Nurse",
        company=None,
        location=None,
        start_date=None,
        end_date=None,
        paid_work=None,
        work_type=None,
        responsibilities=["Administered medication", "Recorded patient vitals"],
    )
    dumped = data.model_dump_json()
    assert "Administered medication" in dumped
    assert "Recorded patient vitals" in dumped


def test_state_without_education_phase_done_deserializes_cleanly():
    """Simulate a stored MongoDB document that predates the education_phase_done field."""
    from app.agent.collect_experiences_agent.collect_experiences_agent import CollectExperiencesAgentState
    raw = {
        "session_id": 1,
        "country_of_user": "UNSPECIFIED",
        "persona_type": "INFORMAL",
        "collected_data": [],
        "unexplored_types": ["FORMAL_SECTOR_WAGED_EMPLOYMENT"],
        "explored_types": [],
        "first_time_visit": True,
    }
    state = CollectExperiencesAgentState.from_document(raw)
    assert state.education_phase_done is False
