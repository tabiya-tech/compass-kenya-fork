import pytest
from unittest.mock import MagicMock
from app.agent.collect_experiences_agent._types import CollectedData
from app.agent.collect_experiences_agent.collect_experiences_agent import CollectExperiencesAgent, CollectExperiencesAgentState


def _make_state(collected_data: list[CollectedData]) -> CollectExperiencesAgentState:
    state = MagicMock(spec=CollectExperiencesAgentState)
    state.collected_data = collected_data
    return state


def _make_agent_with_state(collected_data: list[CollectedData]) -> CollectExperiencesAgent:
    agent = CollectExperiencesAgent.__new__(CollectExperiencesAgent)
    agent._state = _make_state(collected_data)
    agent._logger = MagicMock()
    return agent


def test_get_experiences_pre_populates_cv_responsibilities():
    data = CollectedData(
        index=0,
        experience_title="Nurse",
        company="Hospital",
        location="Nairobi",
        start_date="2020-01",
        end_date="2023-06",
        paid_work=True,
        work_type="FORMAL_SECTOR_WAGED_EMPLOYMENT",
        source="cv",
        responsibilities=["Administered medication", "Recorded patient vitals"],
    )
    agent = _make_agent_with_state([data])
    experiences = agent.get_experiences()
    assert len(experiences) == 1
    assert experiences[0].responsibilities.responsibilities == ["Administered medication", "Recorded patient vitals"]


def test_get_experiences_sets_source_from_cv():
    data = CollectedData(
        index=0,
        experience_title="Teacher",
        company="School",
        location="Mombasa",
        start_date="2018-01",
        end_date="2020-12",
        paid_work=True,
        work_type="FORMAL_SECTOR_WAGED_EMPLOYMENT",
        source="cv",
        responsibilities=["Prepared lesson plans"],
    )
    agent = _make_agent_with_state([data])
    experiences = agent.get_experiences()
    assert experiences[0].source == "cv"


def test_get_experiences_source_none_for_conversational_data():
    data = CollectedData(
        index=0,
        experience_title="Driver",
        company=None,
        location=None,
        start_date=None,
        end_date=None,
        paid_work=None,
        work_type=None,
        source=None,
        responsibilities=[],
    )
    agent = _make_agent_with_state([data])
    experiences = agent.get_experiences()
    assert experiences[0].source is None
    assert experiences[0].responsibilities.responsibilities == []


def test_get_experiences_empty_responsibilities_produces_empty_responsibilities_data():
    data = CollectedData(
        index=0,
        experience_title="Intern",
        company="Corp",
        location=None,
        start_date=None,
        end_date=None,
        paid_work=None,
        work_type=None,
        source="cv",
        responsibilities=[],
    )
    agent = _make_agent_with_state([data])
    experiences = agent.get_experiences()
    assert experiences[0].responsibilities.responsibilities == []
