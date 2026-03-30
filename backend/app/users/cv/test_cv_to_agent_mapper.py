import pytest
from app.users.cv.cv_to_agent_mapper import map_cv_to_collected_data
from app.users.cv.types import CVExtractedExperience


def _make_cv_exp(**kwargs) -> CVExtractedExperience:
    defaults = dict(
        experience_title="Software Engineer",
        company="Acme",
        location="Nairobi",
        start_date="2020-01",
        end_date="2022-12",
        work_type="FORMAL_SECTOR_WAGED_EMPLOYMENT",
        responsibilities=[],
    )
    defaults.update(kwargs)
    return CVExtractedExperience(**defaults)


def test_mapper_passes_responsibilities():
    cv_exp = _make_cv_exp(responsibilities=["Built REST APIs", "Led code reviews"])
    result = map_cv_to_collected_data([cv_exp], [])
    assert len(result) == 1
    assert result[0].responsibilities == ["Built REST APIs", "Led code reviews"]


def test_mapper_handles_empty_responsibilities():
    cv_exp = _make_cv_exp(responsibilities=[])
    result = map_cv_to_collected_data([cv_exp], [])
    assert len(result) == 1
    assert result[0].responsibilities == []


def test_mapper_preserves_source_as_cv():
    cv_exp = _make_cv_exp()
    result = map_cv_to_collected_data([cv_exp], [])
    assert result[0].source == "cv"


def test_mapper_deduplicates_and_responsibilities_not_used_for_comparison():
    """Responsibilities do not affect deduplication — same title/dates/company deduplicate."""
    cv_exp1 = _make_cv_exp(responsibilities=["Task A"])
    cv_exp2 = _make_cv_exp(responsibilities=["Task B"])  # same structural fields
    result = map_cv_to_collected_data([cv_exp1, cv_exp2], [])
    # cv_exp2 is a duplicate of cv_exp1 — only one should appear
    assert len(result) == 1
