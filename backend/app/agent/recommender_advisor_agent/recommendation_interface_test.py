"""
Unit tests for the BWS data wiring in `recommendation_interface._to_matching_preference_vector`.

These tests guard the wire contract with the matching service: when the recommender
has HB-derived BWS data, both `bws_scores` (per-WA posterior means) and `top_10_bws`
(HB-ranked WA IDs) must arrive on the outgoing `MatchingPreferenceVector`.
"""

from app.agent.preference_elicitation_agent.types import PreferenceVector
from app.agent.recommender_advisor_agent.recommendation_interface import (
    _to_matching_preference_vector,
)


def _agent_pref_vector() -> PreferenceVector:
    pv = PreferenceVector()
    pv.financial_importance = 0.8
    pv.work_life_balance_importance = 0.6
    pv.career_advancement_importance = 0.4
    pv.n_vignettes_completed = 4
    return pv


def test_bws_scores_and_top_10_bws_are_forwarded_when_provided():
    bws_scores = {"4.A.1.a.1": 1.4, "4.A.2.b.3": -0.8, "4.A.3.a.1": 0.1}
    top_10_bws = ["4.A.1.a.1", "4.A.3.a.1", "4.A.2.b.3"]

    out = _to_matching_preference_vector(
        _agent_pref_vector(),
        bws_scores=bws_scores,
        top_10_bws=top_10_bws,
    )

    assert out.bws_scores == bws_scores
    assert out.top_10_bws == top_10_bws


def test_bws_fields_default_to_none_when_not_provided():
    out = _to_matching_preference_vector(_agent_pref_vector())
    assert out.bws_scores is None
    assert out.top_10_bws is None


def test_bws_fields_forwarded_even_when_agent_pref_vector_is_none():
    bws_scores = {"4.A.1.a.1": 1.4}
    top_10_bws = ["4.A.1.a.1"]

    out = _to_matching_preference_vector(
        None,
        bws_scores=bws_scores,
        top_10_bws=top_10_bws,
    )

    assert out.bws_scores == bws_scores
    assert out.top_10_bws == top_10_bws


# --- salary & labor-demand passthrough (CompassMatchingResult -> agent Node2Vec model) ---
from app.agent.recommender_advisor_agent.recommendation_interface import _compass_result_to_node2vec
from app.matching.matching_types import CompassMatchingResult, CompassOccupation, CompassOpportunity


def test_opportunity_salary_and_demand_reach_agent_model():
    """Salary (from salary_text/salary_range) and labor demand must survive the live conversion."""
    result = CompassMatchingResult(
        user_id="y1",
        algorithm_version="v1",
        opportunities=[
            CompassOpportunity(
                uuid="opp1", rank=1, opportunity_title="Borehole Driller",
                salary_text="KES 30,000/month", demand_label="Moderate Expected Demand",
                demand_score=0.5,
            ),
            # salary only in salary_range (salary_text empty) -> must still come through
            CompassOpportunity(
                uuid="opp2", rank=2, opportunity_title="Web Developer",
                salary_range="KES 80,000/month", demand_label="High Expected Demand",
            ),
        ],
    )
    recs = _compass_result_to_node2vec(result)
    o1, o2 = recs.opportunity_recommendations
    assert o1.salary_range == "KES 30,000/month"
    assert o1.demand_label == "Moderate Expected Demand"
    assert o1.labor_demand_category == "medium"
    assert o1.labor_demand_score == 0.5
    assert o2.salary_range == "KES 80,000/month"   # fell back from salary_range
    assert o2.labor_demand_category == "high"


def test_occupation_demand_reaches_agent_model_via_fallback():
    """Occupation labor demand (no score_breakdown on the live path) reaches the model via demand_label."""
    result = CompassMatchingResult(
        user_id="y1",
        algorithm_version="v1",
        occupations=[
            CompassOccupation(uuid="occ1", rank=1, label="Electrician",
                              demand_label="High Expected Demand", demand_score=0.9),
        ],
    )
    occ = _compass_result_to_node2vec(result).occupation_recommendations[0]
    assert occ.demand_label == "High Expected Demand"
    assert occ.labor_demand_category == "high"
    assert occ.labor_demand_score == 0.9


# --- _to_matching_skills_vector: agent skills dict -> matching-service SkillsVector ---
import pytest

from app.agent.recommender_advisor_agent.recommendation_interface import (
    _to_matching_skills_vector,
)


@pytest.mark.parametrize(
    "skills_vector",
    [
        None,                  # missing vector entirely
        {},                    # falsy dict
        {"skills": []},        # present but no skills
        {"skills": None},      # "skills" key present but null
        {"other": "noise"},    # dict without a "skills" key
    ],
    ids=["none", "empty_dict", "empty_skills_list", "skills_none", "no_skills_key"],
)
def test_empty_or_missing_skills_returns_empty_vector(skills_vector):
    """No usable skills -> a valid, empty SkillsVector (never None, never a crash)."""
    out = _to_matching_skills_vector(skills_vector)
    assert out.top_skills == []


def test_skill_with_all_fields_is_converted():
    out = _to_matching_skills_vector({
        "skills": [
            {"preferred_label": "Welding", "origin_uuid": "origin-1", "proficiency": 0.9},
        ]
    })
    assert len(out.top_skills) == 1
    skill = out.top_skills[0]
    assert skill.preferred_label == "Welding"
    assert skill.origin_uuid == "origin-1"
    assert skill.proficiency == 0.9


def test_skill_without_preferred_label_is_dropped():
    out = _to_matching_skills_vector({
        "skills": [
            {"origin_uuid": "origin-1", "proficiency": 0.9},          # no preferred_label -> dropped
            {"preferred_label": "Welding", "origin_uuid": "o2", "proficiency": 0.8},
        ]
    })
    assert [s.preferred_label for s in out.top_skills] == ["Welding"]


def test_origin_uuid_falls_back_to_uuid_when_missing():
    """When origin_uuid is absent, the skill's `uuid` is used as the origin identifier."""
    out = _to_matching_skills_vector({
        "skills": [
            {"preferred_label": "Plumbing", "uuid": "fallback-uuid", "proficiency": 0.7},
        ]
    })
    assert len(out.top_skills) == 1
    assert out.top_skills[0].origin_uuid == "fallback-uuid"


def test_skill_without_any_identifier_is_dropped_not_crashed():
    """Edge case: a labelled skill with neither origin_uuid nor uuid must be dropped,
    not raise a pydantic ValidationError that takes down the whole conversion."""
    out = _to_matching_skills_vector({
        "skills": [
            {"preferred_label": "Ghost Skill", "proficiency": 0.6},   # no origin_uuid, no uuid
            {"preferred_label": "Real Skill", "origin_uuid": "o1", "proficiency": 0.6},
        ]
    })
    assert [s.preferred_label for s in out.top_skills] == ["Real Skill"]


def test_missing_proficiency_defaults_to_half():
    out = _to_matching_skills_vector({
        "skills": [
            {"preferred_label": "Carpentry", "origin_uuid": "o1"},           # no proficiency key
        ]
    })
    assert out.top_skills[0].proficiency == 0.5


def test_falsy_proficiency_defaults_to_half():
    """0 / 0.0 proficiency is treated as 'not provided' and defaults to 0.5 (current contract)."""
    out = _to_matching_skills_vector({
        "skills": [
            {"preferred_label": "Carpentry", "origin_uuid": "o1", "proficiency": 0},
        ]
    })
    assert out.top_skills[0].proficiency == 0.5


def test_mixed_batch_converts_keepers_and_drops_invalid():
    out = _to_matching_skills_vector({
        "skills": [
            {"preferred_label": "Keep A", "origin_uuid": "a", "proficiency": 0.9},
            {"origin_uuid": "b", "proficiency": 0.9},                 # no label -> drop
            {"preferred_label": "Keep B", "uuid": "c"},               # uuid fallback + default prof
            {"preferred_label": "Drop C", "proficiency": 0.5},        # no id at all -> drop
        ]
    })
    assert [(s.preferred_label, s.origin_uuid, s.proficiency) for s in out.top_skills] == [
        ("Keep A", "a", 0.9),
        ("Keep B", "c", 0.5),
    ]
