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
