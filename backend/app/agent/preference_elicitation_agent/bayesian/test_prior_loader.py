"""
Tests for PriorsLoader and schema_loader term-level interface.
"""
import math
import json
import pytest
import tempfile
import os

from app.agent.preference_elicitation_agent.bayesian.schema_loader import SchemaLoader
from app.agent.preference_elicitation_agent.bayesian.prior_loader import PriorsLoader


# ------------------------------------------------------------------
# Minimal schema fixture matching preference_parameters.json structure
# ------------------------------------------------------------------

MINIMAL_SCHEMA = {
    "attributes": [
        {
            "name": "earnings_per_month",
            "label": "Monthly Earnings",
            "group": "Financial",
            "type": "ordered",
            "coding": "linear",
            "levels": [
                {"id": "earn_15k", "label": "KES 15,000", "value": 15000},
                {"id": "earn_30k", "label": "KES 30,000", "value": 30000},
                {"id": "earn_50k", "label": "KES 50,000", "value": 50000},
                {"id": "earn_70k", "label": "KES 70,000", "value": 70000},
            ],
        },
        {
            "name": "physical_demand",
            "label": "Physical Demand",
            "group": "Work Environment",
            "type": "categorical",
            "coding": "dummy",
            "base_level_id": "phys_light",
            "direction": "negative",
            "levels": [
                {"id": "phys_light", "label": "Light"},
                {"id": "phys_heavy", "label": "Heavy"},
            ],
        },
        {
            "name": "social_interaction",
            "label": "Social Interaction",
            "group": "Work Environment",
            "type": "categorical",
            "coding": "dummy",
            "base_level_id": "soc_alone",
            "levels": [
                {"id": "soc_alone", "label": "Alone"},
                {"id": "soc_peers", "label": "Peers"},
                {"id": "soc_customers", "label": "Customers"},
            ],
        },
        {
            "name": "career_growth",
            "label": "Career Growth",
            "group": "Future Prospects",
            "type": "ordered",
            "coding": "dummy",
            "base_level_id": "growth_low",
            "levels": [
                {"id": "growth_low", "label": "Few"},
                {"id": "growth_med", "label": "Some"},
                {"id": "growth_high", "label": "Strong"},
            ],
        },
    ]
}

# Priors JSON matching dce_dynamic_priors_batch1_short.json structure
SAMPLE_PRIORS = {
    "earnings_k": {
        "type": "continuous",
        "term": "earnings_k",
        "prior_mean": 0.0062,
        "prior_sd": 0.013,
    },
    "physical_demand_risk": {
        "type": "categorical_treatment",
        "reference_level": "risky",
        "reference_prior": {"level": "risky", "prior_mean": 0, "prior_sd": 0, "fixed": True},
        "levels": [
            {
                "level": "safe",
                "term": "physical_demand_risksafe",
                "prior_mean": 0.7817,
                "prior_sd": 0.05,
            }
        ],
    },
    "social_interaction": {
        "type": "categorical_treatment",
        "reference_level": "alone",
        "reference_prior": {"level": "alone", "prior_mean": 0, "prior_sd": 0, "fixed": True},
        "levels": [
            {
                "level": "peers",
                "term": "social_interactionpeers",
                "prior_mean": 0.3553,
                "prior_sd": 0.0539,
            },
            {
                "level": "clients",
                "term": "social_interactionclients",
                "prior_mean": 0.0769,
                "prior_sd": 0.0619,
            },
        ],
    },
    "career_growth": {
        "type": "categorical_treatment",
        "reference_level": "few",
        "reference_prior": {"level": "few", "prior_mean": 0, "prior_sd": 0, "fixed": True},
        "levels": [
            {
                "level": "some",
                "term": "career_growthsome",
                "prior_mean": 0.4596,
                "prior_sd": 0.0581,
            },
            {
                "level": "strong",
                "term": "career_growthstrong",
                "prior_mean": 0.739,
                "prior_sd": 0.0662,
            },
        ],
    },
}


@pytest.fixture
def schema_loader():
    return SchemaLoader.from_dict(MINIMAL_SCHEMA)


@pytest.fixture
def priors_file(tmp_path):
    path = tmp_path / "test_priors.json"
    path.write_text(json.dumps(SAMPLE_PRIORS))
    return str(path)


# ------------------------------------------------------------------
# SchemaLoader term-level tests
# ------------------------------------------------------------------

class TestSchemaLoaderTermLevel:

    def test_term_names_count(self, schema_loader):
        # 1 linear + 1 dummy(1 non-base) + 1 dummy(2 non-base) + 1 dummy(2 non-base) = 6
        assert schema_loader.n_terms == 6

    def test_term_names_order(self, schema_loader):
        names = schema_loader.term_names
        assert names[0] == "earnings_per_month"          # linear
        assert names[1] == "physical_demand_phys_heavy"  # dummy non-base
        assert names[2] == "social_interaction_soc_peers"
        assert names[3] == "social_interaction_soc_customers"
        assert names[4] == "career_growth_growth_med"
        assert names[5] == "career_growth_growth_high"

    def test_default_term_prior_mean_length(self, schema_loader):
        assert len(schema_loader.default_term_prior_mean) == 6
        assert all(v == 0.5 for v in schema_loader.default_term_prior_mean)

    def test_extract_term_features_earnings_linear(self, schema_loader):
        # earn_70k is max (1.0), earn_15k is min (0.0)
        feats = schema_loader.extract_term_features({"earnings_per_month": "earn_70k"})
        assert feats[0] == pytest.approx(1.0)

        feats = schema_loader.extract_term_features({"earnings_per_month": "earn_15k"})
        assert feats[0] == pytest.approx(0.0)

    def test_extract_term_features_dummy_present(self, schema_loader):
        # phys_heavy is the non-base level; direction=negative flips it → 0.0 when heavy
        feats_heavy = schema_loader.extract_term_features({"physical_demand": "phys_heavy"})
        feats_light = schema_loader.extract_term_features({"physical_demand": "phys_light"})
        assert feats_heavy[1] == pytest.approx(0.0)  # risky (heavy) = 0 after flip
        assert feats_light[1] == pytest.approx(1.0)  # safe (light) = 1 after flip

    def test_extract_term_features_social_peers(self, schema_loader):
        feats = schema_loader.extract_term_features({"social_interaction": "soc_peers"})
        assert feats[2] == pytest.approx(1.0)  # peers term
        assert feats[3] == pytest.approx(0.0)  # customers term

    def test_extract_term_features_length(self, schema_loader):
        feats = schema_loader.extract_term_features({
            "earnings_per_month": "earn_30k",
            "physical_demand": "phys_light",
            "social_interaction": "soc_alone",
            "career_growth": "growth_high",
        })
        assert len(feats) == 6

    def test_extract_term_features_missing_attribute_returns_correct_length(self, schema_loader):
        # Missing attributes: result length must still match n_terms
        feats = schema_loader.extract_term_features({})
        assert len(feats) == 6

    def test_extract_term_features_linear_missing_is_zero(self, schema_loader):
        # Missing linear attribute → 0.0 for that term
        feats = schema_loader.extract_term_features({})
        assert feats[0] == pytest.approx(0.0)  # earnings


# ------------------------------------------------------------------
# PriorsLoader tests
# ------------------------------------------------------------------

class TestPriorsLoader:

    def test_load_returns_correct_term_count(self, priors_file, schema_loader):
        result = PriorsLoader.load(priors_file, schema_loader)
        assert len(result.mean) == schema_loader.n_terms
        assert len(result.variances) == schema_loader.n_terms
        assert len(result.term_names) == schema_loader.n_terms

    def test_earnings_prior_scaled(self, priors_file, schema_loader):
        result = PriorsLoader.load(priors_file, schema_loader)
        # earnings_k = 0.0062, range = 55k → scaled mean ≈ 0.341
        assert result.mean[0] == pytest.approx(0.0062 * 55.0, abs=0.01)

    def test_categorical_priors_unchanged(self, priors_file, schema_loader):
        result = PriorsLoader.load(priors_file, schema_loader)
        # physical_demand safe ≈ 0.7817 (no scaling for dummy terms)
        # term index 1 = physical_demand_phys_heavy
        assert result.mean[1] == pytest.approx(0.7817, abs=0.01)

    def test_social_interaction_priors(self, priors_file, schema_loader):
        result = PriorsLoader.load(priors_file, schema_loader)
        assert result.mean[2] == pytest.approx(0.3553, abs=0.01)  # peers
        assert result.mean[3] == pytest.approx(0.0769, abs=0.01)  # clients

    def test_career_growth_priors(self, priors_file, schema_loader):
        result = PriorsLoader.load(priors_file, schema_loader)
        assert result.mean[4] == pytest.approx(0.4596, abs=0.01)  # some
        assert result.mean[5] == pytest.approx(0.7390, abs=0.01)  # strong

    def test_variances_are_sd_squared(self, priors_file, schema_loader):
        result = PriorsLoader.load(priors_file, schema_loader)
        # physical_demand sd=0.05 → variance=0.0025
        assert result.variances[1] == pytest.approx(0.05 ** 2, abs=1e-6)

    def test_fim_determinant_is_product_of_precisions(self, priors_file, schema_loader):
        result = PriorsLoader.load(priors_file, schema_loader)
        expected = math.prod(1.0 / v for v in result.variances)
        assert result.fim_determinant == pytest.approx(expected, rel=1e-6)

    def test_fim_determinant_positive(self, priors_file, schema_loader):
        result = PriorsLoader.load(priors_file, schema_loader)
        assert result.fim_determinant > 0
