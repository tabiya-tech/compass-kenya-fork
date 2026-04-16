"""
Unit tests for LikelihoodCalculator and SchemaLoader.

All tests are schema-driven: a minimal in-memory schema is defined here so
the tests are completely independent of which JSON file is deployed.
The same SchemaLoader + LikelihoodCalculator logic must work for any
conforming schema (original 10-attribute, reduced 4-attribute, or future ones).
"""

import pytest
import numpy as np
from app.agent.preference_elicitation_agent.bayesian.schema_loader import SchemaLoader
from app.agent.preference_elicitation_agent.bayesian.likelihood_calculator import LikelihoodCalculator
from app.agent.preference_elicitation_agent.types import Vignette, VignetteOption


# ---------------------------------------------------------------------------
# Minimal test schema — 3 attributes across 2 dimensions.
# Deliberately different from both old and new production schemas.
# ---------------------------------------------------------------------------
MINIMAL_SCHEMA = {
    "attributes": [
        {
            "name": "earnings",
            "label": "Monthly earnings",
            "group": "Financial",
            "type": "ordered",
            "coding": "linear",
            "levels": [
                {"id": "low",  "label": "Low",  "value": 10000},
                {"id": "mid",  "label": "Mid",  "value": 30000},
                {"id": "high", "label": "High", "value": 50000},
            ],
        },
        {
            "name": "physical_demand",
            "label": "Physical demand",
            "group": "Work Environment",
            "type": "categorical",
            "coding": "dummy",
            "base_level_id": "light",
            "direction": "negative",
            "levels": [
                {"id": "light", "label": "Light"},
                {"id": "heavy", "label": "Heavy"},
            ],
        },
        {
            "name": "social_interaction",
            "label": "Social interaction",
            "group": "Work Environment",
            "type": "categorical",
            "coding": "dummy",
            "base_level_id": "alone",
            "levels": [
                {"id": "alone",     "label": "Work alone"},
                {"id": "team",      "label": "Work in team"},
            ],
        },
    ]
}
# Derived: 2 dimensions — financial_importance, work_environment_importance


@pytest.fixture(scope="module")
def schema():
    return SchemaLoader.from_dict(MINIMAL_SCHEMA)


@pytest.fixture(scope="module")
def calculator(schema):
    return LikelihoodCalculator(schema_loader=schema, temperature=1.0)


def _make_vignette(attrs_a: dict, attrs_b: dict) -> Vignette:
    return Vignette(
        vignette_id="test",
        category="test",
        scenario_text="Choose between these two jobs:",
        options=[
            VignetteOption(option_id="A", title="Job A", description="A", attributes=attrs_a),
            VignetteOption(option_id="B", title="Job B", description="B", attributes=attrs_b),
        ],
    )


# ---------------------------------------------------------------------------
# SchemaLoader tests
# ---------------------------------------------------------------------------

class TestSchemaLoader:

    def test_dimensions_derived_from_groups(self, schema):
        assert schema.dimensions == ["financial_importance", "work_environment_importance"]

    def test_n_dimensions(self, schema):
        assert schema.n_dimensions == 2

    def test_default_prior_mean_length(self, schema):
        assert len(schema.default_prior_mean) == schema.n_dimensions
        assert all(v == 0.5 for v in schema.default_prior_mean)

    def test_attribute_names(self, schema):
        assert schema.attribute_names == ["earnings", "physical_demand", "social_interaction"]

    def test_extract_features_length(self, schema):
        features = schema.extract_features({"earnings": "mid"})
        assert len(features) == schema.n_dimensions

    def test_linear_attribute_low_encodes_to_zero(self, schema):
        features = schema.extract_features({"earnings": "low"})
        assert np.isclose(features[0], 0.0)

    def test_linear_attribute_high_encodes_to_one(self, schema):
        features = schema.extract_features({"earnings": "high"})
        assert np.isclose(features[0], 1.0)

    def test_linear_attribute_mid_encodes_to_half(self, schema):
        features = schema.extract_features({"earnings": "mid"})
        assert np.isclose(features[0], 0.5)

    def test_negative_direction_flips_encoding(self, schema):
        # physical_demand direction=negative: heavy (non-base) should map LOWER than light (base)
        features_light = schema.extract_features({"physical_demand": "light"})
        features_heavy = schema.extract_features({"physical_demand": "heavy"})
        assert features_light[1] > features_heavy[1]

    def test_dummy_base_level_encodes_to_zero(self, schema):
        # social_interaction base = "alone" (no negative direction)
        features = schema.extract_features({"social_interaction": "alone"})
        assert np.isclose(features[1], 0.0)  # alone is base, maps to 0

    def test_dummy_non_base_encodes_to_one(self, schema):
        features = schema.extract_features({"social_interaction": "team"})
        assert np.isclose(features[1], 1.0)

    def test_multiple_attributes_same_dimension_averaged(self, schema):
        # Both physical_demand and social_interaction map to work_environment_importance
        # light → 1.0 (negative flipped), alone → 0.0 → avg = 0.5
        features = schema.extract_features({"physical_demand": "light", "social_interaction": "alone"})
        assert np.isclose(features[1], 0.5)

    def test_missing_attribute_gives_zero(self, schema):
        features = schema.extract_features({})
        assert all(v == 0.0 for v in features)

    def test_unknown_level_id_gives_zero(self, schema):
        features = schema.extract_features({"earnings": "nonexistent_level"})
        assert np.isclose(features[0], 0.0)

    def test_describe_returns_expected_keys(self, schema):
        desc = schema.describe()
        assert "n_dimensions" in desc
        assert "dimensions" in desc
        assert "attributes_per_dimension" in desc

    def test_any_conforming_schema_loads(self):
        """SchemaLoader must work with an arbitrary conforming schema."""
        custom = {
            "attributes": [
                {
                    "name": "salary",
                    "label": "Salary",
                    "group": "Pay",
                    "type": "ordered",
                    "coding": "linear",
                    "levels": [{"id": "a", "value": 1}, {"id": "b", "value": 2}],
                },
                {
                    "name": "commute",
                    "label": "Commute",
                    "group": "Lifestyle",
                    "type": "categorical",
                    "coding": "dummy",
                    "base_level_id": "short",
                    "levels": [{"id": "short"}, {"id": "long"}],
                },
            ]
        }
        loader = SchemaLoader.from_dict(custom)
        assert loader.n_dimensions == 2
        assert loader.dimensions == ["pay_importance", "lifestyle_importance"]


# ---------------------------------------------------------------------------
# LikelihoodCalculator tests
# ---------------------------------------------------------------------------

class TestLikelihoodCalculator:

    def test_init(self, schema):
        calc = LikelihoodCalculator(schema_loader=schema, temperature=1.0)
        assert calc.temperature == 1.0

    def test_custom_temperature(self, schema):
        calc = LikelihoodCalculator(schema_loader=schema, temperature=2.0)
        assert calc.temperature == 2.0

    def test_likelihood_is_valid_probability(self, calculator):
        vignette = _make_vignette(
            {"earnings": "high", "physical_demand": "light"},
            {"earnings": "low",  "physical_demand": "heavy"},
        )
        weights = np.array([0.8, 0.5])
        p = calculator.compute_choice_likelihood(vignette, "A", weights)
        assert 0.0 <= p <= 1.0

    def test_probabilities_sum_to_one(self, calculator):
        vignette = _make_vignette({"earnings": "high"}, {"earnings": "low"})
        weights = np.array([0.8, 0.5])
        p_a = calculator.compute_choice_likelihood(vignette, "A", weights)
        p_b = calculator.compute_choice_likelihood(vignette, "B", weights)
        assert np.isclose(p_a + p_b, 1.0)

    def test_extract_features_shape(self, calculator, schema):
        vignette = _make_vignette({"earnings": "high"}, {"earnings": "low"})
        features = calculator._extract_features(vignette.options[0])
        assert features.shape == (schema.n_dimensions,)

    def test_extract_features_empty_attributes(self, calculator, schema):
        vignette = _make_vignette({}, {})
        features = calculator._extract_features(vignette.options[0])
        assert features.shape == (schema.n_dimensions,)
        assert np.allclose(features, 0.0)

    def test_zero_weights_gives_fifty_fifty(self, calculator, schema):
        vignette = _make_vignette({"earnings": "high"}, {"earnings": "low"})
        weights = np.zeros(schema.n_dimensions)
        p_a = calculator.compute_choice_likelihood(vignette, "A", weights)
        assert np.isclose(p_a, 0.5, atol=1e-5)

    def test_strong_preference_dominates(self, calculator, schema):
        """A very strong financial preference should strongly favour the high-pay option."""
        vignette = _make_vignette({"earnings": "high"}, {"earnings": "low"})
        weights = np.array([10.0] + [0.0] * (schema.n_dimensions - 1))
        p_a = calculator.compute_choice_likelihood(vignette, "A", weights)
        assert p_a > 0.99

    def test_temperature_affects_probabilities(self, schema):
        vignette = _make_vignette({"earnings": "high"}, {"earnings": "low"})
        weights = np.array([0.8] + [0.5] * (schema.n_dimensions - 1))

        calc_low  = LikelihoodCalculator(schema_loader=schema, temperature=0.5)
        calc_high = LikelihoodCalculator(schema_loader=schema, temperature=2.0)

        p_low  = calc_low.compute_choice_likelihood(vignette, "A", weights)
        p_high = calc_high.compute_choice_likelihood(vignette, "A", weights)
        assert p_low != p_high

    def test_extreme_weights_still_finite(self, calculator, schema):
        vignette = _make_vignette({"earnings": "high"}, {"earnings": "low"})
        weights = np.array([100.0] * schema.n_dimensions)
        p = calculator.compute_choice_likelihood(vignette, "A", weights)
        assert np.isfinite(p)
        assert 0.0 <= p <= 1.0

    def test_negative_weights_still_valid(self, calculator, schema):
        vignette = _make_vignette({"earnings": "high"}, {"earnings": "low"})
        weights = np.array([-0.5] * schema.n_dimensions)
        p = calculator.compute_choice_likelihood(vignette, "A", weights)
        assert 0.0 <= p <= 1.0

    def test_create_likelihood_function_is_callable(self, calculator):
        vignette = _make_vignette({"earnings": "high"}, {"earnings": "low"})
        fn = calculator.create_likelihood_function(vignette, "A")
        assert callable(fn)

    def test_create_likelihood_function_returns_probability(self, calculator, schema):
        vignette = _make_vignette({"earnings": "high"}, {"earnings": "low"})
        fn = calculator.create_likelihood_function(vignette, "A")
        obs = {"vignette": vignette, "chosen_option": "A"}
        beta = np.array([0.5] * schema.n_dimensions)
        result = fn(obs, beta)
        assert 0.0 <= result <= 1.0

    def test_options_resolved_by_id(self, calculator, schema):
        """Options must be matched by option_id, not list position."""
        # Put B first in the list deliberately
        vignette = Vignette(
            vignette_id="order_test",
            category="test",
            scenario_text="Choose:",
            options=[
                VignetteOption(option_id="B", title="B", description="B", attributes={"earnings": "low"}),
                VignetteOption(option_id="A", title="A", description="A", attributes={"earnings": "high"}),
            ],
        )
        weights = np.array([10.0] + [0.0] * (schema.n_dimensions - 1))
        # Choosing "A" should still strongly prefer the high-earnings option
        p_a = calculator.compute_choice_likelihood(vignette, "A", weights)
        assert p_a > 0.99

    def test_insufficient_options_raises(self, calculator):
        vignette = Vignette(
            vignette_id="bad",
            category="test",
            scenario_text="Only one option:",
            options=[
                VignetteOption(option_id="X", title="Only", description="", attributes={}),
            ],
        )
        with pytest.raises(ValueError):
            calculator.compute_choice_likelihood(vignette, "A", np.array([0.5, 0.5]))
