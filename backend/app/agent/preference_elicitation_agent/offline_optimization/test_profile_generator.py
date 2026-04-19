"""
Unit tests for ProfileGenerator.
"""

import pytest
import json
import tempfile
from pathlib import Path
from app.agent.preference_elicitation_agent.offline_optimization.profile_generator import ProfileGenerator


@pytest.fixture
def minimal_config():
    """Minimal valid configuration for testing."""
    return {
        "attributes": [
            {
                "name": "wage",
                "label": "Monthly wage",
                "group": "Financial",
                "type": "ordered",
                "coding": "linear",
                "levels": [
                    {"id": "w_low", "label": "KES 10,000", "value": 10000},
                    {"id": "w_high", "label": "KES 20,000", "value": 20000}
                ]
            },
            {
                "name": "flexibility",
                "label": "Work schedule",
                "group": "Work Life Balance",
                "type": "categorical",
                "coding": "dummy",
                "base_level_id": "flex_fixed",
                "levels": [
                    {"id": "flex_fixed", "label": "Fixed shifts"},
                    {"id": "flex_flexible", "label": "Flexible hours"}
                ]
            }
        ],
        "model": {
            "utility_spec": "mnl",
            "parameters": [
                {
                    "name": "beta_wage",
                    "attribute": "wage",
                    "coding": "linear",
                    "prior": {"distribution": "normal", "mean": 0.8, "sd": 0.5}
                },
                {
                    "name": "beta_flex_flexible",
                    "attribute": "flexibility",
                    "level_id": "flex_flexible",
                    "coding": "dummy",
                    "prior": {"distribution": "normal", "mean": 0.4, "sd": 0.5}
                }
            ]
        },
        "attribute_directions": {
            "wage": "positive",
            "flexibility": "positive"
        }
    }


@pytest.fixture
def config_file(minimal_config):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(minimal_config, f)
        return f.name


class TestProfileGenerator:
    """Tests for ProfileGenerator class."""

    def test_init_with_config_file(self, config_file):
        """Test initialization with config file."""
        generator = ProfileGenerator(config_path=config_file)
        assert generator.config is not None
        assert len(generator.attributes) == 2
        assert generator.attribute_directions["wage"] == "positive"

    def test_load_config_file_not_found(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            ProfileGenerator(config_path="/nonexistent/path/config.json")

    def test_load_config_invalid_json(self):
        """Test error handling for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{invalid json")
            invalid_file = f.name

        with pytest.raises(json.JSONDecodeError):
            ProfileGenerator(config_path=invalid_file)

    def test_generate_all_profiles(self, config_file):
        """Test profile generation."""
        generator = ProfileGenerator(config_path=config_file)
        profiles = generator.generate_all_profiles()

        # Should generate 2 wage levels × 2 flexibility levels = 4 profiles
        assert len(profiles) == 4

        # Check profile structure
        for profile in profiles:
            assert "wage" in profile
            assert "flexibility" in profile
            assert profile["wage"] in [10000, 20000]
            assert profile["flexibility"] in ["flex_fixed", "flex_flexible"]

    def test_generate_all_profiles_with_max_limit(self, config_file):
        """Test profile generation with max limit."""
        generator = ProfileGenerator(config_path=config_file)
        profiles = generator.generate_all_profiles(max_profiles=2)

        assert len(profiles) == 2

    def test_encode_profile_linear_coding(self, config_file):
        """Test profile encoding for linear attributes."""
        generator = ProfileGenerator(config_path=config_file)
        # Use level IDs as expected by SchemaLoader for non-numeric profiles
        profile = {"wage": 10000, "flexibility": "flex_fixed"}

        features = generator.encode_profile(profile)

        # minimal_config has 2 groups (Financial, Work Life Balance) → 2 dimensions
        assert len(features) == 2
        assert features[0] == 0.0  # financial: min wage → normalised 0.0
        assert features[1] == 0.0  # work_life_balance: base level flex_fixed → 0.0

    def test_encode_profile_categorical_coding(self, config_file):
        """Test profile encoding for categorical attributes."""
        generator = ProfileGenerator(config_path=config_file)
        profile = {"wage": 20000, "flexibility": "flex_flexible"}

        features = generator.encode_profile(profile)

        # minimal_config has 2 groups (Financial, Work Life Balance) → 2 dimensions
        assert len(features) == 2
        assert features[0] == 1.0  # financial: max wage → normalised 1.0
        assert features[1] == 1.0  # work_life_balance: non-base level flex_flexible → 1.0

    def test_profile_to_string(self, config_file):
        """Test profile to string conversion."""
        generator = ProfileGenerator(config_path=config_file)
        profile = {"wage": 10000, "flexibility": "flex_flexible"}

        result = generator.profile_to_string(profile)

        assert "Monthly wage: KES 10,000" in result
        assert "Work schedule: Flexible hours" in result
        assert "|" in result

    def test_get_attribute_info(self, config_file):
        """Test attribute info retrieval."""
        generator = ProfileGenerator(config_path=config_file)
        info = generator.get_attribute_info()

        assert info["total_attributes"] == 2
        assert info["total_combinations"] == 4
        assert len(info["attributes"]) == 2

        wage_info = next(a for a in info["attributes"] if a["name"] == "wage")
        assert wage_info["type"] == "ordered"
        assert wage_info["num_levels"] == 2
        assert wage_info["direction"] == "positive"

    def test_calculate_total_combinations(self, config_file):
        """Test total combinations calculation."""
        generator = ProfileGenerator(config_path=config_file)
        total = generator._calculate_total_combinations()

        assert total == 4  # 2 × 2

    def test_real_config_profile_count(self):
        """Test profile generation with real configuration."""
        # This test assumes the real config exists in the same directory
        config_path = Path(__file__).parent.parent / "config" / "preference_parameters.json"
        if not config_path.exists():
            pytest.skip("Real config file not found")

        generator = ProfileGenerator(config_path=str(config_path))
        profiles = generator.generate_all_profiles()

        # Real config (v1.1.0): 4 earnings × 2 physical_demand × 3 social_interaction × 3 career_growth
        # = 4 × 2 × 3 × 3 = 72
        assert len(profiles) == 72

    def test_encode_profile_with_real_config(self):
        """Test encoding with real configuration."""
        config_path = Path(__file__).parent.parent / "config" / "preference_parameters.json"
        if not config_path.exists():
            pytest.skip("Real config file not found")

        generator = ProfileGenerator(config_path=str(config_path))
        profiles = generator.generate_all_profiles(max_profiles=1)

        features = generator.encode_profile(profiles[0])

        # Real config (v1.1.0) has 3 preference dimensions:
        # Financial, Work Environment, Future Prospects
        assert len(features) == 3
        assert all(isinstance(f, float) for f in features)

    def test_profile_uniqueness(self, config_file):
        """Test that generated profiles are unique."""
        generator = ProfileGenerator(config_path=config_file)
        profiles = generator.generate_all_profiles()

        # Convert to tuples for set comparison
        profile_tuples = [tuple(sorted(p.items())) for p in profiles]

        assert len(profile_tuples) == len(set(profile_tuples))

    def test_profile_coverage(self, config_file):
        """Test that all attribute levels are covered."""
        generator = ProfileGenerator(config_path=config_file)
        profiles = generator.generate_all_profiles()

        # Check wage coverage
        wages = {p["wage"] for p in profiles}
        assert wages == {10000, 20000}

        # Check flexibility coverage
        flexibilities = {p["flexibility"] for p in profiles}
        assert flexibilities == {"flex_fixed", "flex_flexible"}
