"""
Profile Generator for offline vignette optimization.

Generates all possible job profile combinations from attribute specifications.
"""

import json
import itertools
import logging
from typing import Dict, List, Any
from pathlib import Path
import sys

# Allow importing SchemaLoader from parent package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from app.agent.preference_elicitation_agent.bayesian.schema_loader import SchemaLoader


class ProfileGenerator:
    """Generates candidate job profiles from attribute specifications."""

    def __init__(self, config_path: str = None):
        """
        Initialize the profile generator.

        Args:
            config_path: Path to preference_parameters.json
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        if config_path is None:
            config_path = Path(__file__).parent / "preference_parameters.json"

        self.config = self._load_config(config_path)
        self.attributes = self.config["attributes"]
        # Load SchemaLoader for schema-driven encoding
        self.schema_loader = SchemaLoader.from_file(str(config_path))
        # Build attribute_directions from schema for backward compat
        self.attribute_directions = {
            attr["name"]: ("negative" if attr.get("direction") == "negative" else "positive")
            for attr in self.attributes
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config: {e}")
            raise

    def generate_all_profiles(self, max_profiles: int = None) -> List[Dict[str, Any]]:
        """
        Generate all possible job profile combinations.

        Args:
            max_profiles: Maximum number of profiles to generate (for testing)

        Returns:
            List of profile dictionaries
        """
        self.logger.info("Generating candidate profiles...")

        # extract all attribute names and their possible values
        attribute_combinations = {}

        for attr in self.attributes:
            attr_name = attr["name"]

            # Get all possible values for this attribute
            has_numeric_values = all("value" in level for level in attr["levels"])
            if attr["type"] == "ordered" and has_numeric_values:
                # Ordered with numeric values: use the numeric values directly
                values = [level["value"] for level in attr["levels"]]
            else:
                # Categorical or ordered-by-id: use level IDs
                values = [level["id"] for level in attr["levels"]]

            attribute_combinations[attr_name] = values

        # Generate all combinations using itertools.product
        attribute_names = list(attribute_combinations.keys())
        value_lists = [attribute_combinations[name] for name in attribute_names]

        all_combinations = list(itertools.product(*value_lists))

        # Convert to profile dictionaries
        profiles = []
        for combination in all_combinations:
            profile = {
                name: value
                for name, value in zip(attribute_names, combination)
            }
            profiles.append(profile)

            if max_profiles and len(profiles) >= max_profiles:
                break

        self.logger.info(
            f"Generated {len(profiles)} candidate profiles "
            f"from {len(all_combinations)} total combinations"
        )

        return profiles

    def encode_profile(self, profile: Dict[str, Any]) -> List[float]:
        """
        Encode a profile as an N-dimensional preference feature vector.

        Delegates to SchemaLoader so encoding is always consistent with
        the online system (LikelihoodCalculator._extract_features).

        Args:
            profile: Profile dictionary with attribute values (level IDs for
                     categoricals, numeric values for ordered attributes)

        Returns:
            Feature vector (N dimensions, one per preference dimension in schema)
        """
        return self.schema_loader.extract_features(profile)

    def profile_to_string(self, profile: Dict[str, Any]) -> str:
        """
        Convert profile to human-readable string.

        Args:
            profile: Profile dictionary

        Returns:
            String representation
        """
        parts = []
        for attr in self.attributes:
            attr_name = attr["name"]
            value = profile[attr_name]

            has_numeric_values = all("value" in l for l in attr["levels"])
            if attr["type"] == "ordered" and has_numeric_values:
                # Find the level with this numeric value
                level = next(
                    (l for l in attr["levels"] if l["value"] == value),
                    None
                )
            else:
                # Categorical or ordered-by-id: match by level id
                level = next(
                    (l for l in attr["levels"] if l["id"] == value),
                    None
                )
            if level:
                parts.append(f"{attr['label']}: {level['label']}")

        return " | ".join(parts)

    def get_attribute_info(self) -> Dict[str, Any]:
        """
        Get information about attributes for documentation.

        Returns:
            Dictionary with attribute metadata
        """
        return {
            "total_attributes": len(self.attributes),
            "attributes": [
                {
                    "name": attr["name"],
                    "type": attr["type"],
                    "num_levels": len(attr["levels"]),
                    "direction": self.attribute_directions.get(attr["name"])
                }
                for attr in self.attributes
            ],
            "total_combinations": self._calculate_total_combinations()
        }

    def _calculate_total_combinations(self) -> int:
        """Calculate total number of possible profile combinations."""
        total = 1
        for attr in self.attributes:
            total *= len(attr["levels"])
        return total
