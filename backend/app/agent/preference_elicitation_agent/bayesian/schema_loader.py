"""
Schema loader for job attribute configurations.

Derives preference dimensions and feature extraction mappings dynamically
from any conforming job_attributes JSON schema. This makes the Bayesian
system configurable: swapping the JSON file is all that's needed to change
the attribute space and dimensions.

Schema contract (attributes array entries):
    - name (str): attribute key used in vignette option.attributes
    - group (str): maps to a preference dimension (snake_cased)
    - type (str): "ordered" | "categorical"
    - coding (str): "linear" | "dummy"
    - levels (list): ordered list of levels
    - base_level_id (str, optional): reference level for dummy coding
    - direction (str, optional): "positive" | "negative" | "neutral"
      (if omitted, defaults to "positive")
"""

import json
import re
from pathlib import Path
from typing import Any


def _group_to_dimension(group: str) -> str:
    """Convert a group label to a snake_case dimension name.

    Examples:
        "Financial"       -> "financial_importance"
        "Work Environment" -> "work_environment_importance"
        "Future Prospects" -> "future_prospects_importance"
    """
    snake = re.sub(r"[\s/]+", "_", group.strip()).lower()
    snake = re.sub(r"[^a-z0-9_]", "", snake)
    return f"{snake}_importance"


class AttributeSpec:
    """Parsed specification for a single job attribute."""

    def __init__(self, raw: dict[str, Any]):
        self.name: str = raw["name"]
        self.group: str = raw["group"]
        self.dimension: str = _group_to_dimension(self.group)
        self.attr_type: str = raw["type"]          # "ordered" | "categorical"
        self.coding: str = raw["coding"]            # "linear" | "dummy"
        self.levels: list[dict] = raw["levels"]
        self.base_level_id: str | None = raw.get("base_level_id")
        self.direction: str = raw.get("direction", "positive")

        # For linear-coded attributes, build a value map: level_id -> float
        self._level_values: dict[str, float] = {}
        if self.coding == "linear":
            raw_vals = [lv.get("value") for lv in self.levels if lv.get("value") is not None]
            if raw_vals:
                # Use explicit numeric values if present
                for lv in self.levels:
                    if lv.get("value") is not None:
                        self._level_values[lv["id"]] = float(lv["value"])
            else:
                # Fall back to 0-based integer index
                for idx, lv in enumerate(self.levels):
                    self._level_values[lv["id"]] = float(idx)

        # For dummy-coded, non-base levels map to 1.0; base maps to 0.0
        if self.coding == "dummy":
            for lv in self.levels:
                self._level_values[lv["id"]] = (
                    0.0 if lv["id"] == self.base_level_id else 1.0
                )

    def encode(self, level_id: str) -> float:
        """
        Encode a level id to a scalar feature value.

        For linear attributes: normalised value in [0, 1].
        For dummy attributes:  0.0 (base) or 1.0 (non-base).

        Returns 0.0 if the level_id is unknown.
        """
        raw_val = self._level_values.get(level_id, 0.0)

        if self.coding == "linear" and self._level_values:
            if level_id not in self._level_values:
                return 0.0
            all_vals = list(self._level_values.values())
            lo, hi = min(all_vals), max(all_vals)
            if hi > lo:
                normalised = (raw_val - lo) / (hi - lo)
            else:
                normalised = 0.5
            # Flip if direction is negative (e.g. physical_demand: high = worse)
            if self.direction == "negative":
                normalised = 1.0 - normalised
            return normalised

        # Dummy coding — direction handled at feature-extraction level
        if self.direction == "negative":
            return 1.0 - raw_val
        return raw_val


class SchemaLoader:
    """
    Loads a job-attribute schema JSON and derives everything the Bayesian
    system needs: dimensions, attribute specs, feature-extraction logic,
    and sensible default priors.

    Usage
    -----
        loader = SchemaLoader.from_file("job_attributes_reduced.json")
        loader.dimensions          # ["financial_importance", ...]
        loader.n_dimensions        # 4
        loader.default_prior_mean  # [0.5, 0.5, 0.5, 0.5]
        feature_vec = loader.extract_features({"earnings_per_month": "earn_30k", ...})
    """

    def __init__(self, schema: dict[str, Any]):
        raw_attributes = schema.get("attributes", [])
        if not raw_attributes:
            raise ValueError("Schema must contain a non-empty 'attributes' array.")

        self._specs: list[AttributeSpec] = [AttributeSpec(a) for a in raw_attributes]

        # Derive ordered, deduplicated dimensions (preserving first-seen order)
        seen: dict[str, None] = {}
        for spec in self._specs:
            seen.setdefault(spec.dimension, None)
        self._dimensions: list[str] = list(seen.keys())

        # Map dimension -> list of AttributeSpec that contribute to it
        self._dim_to_specs: dict[str, list[AttributeSpec]] = {d: [] for d in self._dimensions}
        for spec in self._specs:
            self._dim_to_specs[spec.dimension].append(spec)

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "SchemaLoader":
        """Load schema from a JSON file path."""
        with open(path, "r", encoding="utf-8") as fh:
            schema = json.load(fh)
        return cls(schema)

    @classmethod
    def from_dict(cls, schema: dict[str, Any]) -> "SchemaLoader":
        """Load schema from an already-parsed dict."""
        return cls(schema)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> list[str]:
        """Ordered list of preference dimension names derived from schema groups."""
        return list(self._dimensions)

    @property
    def n_dimensions(self) -> int:
        """Number of preference dimensions."""
        return len(self._dimensions)

    @property
    def attribute_names(self) -> list[str]:
        """All attribute names in schema order."""
        return [s.name for s in self._specs]

    @property
    def default_prior_mean(self) -> list[float]:
        """Neutral prior: 0.5 for every dimension."""
        return [0.5] * self.n_dimensions

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, attributes: dict[str, Any]) -> list[float]:
        """
        Convert a vignette option's attributes dict into a feature vector
        aligned with self.dimensions.

        Each dimension's value is the mean of the encoded values of all
        attributes that belong to that dimension.  If no attribute for a
        dimension is present in the input, that dimension gets 0.0.

        Args:
            attributes: dict mapping attribute name -> level_id (str) or
                        raw numeric value (for linear attributes).

        Returns:
            Feature vector of length self.n_dimensions.
        """
        features = [0.0] * self.n_dimensions

        for dim_idx, dim in enumerate(self._dimensions):
            specs = self._dim_to_specs[dim]
            scores: list[float] = []

            for spec in specs:
                raw = attributes.get(spec.name)
                if raw is None:
                    continue

                if spec.coding == "linear":
                    # Accept either a level_id string or a raw numeric value
                    if isinstance(raw, str):
                        score = spec.encode(raw)
                    else:
                        # Raw number — normalise using the level value range
                        all_vals = list(spec._level_values.values())
                        if all_vals:
                            lo, hi = min(all_vals), max(all_vals)
                            score = (float(raw) - lo) / (hi - lo) if hi > lo else 0.5
                        else:
                            score = float(raw)
                        if spec.direction == "negative":
                            score = 1.0 - score
                else:
                    # Dummy-coded — raw value should be a level_id string
                    score = spec.encode(str(raw))

                scores.append(score)

            features[dim_idx] = sum(scores) / len(scores) if scores else 0.0

        return features

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Human-readable summary of the loaded schema."""
        return {
            "n_dimensions": self.n_dimensions,
            "dimensions": self.dimensions,
            "attributes_per_dimension": {
                dim: [s.name for s in specs]
                for dim, specs in self._dim_to_specs.items()
            },
        }
