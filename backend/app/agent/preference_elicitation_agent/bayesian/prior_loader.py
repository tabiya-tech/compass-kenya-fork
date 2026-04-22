"""
Loads empirical DCE priors from a JSON file into term-level prior distributions.

The JSON format (dce_dynamic_priors_batch*.json) stores MNL utility coefficients
estimated from a pilot study. This loader maps those coefficients onto the
SchemaLoader's term_names by matching attribute groups positionally, so name
differences between the pilot study and the current schema are handled gracefully.

Usage:
    loader = SchemaLoader.from_file("preference_parameters.json")
    result = PriorsLoader.load("dce_dynamic_priors_batch1_short.json", loader)
    # result.mean          -> List[float], one per term
    # result.variances     -> List[float], one per term (from prior_sd²)
    # result.fim_determinant -> float, product(1/v), used for ratio-mode stopping
    # result.term_names    -> List[str], aligned with schema_loader.term_names
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema_loader import SchemaLoader, AttributeSpec


@dataclass
class PriorsResult:
    """Term-level prior distribution loaded from a DCE priors file."""
    mean: list[float]
    variances: list[float]
    fim_determinant: float
    term_names: list[str]


@dataclass
class _JsonAttrEntry:
    """Parsed attribute entry from the priors JSON."""
    key: str                          # JSON top-level key
    attr_type: str                    # "continuous" or "categorical_treatment"
    levels: list[tuple[float, float]] # (prior_mean, prior_sd) per non-reference level


class PriorsLoader:
    """
    Loads a DCE priors JSON file and maps coefficients to SchemaLoader term order.

    Matching strategy
    -----------------
    Both the JSON and schema represent the same set of job attributes (earnings,
    physical demand, social interaction, career growth), but may use different
    attribute names and level IDs. The loader matches them by:

    1. Parsing JSON into ordered attribute groups (continuous first, then
       categorical_treatment entries in the order they appear in the JSON).
    2. Parsing schema specs into the same attribute order (linear first, then dummy).
    3. Fuzzy-matching each JSON attribute key → nearest schema attribute name.
    4. Within each matched pair, mapping JSON non-reference levels → schema
       non-base levels positionally (first non-ref → first non-base, etc.).

    This is robust to naming differences such as "earnings_k" vs "earnings_per_month"
    or "physical_demand_risk" vs "physical_demand".
    """

    @classmethod
    def load(cls, path: str | Path, schema_loader: SchemaLoader) -> PriorsResult:
        """
        Load priors from a JSON file and align to schema term order.

        Args:
            path: Path to the DCE priors JSON file.
            schema_loader: Loaded SchemaLoader with term_names defined.

        Returns:
            PriorsResult with mean, variances, fim_determinant, term_names.

        Raises:
            ValueError: If the number of attributes or levels cannot be reconciled.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)

        json_attrs = cls._parse_json_attrs(raw)
        schema_attrs = schema_loader._specs

        if len(json_attrs) != len(schema_attrs):
            raise ValueError(
                f"JSON has {len(json_attrs)} attribute entries but schema has "
                f"{len(schema_attrs)} attributes. They must match in count."
            )

        # Match JSON attrs → schema attrs by fuzzy name similarity
        # Use a greedy best-match (closest normalised name)
        matched_pairs = cls._match_attributes(json_attrs, schema_attrs)

        # Compute earnings range for scaling linear/continuous terms
        earnings_range_k = cls._get_earnings_range_k(schema_loader)

        # Build aligned prior vectors following schema term order
        # schema term order: linear attrs first (one term each), then dummy attrs
        # (one term per non-base level), in schema spec order
        term_name_to_prior: dict[str, tuple[float, float]] = {}

        for json_attr, schema_spec in matched_pairs:
            is_linear = schema_spec.coding == "linear"

            if len(json_attr.levels) != cls._n_nonbase_levels(schema_spec):
                raise ValueError(
                    f"JSON attribute '{json_attr.key}' has {len(json_attr.levels)} "
                    f"non-reference levels but schema attribute '{schema_spec.name}' "
                    f"has {cls._n_nonbase_levels(schema_spec)} non-base levels."
                )

            non_base_level_ids = [
                lv["id"] for lv in schema_spec.levels
                if lv["id"] != schema_spec.base_level_id
            ]

            for i, (pmean, psd) in enumerate(json_attr.levels):
                if is_linear:
                    # Linear: single term named after the attribute
                    term_name = schema_spec.name
                    pmean = pmean * earnings_range_k
                    psd = psd * earnings_range_k
                else:
                    # Dummy: term named "<attr>_<level_id>"
                    term_name = f"{schema_spec.name}_{non_base_level_ids[i]}"

                term_name_to_prior[term_name] = (pmean, psd)

        # Assemble vectors in schema term order
        schema_term_names = schema_loader.term_names
        means: list[float] = []
        variances: list[float] = []

        for term in schema_term_names:
            if term not in term_name_to_prior:
                raise ValueError(
                    f"Schema term '{term}' has no matching prior. "
                    f"Available: {list(term_name_to_prior.keys())}"
                )
            m, s = term_name_to_prior[term]
            means.append(m)
            variances.append(s ** 2)

        fim_det = math.prod(1.0 / v for v in variances)

        return PriorsResult(
            mean=means,
            variances=variances,
            fim_determinant=fim_det,
            term_names=schema_term_names,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _parse_json_attrs(cls, raw: dict[str, Any]) -> list["_JsonAttrEntry"]:
        """Parse JSON top-level entries into ordered _JsonAttrEntry list."""
        entries: list[_JsonAttrEntry] = []
        for key, entry in raw.items():
            if entry.get("type") == "continuous":
                entries.append(_JsonAttrEntry(
                    key=key,
                    attr_type="continuous",
                    levels=[(entry["prior_mean"], entry["prior_sd"])],
                ))
            elif entry.get("type") == "categorical_treatment":
                level_priors = [
                    (lv["prior_mean"], lv["prior_sd"])
                    for lv in entry.get("levels", [])
                ]
                entries.append(_JsonAttrEntry(
                    key=key,
                    attr_type="categorical_treatment",
                    levels=level_priors,
                ))
        return entries

    @classmethod
    def _match_attributes(
        cls,
        json_attrs: list[_JsonAttrEntry],
        schema_specs: list[AttributeSpec],
    ) -> list[tuple[_JsonAttrEntry, AttributeSpec]]:
        """
        Pair each JSON attribute with the closest schema attribute by name similarity.

        Uses a greedy approach: for each JSON attr, find the best-scoring unmatched
        schema attr. Similarity is measured by longest common substring length after
        normalisation (strips underscores, lowercased).
        """
        def normalise(t: str) -> str:
            return t.replace("_", "").replace(" ", "").lower()

        def similarity(a: str, b: str) -> int:
            a_n, b_n = normalise(a), normalise(b)
            # Count characters of longer that appear in shorter
            shorter, longer = (a_n, b_n) if len(a_n) <= len(b_n) else (b_n, a_n)
            return sum(1 for c in shorter if c in longer)

        remaining = list(schema_specs)
        pairs: list[tuple[_JsonAttrEntry, AttributeSpec]] = []

        for jattr in json_attrs:
            best = max(remaining, key=lambda s: similarity(jattr.key, s.name))
            pairs.append((jattr, best))
            remaining.remove(best)

        return pairs

    @classmethod
    def _n_nonbase_levels(cls, spec: AttributeSpec) -> int:
        """Number of non-base levels (= number of dummy terms) for a schema attribute."""
        if spec.coding == "linear":
            return 1
        return sum(1 for lv in spec.levels if lv["id"] != spec.base_level_id)

    @classmethod
    def _get_earnings_range_k(cls, schema_loader: SchemaLoader) -> float:
        """
        Derive the earnings value range in thousands of KES from the schema.
        Falls back to 55.0 (the 15k-70k KES range used in batch 1).
        """
        for spec in schema_loader._specs:
            if spec.coding == "linear" and "earning" in spec.name.lower():
                vals = list(spec._level_values.values())
                if vals:
                    return (max(vals) - min(vals)) / 1000.0
        return 55.0
