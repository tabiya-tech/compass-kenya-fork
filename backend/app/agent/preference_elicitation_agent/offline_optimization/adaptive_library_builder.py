"""
Adaptive Library Builder for offline vignette optimization.

Builds a library of 40 diverse vignettes for adaptive runtime selection.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Set
from itertools import combinations
from collections import defaultdict
import random


class AdaptiveLibraryBuilder:
    """Builds library of diverse vignettes for adaptive selection."""

    def __init__(self, profile_generator):
        """
        Initialize the adaptive library builder.

        Args:
            profile_generator: ProfileGenerator instance for encoding profiles
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.profile_generator = profile_generator
        # Use term-level count so individual attribute signals are kept separate.
        self.n_params = profile_generator.schema_loader.n_terms

    def build_adaptive_library(
        self,
        profiles: List[Dict[str, Any]],
        num_library: int = 40,
        excluded_vignettes: List[Tuple[Dict, Dict]] = None,
        prior_mean: np.ndarray = None,
        diversity_weight: float = 0.3,
        sample_size: int = 10000,
        n_focused_per_attr: int = 3,
        n_dim_targeted_per_dim: int = 3
    ) -> List[Tuple[Dict, Dict]]:
        """
        Build adaptive library using diversity-aware selection with sampling.

        Phase 1: Focused vignettes — pairs where only one attribute differs.
        These enable true adaptivity: the Bayesian trace(FIM @ Cov) criterion
        gives focused vignettes a high score when the persona is uncertain in
        exactly that dimension, and a low score otherwise.

        Phase 2: Greedy D-optimal + diversity for remaining slots.

        Args:
            profiles: List of non-dominated job profiles
            num_library: Number of vignettes in adaptive library (default: 40)
            excluded_vignettes: Vignettes already in static set (to exclude)
            prior_mean: Prior mean for preference weights
            diversity_weight: Weight for diversity term (0-1, default: 0.3)
            sample_size: Number of candidates to sample per round (default: 10,000)
            n_focused_per_attr: Focused vignettes to reserve per attribute (default: 3)

        Returns:
            List of vignette pairs for adaptive library
        """
        self.logger.info(
            f"Building adaptive library with {num_library} vignettes..."
        )

        if prior_mean is None:
            prior_mean = np.ones(self.n_params) * 0.5  # neutral prior for all terms

        if excluded_vignettes is None:
            excluded_vignettes = []

        # Convert excluded vignettes to set for faster lookup
        excluded_set = set(
            (self._profile_hash(a), self._profile_hash(b))
            for a, b in excluded_vignettes
        )

        # Calculate total possible pairs
        total_possible_pairs = len(profiles) * (len(profiles) - 1) // 2
        self.logger.info(f"Total possible vignette pairs: {total_possible_pairs:,}")

        selected_hashes = set()
        selected_vignettes = []
        selected_feature_vectors = []

        # =====================================================================
        # Phase 1: Focused vignettes — one attribute differs, all others equal.
        # These are critical for true adaptivity: the Bayesian criterion
        # trace(FIM @ Cov) selects them when a persona's posterior has high
        # variance in exactly that attribute's direction.
        # =====================================================================
        attr_names = [attr["name"] for attr in self.profile_generator.attributes]
        self.logger.info(
            f"Phase 1: Building focused vignettes ({n_focused_per_attr} per attribute × "
            f"{len(attr_names)} attributes = up to {n_focused_per_attr * len(attr_names)} focused)"
        )

        for attr_name in attr_names:
            if len(selected_vignettes) >= num_library:
                break
            focused_candidates = self._find_focused_pairs_for_attr(profiles, attr_name)
            count_added = 0

            for pair in focused_candidates:
                if count_added >= n_focused_per_attr:
                    break
                if len(selected_vignettes) >= num_library:
                    break

                pair_hash = (self._profile_hash(pair[0]), self._profile_hash(pair[1]))
                if pair_hash in excluded_set or pair_hash in selected_hashes:
                    continue
                if self._has_pairwise_dominance(pair[0], pair[1]):
                    continue
                if self._has_excessive_wage_gap(pair[0], pair[1]):
                    continue
                if self._has_attribute_cancellation(pair[0], pair[1]):
                    continue

                selected_vignettes.append(pair)
                selected_hashes.add(pair_hash)
                x_a = np.array(self.profile_generator.encode_profile_terms(pair[0]))
                x_b = np.array(self.profile_generator.encode_profile_terms(pair[1]))
                selected_feature_vectors.append(x_a - x_b)
                count_added += 1

            self.logger.info(
                f"  Attribute '{attr_name}': added {count_added} focused vignettes "
                f"(from {len(focused_candidates)} candidates)"
            )

        self.logger.info(
            f"Phase 1 complete: {len(selected_vignettes)} focused vignettes added"
        )

        # =====================================================================
        # Phase 1.5: Dimension-targeted trade-off vignettes.
        # For monotone attributes (earnings, physical demand), focused pairs
        # are always dominated (higher earnings is universally preferred).
        # Instead, we find genuine trade-off pairs where ONE dimension accounts
        # for ≥50% of the variance in ||x_diff||² — these isolate a dimension
        # signal without being dominated.
        # Example: (30k+heavy+alone) vs (50k+light+peers) has conflicting
        # earnings/work-env trade-off; if earnings² > work_env² + career²,
        # this vignette is classified as "earnings-dominant".
        # =====================================================================
        schema_loader = self.profile_generator.schema_loader
        n_schema_terms = schema_loader.n_terms
        term_names = schema_loader.term_names  # e.g. ['earnings_per_month', 'physical_demand_phys_safe', ...]

        self.logger.info(
            f"Phase 1.5: Building term-targeted vignettes "
            f"({n_dim_targeted_per_dim} per term × {n_schema_terms} terms = "
            f"up to {n_dim_targeted_per_dim * n_schema_terms} targeted)"
        )

        # Enumerate all valid pairs once (72 profiles → 2556 pairs at most)
        all_pairs_for_targeting = list(combinations(profiles, 2))

        for dim_idx in range(n_schema_terms):
            if len(selected_vignettes) >= num_library:
                break
            dim_name = term_names[dim_idx] if dim_idx < len(term_names) else f"term_{dim_idx}"

            # Score each pair by how much this dimension dominates x_diff
            scored_pairs = []
            for pair in all_pairs_for_targeting:
                pair_hash = (self._profile_hash(pair[0]), self._profile_hash(pair[1]))
                if pair_hash in excluded_set or pair_hash in selected_hashes:
                    continue
                if self._has_pairwise_dominance(pair[0], pair[1]):
                    continue
                if self._has_excessive_wage_gap(pair[0], pair[1]):
                    continue
                if self._has_attribute_cancellation(pair[0], pair[1]):
                    continue

                x_a = np.array(self.profile_generator.encode_profile_terms(pair[0]))
                x_b = np.array(self.profile_generator.encode_profile_terms(pair[1]))
                x_diff = x_a - x_b

                total_sq = np.dot(x_diff, x_diff)
                if total_sq < 1e-8:
                    continue

                dim_sq = x_diff[dim_idx] ** 2
                dominance_ratio = dim_sq / total_sq  # fraction of variance in target dim

                if dominance_ratio >= 0.5:  # this dim accounts for ≥50% of signal
                    scored_pairs.append((dominance_ratio, pair))

            # Sort by dominance ratio descending (most dimension-specific first)
            scored_pairs.sort(key=lambda t: -t[0])

            count_added = 0
            for _, pair in scored_pairs:
                if count_added >= n_dim_targeted_per_dim:
                    break
                if len(selected_vignettes) >= num_library:
                    break
                pair_hash = (self._profile_hash(pair[0]), self._profile_hash(pair[1]))
                if pair_hash in selected_hashes:
                    continue

                selected_vignettes.append(pair)
                selected_hashes.add(pair_hash)
                x_a = np.array(self.profile_generator.encode_profile_terms(pair[0]))
                x_b = np.array(self.profile_generator.encode_profile_terms(pair[1]))
                selected_feature_vectors.append(x_a - x_b)
                count_added += 1

            self.logger.info(
                f"  Term '{dim_name}' (idx={dim_idx}): added {count_added} targeted vignettes "
                f"(from {len(scored_pairs)} candidates with dominance_ratio ≥ 0.5)"
            )

        self.logger.info(
            f"Phase 1.5 complete: {len(selected_vignettes)} total vignettes so far "
            f"(focused + term-targeted)"
        )

        # =====================================================================
        # Phase 2: Greedy D-optimal + diversity for remaining slots
        # =====================================================================
        remaining_slots = num_library - len(selected_vignettes)
        self.logger.info(f"Phase 2: Filling {remaining_slots} remaining slots with greedy diversity selection")

        for round_idx in range(remaining_slots):
            total_so_far = len(selected_vignettes) + 1
            if round_idx % 10 == 0:
                self.logger.info(f"Selecting greedy vignette {round_idx + 1}/{remaining_slots} (total {total_so_far}/{num_library})...")

            best_vignette = None
            best_score = -np.inf

            # Sample candidate pairs for this round
            candidates_to_evaluate = []
            sampled_this_round = set()

            # Calculate max possible unique pairs
            max_possible_pairs = len(profiles) * (len(profiles) - 1) // 2
            target_sample_size = min(sample_size, max_possible_pairs)

            # Prevent infinite loop: stop if we can't find new candidates after many attempts
            max_attempts = target_sample_size * 10
            attempts = 0

            while len(candidates_to_evaluate) < target_sample_size and attempts < max_attempts:
                attempts += 1

                # Random sample
                i = random.randint(0, len(profiles) - 1)  # nosec B311
                j = random.randint(0, len(profiles) - 1)  # nosec B311

                if i == j:
                    continue

                # Ensure consistent ordering
                if i > j:
                    i, j = j, i

                pair_key = (i, j)
                if pair_key in sampled_this_round:
                    continue

                sampled_this_round.add(pair_key)
                vignette_pair = (profiles[i], profiles[j])

                # Skip if excluded or already selected
                vignette_hash = (
                    self._profile_hash(vignette_pair[0]),
                    self._profile_hash(vignette_pair[1])
                )
                if vignette_hash in excluded_set or vignette_hash in selected_hashes:
                    continue

                candidates_to_evaluate.append(vignette_pair)

            self.logger.info(f"  Sampled {len(candidates_to_evaluate):,} unique candidates")

            for vignette_pair in candidates_to_evaluate:
                # Skip if vignette has attribute cancellation (opposing changes in aggregated dimensions)
                if self._has_attribute_cancellation(vignette_pair[0], vignette_pair[1]):
                    continue

                # Skip if one option dominates or quasi-dominates the other
                if self._has_pairwise_dominance(vignette_pair[0], vignette_pair[1]):
                    continue

                # Skip if wage gap is too large (psychological anchoring)
                if self._has_excessive_wage_gap(vignette_pair[0], vignette_pair[1]):
                    continue

                # Compute informativeness score (FIM determinant)
                informativeness = self._compute_informativeness(
                    vignette_pair[0],
                    vignette_pair[1],
                    prior_mean
                )

                # Compute diversity score (distance to selected vignettes)
                diversity = self._compute_diversity(
                    vignette_pair,
                    selected_feature_vectors
                )

                # Combined score
                score = (1 - diversity_weight) * informativeness + diversity_weight * diversity

                if score > best_score:
                    best_score = score
                    best_vignette = vignette_pair

            if best_vignette is None:
                self.logger.warning(
                    f"Could not find vignette for round {round_idx + 1}, stopping early"
                )
                break

            # Add to selected
            selected_vignettes.append(best_vignette)

            # Track hash to avoid re-selecting
            best_hash = (
                self._profile_hash(best_vignette[0]),
                self._profile_hash(best_vignette[1])
            )
            selected_hashes.add(best_hash)

            # Track feature representation for diversity
            x_a = np.array(self.profile_generator.encode_profile_terms(best_vignette[0]))
            x_b = np.array(self.profile_generator.encode_profile_terms(best_vignette[1]))
            x_diff = x_a - x_b
            selected_feature_vectors.append(x_diff)

        self.logger.info(
            f"Built adaptive library with {len(selected_vignettes)} vignettes"
        )

        return selected_vignettes

    def _compute_informativeness(
        self,
        profile_a: Dict[str, Any],
        profile_b: Dict[str, Any],
        preference_weights: np.ndarray,
        temperature: float = 1.0
    ) -> float:
        """
        Compute informativeness score for a vignette.

        Uses determinant of FIM as measure of information.

        Args:
            profile_a: First job profile
            profile_b: Second job profile
            preference_weights: Current preference weights
            temperature: Temperature parameter

        Returns:
            Informativeness score (FIM determinant)
        """
        # Encode profiles
        x_a = np.array(self.profile_generator.encode_profile_terms(profile_a))
        x_b = np.array(self.profile_generator.encode_profile_terms(profile_b))

        # Compute utilities
        u_a = np.dot(x_a, preference_weights) / temperature
        u_b = np.dot(x_b, preference_weights) / temperature

        # Compute probabilities
        max_u = max(u_a, u_b)
        exp_u_a = np.exp(u_a - max_u)
        exp_u_b = np.exp(u_b - max_u)
        p_a = exp_u_a / (exp_u_a + exp_u_b)
        p_b = 1 - p_a

        # Fisher Information
        x_diff = x_a - x_b
        fim = p_a * p_b * np.outer(x_diff, x_diff)

        # Determinant as informativeness
        det = np.linalg.det(fim + np.eye(self.n_params) * 1e-8)

        return float(det)

    def _compute_diversity(
        self,
        vignette_pair: Tuple[Dict, Dict],
        selected_feature_vectors: List[np.ndarray]
    ) -> float:
        """
        Compute diversity score for a vignette.

        Measures minimum distance to already-selected vignettes.

        Args:
            vignette_pair: Candidate vignette
            selected_feature_vectors: Feature vectors of selected vignettes

        Returns:
            Diversity score (higher = more diverse)
        """
        if not selected_feature_vectors:
            return 1.0  # First vignette is maximally diverse

        # Get feature representation
        x_a = np.array(self.profile_generator.encode_profile_terms(vignette_pair[0]))
        x_b = np.array(self.profile_generator.encode_profile_terms(vignette_pair[1]))
        x_diff = x_a - x_b

        # Compute minimum distance to selected vignettes
        min_distance = float('inf')
        for selected_x_diff in selected_feature_vectors:
            # Cosine distance
            cos_sim = np.dot(x_diff, selected_x_diff) / (
                np.linalg.norm(x_diff) * np.linalg.norm(selected_x_diff) + 1e-8
            )
            distance = 1 - abs(cos_sim)  # 0 = identical, 1 = orthogonal
            min_distance = min(min_distance, distance)

        return min_distance

    def _find_focused_pairs_for_attr(
        self,
        profiles: List[Dict[str, Any]],
        attr_name: str
    ) -> List[Tuple[Dict, Dict]]:
        """
        Find all profile pairs that differ ONLY in attr_name.

        Groups profiles by all-other-attribute values; any two profiles in the
        same group differ in exactly attr_name and nothing else.

        Args:
            profiles: Full profile list
            attr_name: The single attribute that may differ

        Returns:
            List of (profile_a, profile_b) pairs, unfiltered
        """
        groups: Dict[tuple, List[Dict]] = defaultdict(list)
        for profile in profiles:
            key = tuple(sorted(
                (k, v) for k, v in profile.items() if k != attr_name
            ))
            groups[key].append(profile)

        focused_pairs = []
        for group_profiles in groups.values():
            if len(group_profiles) >= 2:
                for p_a, p_b in combinations(group_profiles, 2):
                    focused_pairs.append((p_a, p_b))

        return focused_pairs

    def _has_attribute_cancellation(
        self,
        profile_a: Dict[str, Any],
        profile_b: Dict[str, Any]
    ) -> bool:
        """
        Check if vignette has attribute cancellation that makes it uninformative.

        Uses encoded feature vectors (via SchemaLoader) to detect cancellation:
        if after encoding, a dimension difference is near zero because multiple
        attributes within that dimension cancel out, the vignette is uninformative.

        Args:
            profile_a: First profile
            profile_b: Second profile

        Returns:
            True if vignette has significant cancellation (uninformative)
        """
        schema_loader = self.profile_generator.schema_loader

        # Group attributes by dimension for cancellation detection
        from collections import defaultdict
        dim_to_attrs = defaultdict(list)
        for spec in schema_loader._specs:
            dim_to_attrs[spec.dimension].append(spec)

        for dim, specs in dim_to_attrs.items():
            if len(specs) < 2:
                continue  # No cancellation possible with a single attribute

            # Compute per-attribute encoded contributions
            diffs = []
            for spec in specs:
                val_a = spec.encode(profile_a.get(spec.name))
                val_b = spec.encode(profile_b.get(spec.name))
                diffs.append(val_a - val_b)

            # Check for cancellation: mixed signs + near-zero net effect
            signs = [np.sign(d) for d in diffs if abs(d) > 0.01]
            if len(signs) >= 2:
                has_positive = any(s > 0 for s in signs)
                has_negative = any(s < 0 for s in signs)
                if has_positive and has_negative:
                    avg_diff = np.mean(diffs)
                    if abs(avg_diff) < 0.15:
                        return True

        return False

    def _profile_hash(self, profile: Dict[str, Any]) -> str:
        """
        Create hash for profile for deduplication.

        Args:
            profile: Job profile dictionary

        Returns:
            Hash string
        """
        # Sort keys for consistent hashing
        items = sorted(profile.items())
        return str(items)

    def get_library_statistics(
        self,
        library: List[Tuple[Dict, Dict]]
    ) -> Dict[str, Any]:
        """
        Get statistics about the adaptive library.

        Args:
            library: List of vignette pairs

        Returns:
            Dictionary with library statistics
        """
        if not library:
            return {"num_vignettes": 0}

        # Extract feature vectors
        feature_vectors = []
        for profile_a, profile_b in library:
            x_a = np.array(self.profile_generator.encode_profile_terms(profile_a))
            x_b = np.array(self.profile_generator.encode_profile_terms(profile_b))
            x_diff = x_a - x_b
            feature_vectors.append(x_diff)

        feature_matrix = np.array(feature_vectors)

        # Compute diversity metrics
        pairwise_distances = []
        for i in range(len(feature_vectors)):
            for j in range(i + 1, len(feature_vectors)):
                cos_sim = np.dot(feature_vectors[i], feature_vectors[j]) / (
                    np.linalg.norm(feature_vectors[i]) * np.linalg.norm(feature_vectors[j]) + 1e-8
                )
                distance = 1 - abs(cos_sim)
                pairwise_distances.append(distance)

        # Coverage of attribute space
        attribute_coverage = self._compute_attribute_coverage(library)

        return {
            "num_vignettes": len(library),
            "avg_pairwise_distance": float(np.mean(pairwise_distances)),
            "min_pairwise_distance": float(np.min(pairwise_distances)),
            "max_pairwise_distance": float(np.max(pairwise_distances)),
            "std_pairwise_distance": float(np.std(pairwise_distances)),
            "attribute_coverage": attribute_coverage
        }

    def _compute_attribute_coverage(
        self,
        library: List[Tuple[Dict, Dict]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Compute how well the library covers different attribute values.

        Args:
            library: List of vignette pairs

        Returns:
            Dictionary mapping attribute name to value counts
        """
        coverage = {}

        # Get attribute names from first profile
        if not library:
            return coverage

        first_profile = library[0][0]
        attr_names = list(first_profile.keys())

        # Count occurrences of each attribute value
        for attr_name in attr_names:
            value_counts = {}
            for profile_a, profile_b in library:
                for profile in [profile_a, profile_b]:
                    value = profile[attr_name]
                    value_counts[value] = value_counts.get(value, 0) + 1
            coverage[attr_name] = value_counts

        return coverage

    def _has_excessive_wage_gap(
        self,
        profile_a: Dict[str, Any],
        profile_b: Dict[str, Any],
        max_ratio: float = 1.5
    ) -> bool:
        """
        Check if wage gap between profiles is too large (psychological anchoring issue).

        Research shows that when salary differences exceed ~60-70%, people tend to
        anchor heavily on the financial dimension and ignore other trade-offs.

        Args:
            profile_a: First profile
            profile_b: Second profile
            max_ratio: Maximum acceptable wage ratio (default: 1.67, i.e., 67% difference)

        Returns:
            True if wage gap is excessive (bad vignette - financial anchoring)
            False if wage gap is reasonable (good vignette)
        """
        wage_a = profile_a.get('earnings_per_month') or profile_a.get('wage', 0)
        wage_b = profile_b.get('earnings_per_month') or profile_b.get('wage', 0)

        if wage_a == 0 or wage_b == 0:
            return False  # No wage info, can't check

        # Calculate ratio (higher / lower)
        ratio = max(wage_a, wage_b) / min(wage_a, wage_b)

        return ratio > max_ratio

    def _has_pairwise_dominance(
        self,
        profile_a: Dict[str, Any],
        profile_b: Dict[str, Any],
        quasi_dominance_threshold: int = 4
    ) -> bool:
        """
        Check if one profile dominates the other in this vignette pair.

        Checks for both strict dominance and quasi-dominance in the term-level
        encoded preference space.

        Args:
            profile_a: First profile in vignette
            profile_b: Second profile in vignette
            quasi_dominance_threshold: Minimum terms where one option must be better
                                       to be considered quasi-dominant (default: 4/6 = 67%)

        Returns:
            True if either profile dominates or quasi-dominates the other
            False if neither dominates (good vignette - has meaningful trade-offs)
        """
        # Encode profiles to term-level preference space
        features_a = np.array(self.profile_generator.encode_profile_terms(profile_a))
        features_b = np.array(self.profile_generator.encode_profile_terms(profile_b))

        # Check strict dominance
        a_dominates_b = self._features_dominate(features_a, features_b)
        b_dominates_a = self._features_dominate(features_b, features_a)

        if a_dominates_b or b_dominates_a:
            return True

        # Check quasi-dominance: count dimensions where each option is better
        tolerance = 1e-6
        a_better_count = sum(1 for i in range(len(features_a))
                           if features_a[i] - features_b[i] > tolerance)
        b_better_count = sum(1 for i in range(len(features_b))
                           if features_b[i] - features_a[i] > tolerance)

        # If either option is better in ≥ threshold dimensions, it's quasi-dominant
        if a_better_count >= quasi_dominance_threshold or b_better_count >= quasi_dominance_threshold:
            return True

        return False

    def _features_dominate(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        tolerance: float = 1e-6
    ) -> bool:
        """
        Check if features_a dominates features_b in the 7-dimensional preference space.

        Features A dominates Features B if:
        - A is better than or equal to B in ALL 7 preference dimensions
        - A is strictly better than B in AT LEAST ONE dimension

        Args:
            features_a: First feature vector (7D)
            features_b: Second feature vector (7D)
            tolerance: Numerical tolerance for "equal" comparison (default: 1e-6)

        Returns:
            True if features_a dominates features_b
        """
        better_or_equal_in_all = True
        strictly_better_in_at_least_one = False

        for i in range(len(features_a)):
            diff = features_a[i] - features_b[i]

            # Check if A is worse than B in this dimension
            if diff < -tolerance:
                better_or_equal_in_all = False
                break

            # Check if A is strictly better than B in this dimension
            if diff > tolerance:
                strictly_better_in_at_least_one = True

        return better_or_equal_in_all and strictly_better_in_at_least_one
