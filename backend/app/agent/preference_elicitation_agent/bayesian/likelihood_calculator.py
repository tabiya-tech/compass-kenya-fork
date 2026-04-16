"""
Likelihood calculation for preference elicitation using Multinomial Logit (MNL) model.

Computes P(choice|preferences, vignette) using standard choice probability formulation.
Feature extraction is driven entirely by a SchemaLoader so no attribute names or
dimension indices are hardcoded here.
"""

from typing import Dict, Callable
import numpy as np
from ..types import Vignette, VignetteOption
from .schema_loader import SchemaLoader


class LikelihoodCalculator:
    """Compute likelihood of observed choices under preference model."""

    def __init__(self, schema_loader: SchemaLoader, temperature: float = 1.0):
        """
        Initialize likelihood calculator.

        Args:
            schema_loader: Loaded SchemaLoader instance that drives feature extraction.
            temperature: Controls choice stochasticity
                - temperature=1.0: standard MNL
                - temperature>1.0: more random
                - temperature<1.0: more deterministic
        """
        self._schema = schema_loader
        self.temperature = temperature

    def compute_choice_likelihood(
        self,
        vignette: Vignette,
        chosen_option: str,  # "A" or "B"
        preference_weights: np.ndarray
    ) -> float:
        """
        Compute P(chose option | preferences, vignette).

        Uses Multinomial Logit (MNL) model:
        P(A|β) = exp(β·x_A) / [exp(β·x_A) + exp(β·x_B)]

        Args:
            vignette: The vignette shown
            chosen_option: Which option user chose ("A" or "B")
            preference_weights: β vector (N dimensions, matching schema)

        Returns:
            Likelihood (probability between 0 and 1)
        """
        option_a, option_b = self._resolve_options(vignette)
        x_A = self._extract_features(option_a)
        x_B = self._extract_features(option_b)

        # Compute utilities
        u_A = np.dot(x_A, preference_weights) / self.temperature
        u_B = np.dot(x_B, preference_weights) / self.temperature

        # Choice probabilities (softmax with log-sum-exp for numerical stability)
        max_u = max(u_A, u_B)
        exp_u_A = np.exp(u_A - max_u)
        exp_u_B = np.exp(u_B - max_u)

        p_A = exp_u_A / (exp_u_A + exp_u_B)
        p_B = 1 - p_A

        return float(p_A) if chosen_option == "A" else float(p_B)

    def _resolve_options(
        self, vignette: Vignette
    ) -> tuple[VignetteOption, VignetteOption]:
        """Return (option_A, option_B) from a vignette regardless of format."""
        # New format: options list
        option_a = next((o for o in vignette.options if o.option_id == "A"), None)
        option_b = next((o for o in vignette.options if o.option_id == "B"), None)

        if option_a is None or option_b is None:
            if len(vignette.options) < 2:
                raise ValueError(
                    f"Vignette {vignette.vignette_id} needs at least 2 options, "
                    f"got {len(vignette.options)}"
                )
            option_a, option_b = vignette.options[0], vignette.options[1]

        return option_a, option_b

    def _extract_features(self, option: VignetteOption) -> np.ndarray:
        """
        Extract feature vector from a vignette option using the schema.

        Returns:
            Feature vector of length schema_loader.n_dimensions
        """
        attrs = option.attributes if option.attributes else {}
        features = self._schema.extract_features(attrs)
        return np.array(features, dtype=float)

    def create_likelihood_function(
        self,
        vignette: Vignette,
        chosen_option: str
    ) -> Callable[[Dict, np.ndarray], float]:
        """
        Create likelihood function that can be passed to PosteriorManager.

        Returns:
            Function signature: likelihood(observation, beta) -> float
        """
        def likelihood_fn(observation: Dict, beta: np.ndarray) -> float:
            return self.compute_choice_likelihood(
                vignette=observation["vignette"],
                chosen_option=observation["chosen_option"],
                preference_weights=beta
            )

        return likelihood_fn
