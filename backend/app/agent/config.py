from enum import StrEnum
from typing import Literal, Final

Model = Literal[
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


class ModelTier(StrEnum):
    """
    The capability tier an LLM call needs. The concrete model for each tier is resolved at call time
    from the LLM_DEFAULT_MODEL, LLM_REASONING_MODEL and LLM_DEEP_REASONING_MODEL environment variables,
    see `common_libs.llm.models_utils.resolve_model_name`.
    """

    DEFAULT = "default"
    """
    The model to use by default.
    Used by conversation-facing agents that need fast response times.
    Swahili support is handled via mapping/normalization layer instead of model upgrade.
    """

    REASONING = "reasoning"
    """
    The model to use for tasks that need good reasoning.

    Expectations
    - Good reasoning
    - Slow in response time compared to the default model
    """

    DEEP_REASONING = "deep_reasoning"
    """
    The model to use for tasks that need the strongest reasoning,
    for specific cases like evaluations that don't run at run time.
    """


GEMINI_DEFAULT_MODELS: Final[dict[ModelTier, Model]] = {
    ModelTier.DEFAULT: "gemini-2.5-flash-lite",
    ModelTier.REASONING: "gemini-2.5-flash",
    ModelTier.DEEP_REASONING: "gemini-2.5-pro",
}
"""
The Gemini model used for each tier when the tier's environment variable is not set.
"""
