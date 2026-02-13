from typing import Literal, final, Final

Model = Literal[
    # gemini-1.5-flash is an auto update version the points to the most recent stable version
    # see https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning#auto-updated-version
    # "gemini-1.5-flash-001",
    # "gemini-2.0-flash-001",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


@final
class AgentsConfig:
    default_model: Final[Model] = "gemini-2.5-flash"
    """
    The LLM model name to use by default.
    Upgraded from gemini-2.5-flash-lite for better Swahili language support (M3).
    """

    fast_model: Final[Model] = "gemini-2.5-flash"
    """
    The fast LLM model name to use.
    
    Expectations
    - Good reasoning with strong multilingual (Swahili) support
    - Fast in response time
    """

    deep_reasoning_model: Final[Model] = "gemini-2.5-flash"
    """
    The LLM model name to use for deep reasoning.
    
    Expectations
    - High reasoning
    - Slow in response time compared to the fast model
    """

    ultra_high_reasoning_model: Final[Model] = "gemini-2.5-pro"
    """
    Ultra high reasoning models to be used. 
    for specific cases like evaluations that don't run at run time.
    """
