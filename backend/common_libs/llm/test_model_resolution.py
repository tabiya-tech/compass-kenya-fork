import pytest

from app.agent.config import ModelTier
from common_libs.llm.models_utils import resolve_model_name


class TestResolveModelName:
    @pytest.mark.parametrize("given_tier, expected_model", [
        (ModelTier.DEFAULT, "gemini-3.5-flash-lite"),
        (ModelTier.REASONING, "gemini-3.8-flash"),
        (ModelTier.DEEP_REASONING, "gemini-3.1-pro-preview"),
    ])
    def test_uses_gemini_defaults_when_no_env_var_is_set(self, monkeypatch, given_tier: ModelTier, expected_model: str):
        # GIVEN no tier environment variable is set
        monkeypatch.delenv("LLM_DEFAULT_MODEL", raising=False)
        monkeypatch.delenv("LLM_REASONING_MODEL", raising=False)
        monkeypatch.delenv("LLM_DEEP_REASONING_MODEL", raising=False)

        # WHEN the model name for the given tier is resolved
        actual_model = resolve_model_name(tier=given_tier)

        # THEN expect the gemini default for the tier
        assert actual_model == expected_model

    @pytest.mark.parametrize("given_tier, given_env_var", [
        (ModelTier.DEFAULT, "LLM_DEFAULT_MODEL"),
        (ModelTier.REASONING, "LLM_REASONING_MODEL"),
        (ModelTier.DEEP_REASONING, "LLM_DEEP_REASONING_MODEL"),
    ])
    def test_uses_the_tier_env_var_when_it_is_set(self, monkeypatch, given_tier: ModelTier, given_env_var: str):
        # GIVEN the tier's environment variable is set
        monkeypatch.setenv(given_env_var, "given-model")

        # WHEN the model name for the given tier is resolved
        actual_model = resolve_model_name(tier=given_tier)

        # THEN expect the model from the environment variable
        assert actual_model == "given-model"

    def test_does_not_fall_back_to_another_tiers_env_var(self, monkeypatch):
        # GIVEN only the default tier's environment variable is set
        monkeypatch.setenv("LLM_DEFAULT_MODEL", "given-default")
        monkeypatch.delenv("LLM_REASONING_MODEL", raising=False)
        monkeypatch.delenv("LLM_DEEP_REASONING_MODEL", raising=False)

        # WHEN the model name for the reasoning tier is resolved
        actual_model = resolve_model_name(tier=ModelTier.REASONING)

        # THEN expect the gemini default for the reasoning tier, not the default tier's model
        assert actual_model == "gemini-3.8-flash"
