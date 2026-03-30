import pytest
from app.agent.skill_explorer_agent._conversation_llm import _ConversationLLM
from app.agent.experience.work_type import WorkType
from app.countries import Country
from app.agent.persona_detector import PersonaType
from app.i18n.translation_service import get_i18n_manager
from app.i18n.types import Locale


@pytest.fixture(autouse=True)
def set_locale():
    get_i18n_manager().set_locale(Locale.EN_US)


def _base_kwargs(**overrides):
    kwargs = dict(
        country_of_user=Country.UNSPECIFIED,
        persona_type=None,
        experiences_explored=[],
        experience_title="Software Engineer",
        experience_index=0,
        rich_response=False,
        work_type=None,
        cv_responsibilities=[],
    )
    kwargs.update(overrides)
    return kwargs


def test_prompt_without_cv_responsibilities_asks_typical_day():
    prompt = _ConversationLLM.create_first_time_generative_prompt(**_base_kwargs(cv_responsibilities=[]))
    assert "describe a typical day" in prompt.lower()
    assert "cv" not in prompt.lower() or "context" not in prompt.lower()


def test_prompt_with_cv_responsibilities_includes_bullets():
    prompt = _ConversationLLM.create_first_time_generative_prompt(
        **_base_kwargs(cv_responsibilities=["Built REST APIs", "Led code reviews"])
    )
    assert "Built REST APIs" in prompt
    assert "Led code reviews" in prompt


def test_prompt_with_cv_responsibilities_does_not_ask_typical_day():
    prompt = _ConversationLLM.create_first_time_generative_prompt(
        **_base_kwargs(cv_responsibilities=["Administered medication daily to 30+ patients",
                                            "Supervised two junior nurses on shift"])
    )
    # The fallback "describe a typical day" instruction should not appear
    assert "describe a typical day" not in prompt.lower()


def test_prompt_with_cv_responsibilities_includes_sufficiency_instruction():
    prompt = _ConversationLLM.create_first_time_generative_prompt(
        **_base_kwargs(cv_responsibilities=["Developed microservices in Python"])
    )
    assert "sufficient" in prompt.lower()


def test_prompt_with_cv_responsibilities_includes_confirm_instruction():
    prompt = _ConversationLLM.create_first_time_generative_prompt(
        **_base_kwargs(cv_responsibilities=["Managed budget of KES 2M annually"])
    )
    assert "confirm" in prompt.lower()


def test_prompt_with_none_cv_responsibilities_asks_typical_day():
    """None should behave identically to empty list — no CV block injected."""
    prompt = _ConversationLLM.create_first_time_generative_prompt(**_base_kwargs(cv_responsibilities=None))
    assert "describe a typical day" in prompt.lower()
