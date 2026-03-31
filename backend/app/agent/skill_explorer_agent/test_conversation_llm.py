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
        source=None,
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


def test_prompt_with_education_source_asks_about_applied_skills():
    prompt = _ConversationLLM.create_first_time_generative_prompt(
        **_base_kwargs(source="education", experience_title="BSc Computer Science")
    )
    assert "learned" in prompt.lower() or "able to" in prompt.lower() or "course" in prompt.lower()


def test_prompt_with_education_source_does_not_use_default_typical_day():
    """Education prompt should not use the default 'describe a typical day as X' instruction."""
    prompt = _ConversationLLM.create_first_time_generative_prompt(
        **_base_kwargs(source="education")
    )
    # The default instruction is "Ask me to describe a typical day as..." — this should NOT appear
    assert "ask me to describe a typical day" not in prompt.lower()
    # Education-specific instructions should appear
    assert "post-secondary education programme" in prompt.lower()


def test_prompt_with_none_source_asks_typical_day():
    """None source (default) should still ask typical day."""
    prompt = _ConversationLLM.create_first_time_generative_prompt(
        **_base_kwargs(source=None)
    )
    assert "describe a typical day" in prompt.lower()


def test_prompt_with_cv_source_still_works():
    """CV source with responsibilities should still show CV bullets."""
    prompt = _ConversationLLM.create_first_time_generative_prompt(
        **_base_kwargs(source="cv", cv_responsibilities=["Built REST APIs"])
    )
    assert "Built REST APIs" in prompt


def test_system_instructions_with_education_source():
    instructions = _ConversationLLM._create_conversation_system_instructions(
        question_asked_until_now=[],
        country_of_user=Country.UNSPECIFIED,
        persona_type=None,
        experience_title="BSc Computer Science",
        experience_index=0,
        rich_response=False,
        work_type=None,
        source="education",
    )
    assert "learned" in instructions.lower() or "course" in instructions.lower() or "studies" in instructions.lower()


def test_get_question_c_education():
    from app.agent.skill_explorer_agent._conversation_llm import _get_question_c
    result = _get_question_c(work_type=None, source="education")
    assert "studies" in result.lower() or "confident" in result.lower()


def test_get_question_c_none_source_unchanged():
    from app.agent.skill_explorer_agent._conversation_llm import _get_question_c
    result = _get_question_c(work_type=None, source=None)
    assert result == ""
