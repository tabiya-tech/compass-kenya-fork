import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.agent_types import AgentInput, AgentType
from app.agent.collect_experiences_agent._conversation_llm import ConversationLLMAgentOutput
from app.agent.collect_experiences_agent._transition_decision_tool import (
    TransitionDecision,
    TransitionReasoning,
)
from app.agent.collect_experiences_agent._types import CollectedData
from app.agent.collect_experiences_agent.collect_experiences_agent import (
    CollectExperiencesAgent,
    CollectExperiencesAgentState,
)
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
)
from app.countries import Country
from app.i18n.translation_service import get_i18n_manager
from app.i18n.types import Locale


@pytest.fixture(autouse=True)
def set_locale():
    get_i18n_manager().set_locale(Locale.EN_US)


def _state_with_pending_education() -> CollectExperiencesAgentState:
    """A realistic state after one education entry, user about to answer the 'any more?' question."""
    return CollectExperiencesAgentState(
        session_id=1,
        country_of_user=Country.UNSPECIFIED,
        collected_data=[
            CollectedData(
                index=0,
                experience_title="Cooking course",
                company="Kitui Master Cook",
                location=None,
                start_date="2019",
                end_date="2020",
                paid_work=False,
                work_type=None,
                source="education",
            )
        ],
        first_time_visit=False,
        education_phase_done=False,
    )


def _empty_context() -> ConversationContext:
    empty_history = ConversationHistory(turns=[])
    return ConversationContext(
        all_history=empty_history,
        history=empty_history,
        summary="",
    )


@pytest.fixture
def patched_llms():
    """Patch the three LLM collaborators instantiated inside CollectExperiencesAgent.execute()."""
    with patch(
        "app.agent.collect_experiences_agent.collect_experiences_agent._DataExtractionLLM"
    ) as data_mock, patch(
        "app.agent.collect_experiences_agent.collect_experiences_agent._ConversationLLM"
    ) as conv_mock, patch(
        "app.agent.collect_experiences_agent.collect_experiences_agent.TransitionDecisionTool"
    ) as trans_mock:
        data_instance = MagicMock()
        data_instance.execute = AsyncMock(return_value=(-1, []))
        data_mock.return_value = data_instance
        yield conv_mock, trans_mock


def _wire_conversation_llm(conv_mock, *, message: str, exploring_type_finished: bool) -> None:
    instance = MagicMock()
    instance.execute = AsyncMock(
        return_value=ConversationLLMAgentOutput(
            message_for_user=message,
            exploring_type_finished=exploring_type_finished,
            finished=False,
            agent_type=AgentType.COLLECT_EXPERIENCES_AGENT,
            agent_response_time_in_sec=0.1,
            llm_stats=[],
        )
    )
    conv_mock.return_value = instance


def _wire_transition_tool(trans_mock, *, decision: TransitionDecision) -> None:
    instance = MagicMock()
    instance.execute = AsyncMock(
        return_value=(
            decision,
            TransitionReasoning(reasoning="test", confidence="medium"),
            [],
        )
    )
    trans_mock.return_value = instance


@pytest.mark.asyncio
async def test_education_phase_flips_done_when_conversation_llm_signals_end_even_if_transition_says_continue(
    patched_llms,
):
    """
    Regression test for the education→work stuck-loop bug.

    When the user says "No" to further post-secondary education, the education
    conversation LLM emits <END_OF_WORKTYPE>, which _ConversationLLM translates
    to "Let's move on to other work experiences." and sets
    exploring_type_finished=True.

    The transition_decision_tool is unaware of the education phase and evaluates
    transitions against FORMAL_SECTOR_WAGED_EMPLOYMENT (the first unexplored
    work type). Because the user has not been asked about waged work yet, it
    returns CONTINUE.

    Before the fix, this combination left education_phase_done=False and the
    agent looped "Let's move on to other work experiences." indefinitely. After
    the fix, exploring_type_finished alone is enough to flip the phase.
    """
    conv_mock, trans_mock = patched_llms
    _wire_conversation_llm(
        conv_mock,
        message="Let's move on to other work experiences.",
        exploring_type_finished=True,
    )
    _wire_transition_tool(trans_mock, decision=TransitionDecision.CONTINUE)

    agent = CollectExperiencesAgent()
    agent.set_state(_state_with_pending_education())

    output = await agent.execute(
        user_input=AgentInput(message="No"),
        context=_empty_context(),
    )

    assert agent._state.education_phase_done is True, (
        "Education phase must flip to done when the conversation LLM signals "
        "exploring_type_finished, even if the transition decision tool returns CONTINUE."
    )
    lowered = output.message_for_user.lower()
    assert "education" in lowered and "work experiences" in lowered, (
        "The education→work transition text must be appended so the user sees the "
        "full handover, not just 'Let's move on to other work experiences.'"
    )


@pytest.mark.asyncio
async def test_education_phase_stays_open_when_neither_signal_fires(patched_llms):
    """
    Partial education answer — agent is still gathering. Neither
    exploring_type_finished nor END_WORKTYPE fire, so education_phase_done must
    stay False and no transition text is appended.
    """
    conv_mock, trans_mock = patched_llms
    _wire_conversation_llm(
        conv_mock,
        message="Got it. When did you start the course?",
        exploring_type_finished=False,
    )
    _wire_transition_tool(trans_mock, decision=TransitionDecision.CONTINUE)

    agent = CollectExperiencesAgent()
    agent.set_state(_state_with_pending_education())

    output = await agent.execute(
        user_input=AgentInput(message="Cooking course at Kitui Master Cook"),
        context=_empty_context(),
    )

    assert agent._state.education_phase_done is False
    assert "Thanks for sharing your education" not in output.message_for_user


@pytest.mark.asyncio
async def test_education_phase_does_not_flip_on_transition_tool_end_alone(patched_llms):
    """
    Regression test for the premature-transition bug.

    TransitionDecisionTool evaluates against FORMAL_SECTOR_WAGED_EMPLOYMENT (the first
    unexplored work type), not education. It can return END_WORKTYPE or END_CONVERSATION
    after a single partial education entry while the conversation LLM is still gathering
    fields (e.g. asking "Which institution did you attend?"). In that case the LLM has
    NOT emitted <END_OF_WORKTYPE>, so exploring_type_finished is False.

    The education phase must stay open — no transition text should be appended and
    education_phase_done must remain False. Only the LLM's explicit completion signal
    may end the phase.
    """
    conv_mock, trans_mock = patched_llms
    _wire_conversation_llm(
        conv_mock,
        message="Which institution did you attend for this engineering program?",
        exploring_type_finished=False,
    )

    agent = CollectExperiencesAgent()
    agent.set_state(_state_with_pending_education())

    # Both end-verdicts from the transition tool must be ignored in education phase.
    for decision in (TransitionDecision.END_WORKTYPE, TransitionDecision.END_CONVERSATION):
        _wire_transition_tool(trans_mock, decision=decision)
        agent._state.education_phase_done = False

        output = await agent.execute(
            user_input=AgentInput(message="engineering program"),
            context=_empty_context(),
        )

        assert agent._state.education_phase_done is False, (
            f"Education phase must NOT end on transition_decision={decision.value} alone "
            "when the conversation LLM has not emitted <END_OF_WORKTYPE>."
        )
        assert "work experiences" not in output.message_for_user.lower(), (
            "Transition text must not be appended while the LLM is still collecting "
            "education fields."
        )
