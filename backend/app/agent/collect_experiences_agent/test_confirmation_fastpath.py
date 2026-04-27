"""
Tests for the confirmation fast-path in CollectExperiencesAgent.execute().

The fast-path skips data extraction entirely on bare yes/no/ok-style replies,
saving ~750ms on the most common turn type. The conversation LLM and transition
tool still run normally.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    _is_simple_confirmation,
)
from app.agent.experience.work_type import WorkType
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


@pytest.fixture
def patched_llms():
    """Patch all three LLM collaborators instantiated inside execute()."""
    with patch(
        "app.agent.collect_experiences_agent.collect_experiences_agent._DataExtractionLLM"
    ) as data_mock, patch(
        "app.agent.collect_experiences_agent.collect_experiences_agent._ConversationLLM"
    ) as conv_mock, patch(
        "app.agent.collect_experiences_agent.collect_experiences_agent.TransitionDecisionTool"
    ) as trans_mock:
        data_instance = MagicMock()
        data_instance.execute = AsyncMock(return_value=(-1, [], []))
        data_mock.return_value = data_instance

        conv_instance = MagicMock()
        conv_instance.execute = AsyncMock(
            return_value=ConversationLLMAgentOutput(
                message_for_user="ok",
                exploring_type_finished=False,
                finished=False,
                agent_type=AgentType.COLLECT_EXPERIENCES_AGENT,
                agent_response_time_in_sec=0.1,
                llm_stats=[],
            )
        )
        conv_mock.return_value = conv_instance

        trans_instance = MagicMock()
        trans_instance.execute = AsyncMock(
            return_value=(
                TransitionDecision.CONTINUE,
                TransitionReasoning(reasoning="test", confidence="medium"),
                [],
            )
        )
        trans_mock.return_value = trans_instance

        yield data_mock, conv_mock, trans_mock


def _empty_context() -> ConversationContext:
    empty = ConversationHistory(turns=[])
    return ConversationContext(all_history=empty, history=empty, summary="")


def _state_with_one_titled_experience() -> CollectExperiencesAgentState:
    return CollectExperiencesAgentState(
        session_id=1,
        country_of_user=Country.KENYA,
        education_phase_done=True,
        first_time_visit=False,
        collected_data=[
            CollectedData(
                index=0,
                experience_title="Cashier",
                company="Carrefour",
                location="Nairobi",
                start_date="2020",
                end_date="2022",
                paid_work=True,
                work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT.name,
                defined_at_turn_number=2,
            )
        ],
        unexplored_types=[
            WorkType.SELF_EMPLOYMENT,
            WorkType.FORMAL_SECTOR_UNPAID_TRAINEE_WORK,
            WorkType.UNSEEN_UNPAID,
        ],
        explored_types=[WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT],
    )


@pytest.mark.parametrize(
    "message,expected",
    [
        ("yes", True),
        ("Yes", True),
        ("YES", True),
        ("yes.", True),
        ("yes!", True),
        ("  yes  ", True),
        ("no", True),
        ("ok", True),
        ("Sawa", True),
        ("ndio", True),
        ("hapana", True),
        ("yes I worked there", False),
        ("ok let me think", False),
        ("I worked at Carrefour", False),
        ("", False),
        ("yesterday", False),
    ],
)
def test_is_simple_confirmation(message, expected):
    assert _is_simple_confirmation(message) is expected


@pytest.mark.asyncio
async def test_fastpath_skips_data_extraction_on_confirmation(patched_llms):
    """A bare confirmation must not invoke the data extraction LLM."""
    data_mock, conv_mock, _ = patched_llms

    agent = CollectExperiencesAgent()
    agent.set_state(_state_with_one_titled_experience())

    await agent.execute(
        user_input=AgentInput(message="yes"),
        context=_empty_context(),
    )

    data_mock.return_value.execute.assert_not_called()
    # Conversation LLM still runs — phase transitions, follow-ups still work.
    conv_mock.return_value.execute.assert_called_once()


@pytest.mark.asyncio
async def test_fastpath_does_not_trigger_for_substantive_input(patched_llms):
    """A normal user reply must still go through full data extraction."""
    data_mock, _, _ = patched_llms

    agent = CollectExperiencesAgent()
    agent.set_state(_state_with_one_titled_experience())

    await agent.execute(
        user_input=AgentInput(message="I worked at Safaricom for 3 years"),
        context=_empty_context(),
    )

    data_mock.return_value.execute.assert_called_once()


@pytest.mark.asyncio
async def test_fastpath_preserves_last_referenced_experience_index(patched_llms):
    """The fast-path should reuse the previously stored index instead of resetting it.

    This ensures the conversation LLM keeps context (knows which experience the user
    is currently discussing) on confirmation turns.
    """
    data_mock, conv_mock, _ = patched_llms
    # First turn: full extraction returns a real index (the user provided new info).
    data_mock.return_value.execute = AsyncMock(return_value=(0, [], []))

    agent = CollectExperiencesAgent()
    agent.set_state(_state_with_one_titled_experience())

    # Substantive input → full extraction; index stored on instance.
    await agent.execute(
        user_input=AgentInput(message="I was a cashier"),
        context=_empty_context(),
    )
    assert agent._last_referenced_experience_index == 0

    # Now the fast-path turn: index must be preserved.
    await agent.execute(
        user_input=AgentInput(message="yes"),
        context=_empty_context(),
    )
    second_call_kwargs = conv_mock.return_value.execute.call_args.kwargs
    assert second_call_kwargs["last_referenced_experience_index"] == 0


@pytest.mark.asyncio
async def test_set_state_resets_last_referenced_index(patched_llms):
    """A new session must not inherit the prior session's index."""
    agent = CollectExperiencesAgent()
    agent.set_state(_state_with_one_titled_experience())
    agent._last_referenced_experience_index = 5

    agent.set_state(_state_with_one_titled_experience())
    assert agent._last_referenced_experience_index == -1
