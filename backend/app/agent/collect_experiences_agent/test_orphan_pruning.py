import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.agent_types import AgentInput, AgentOutput, AgentType
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
from app.agent.experience.work_type import WorkType
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
    ConversationTurn,
)
from app.countries import Country
from app.i18n.translation_service import get_i18n_manager
from app.i18n.types import Locale


@pytest.fixture(autouse=True)
def set_locale():
    get_i18n_manager().set_locale(Locale.EN_US)


def _titled(*, title: str, work_type: WorkType, defined_at_turn: int, index: int = 0) -> CollectedData:
    return CollectedData(
        index=index,
        experience_title=title,
        company="Some company",
        location="Some place",
        start_date="2020",
        end_date="2021",
        paid_work=True,
        work_type=work_type.name,
        defined_at_turn_number=defined_at_turn,
    )


def _orphan(*, work_type: WorkType, defined_at_turn: int, index: int = 0) -> CollectedData:
    """Titleless experience — a data-extraction phantom."""
    return CollectedData(
        index=index,
        experience_title=None,
        company=None,
        location=None,
        start_date="2021",
        end_date="2021",
        paid_work=None,
        work_type=work_type.name,
        defined_at_turn_number=defined_at_turn,
    )


def _context_with_turn_count(count: int) -> ConversationContext:
    """Build a context whose all_history has `count` completed turns."""
    turns = []
    for i in range(count):
        turns.append(ConversationTurn(
            index=i + 1,
            input=AgentInput(message=f"user-{i}"),
            output=AgentOutput(
                message_for_user=f"agent-{i}",
                finished=False,
                agent_type=AgentType.COLLECT_EXPERIENCES_AGENT,
                agent_response_time_in_sec=0.1,
                llm_stats=[],
            ),
        ))
    history = ConversationHistory(turns=turns)
    return ConversationContext(
        all_history=history,
        history=history,
        summary="",
    )


def _state_with(collected: list[CollectedData], *, education_done: bool = True) -> CollectExperiencesAgentState:
    return CollectExperiencesAgentState(
        session_id=1,
        country_of_user=Country.UNSPECIFIED,
        collected_data=collected,
        first_time_visit=False,
        education_phase_done=education_done,
        unexplored_types=[WorkType.SELF_EMPLOYMENT,
                          WorkType.FORMAL_SECTOR_UNPAID_TRAINEE_WORK,
                          WorkType.UNSEEN_UNPAID],
        explored_types=[WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT],
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


@pytest.mark.asyncio
async def test_stale_orphan_is_pruned_from_state(patched_llms):
    """
    Regression test for the jasmin 145369667600005 transition bug.

    A titleless SELF_EMPLOYMENT entry defined at turn 14 (orphaned because a
    later ADD created a sibling with the real title instead of updating it)
    must be pruned so TransitionDecisionTool can move past SELF_EMPLOYMENT.
    """
    titled = _titled(title="Qualitative Research Assistant",
                     work_type=WorkType.SELF_EMPLOYMENT,
                     defined_at_turn=15, index=0)
    orphan = _orphan(work_type=WorkType.SELF_EMPLOYMENT, defined_at_turn=14, index=1)

    agent = CollectExperiencesAgent()
    agent.set_state(_state_with([titled, orphan]))

    # Current turn is much later than both — the orphan is stale.
    await agent.execute(
        user_input=AgentInput(message="ok"),
        context=_context_with_turn_count(50),
    )

    remaining_titles = [e.experience_title for e in agent._state.collected_data]
    assert remaining_titles == ["Qualitative Research Assistant"]
    assert len(agent._state.collected_data) == 1
    # The kept experience is re-indexed to position 0.
    assert agent._state.collected_data[0].index == 0


@pytest.mark.asyncio
async def test_fresh_orphan_is_preserved(patched_llms):
    """
    A titleless entry created on the CURRENT turn must stay. The conversation
    LLM is presumed to be asking for the title in its response this turn.
    """
    fresh_orphan = _orphan(work_type=WorkType.SELF_EMPLOYMENT, defined_at_turn=50, index=0)

    agent = CollectExperiencesAgent()
    agent.set_state(_state_with([fresh_orphan]))

    # current_turn_count matches the orphan's defined_at_turn_number: fresh, not stale.
    await agent.execute(
        user_input=AgentInput(message="some context"),
        context=_context_with_turn_count(50),
    )

    assert len(agent._state.collected_data) == 1
    assert agent._state.collected_data[0].experience_title is None


@pytest.mark.asyncio
async def test_pruning_preserves_last_referenced_experience_index(patched_llms):
    """
    When data extraction reports last_referenced_experience_index pointing at a
    survivor, pruning must remap the index so the conversation LLM still gets
    the correct 'last referenced' entry after list compaction.
    """
    orphan = _orphan(work_type=WorkType.SELF_EMPLOYMENT, defined_at_turn=10, index=0)
    titled = _titled(title="Selling Fruit", work_type=WorkType.SELF_EMPLOYMENT,
                     defined_at_turn=13, index=1)

    # Data extraction says the last referenced one was the titled entry (index 1).
    data_mock, conv_mock, _ = patched_llms
    data_mock.return_value.execute = AsyncMock(return_value=(1, []))

    agent = CollectExperiencesAgent()
    agent.set_state(_state_with([orphan, titled]))

    await agent.execute(
        user_input=AgentInput(message="ok"),
        context=_context_with_turn_count(50),
    )

    # After pruning, only the titled entry remains at position 0.
    assert len(agent._state.collected_data) == 1
    assert agent._state.collected_data[0].experience_title == "Selling Fruit"

    # The conversation LLM should have been called with the remapped index (0).
    call_kwargs = conv_mock.return_value.execute.call_args.kwargs
    assert call_kwargs["last_referenced_experience_index"] == 0


@pytest.mark.asyncio
async def test_no_pruning_when_all_titled(patched_llms):
    """Sanity: no-op when every experience already has a title."""
    a = _titled(title="A", work_type=WorkType.SELF_EMPLOYMENT, defined_at_turn=5, index=0)
    b = _titled(title="B", work_type=WorkType.SELF_EMPLOYMENT, defined_at_turn=6, index=1)

    agent = CollectExperiencesAgent()
    agent.set_state(_state_with([a, b]))

    await agent.execute(
        user_input=AgentInput(message="ok"),
        context=_context_with_turn_count(50),
    )

    assert [e.experience_title for e in agent._state.collected_data] == ["A", "B"]
