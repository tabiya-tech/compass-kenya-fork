
import asyncio
import logging
import os
import pytest
from unittest.mock import MagicMock

from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import ConversationPhase
from app.conversation_memory.conversation_memory_manager import ConversationMemoryManager
from app.conversation_memory.conversation_memory_types import ConversationMemoryManagerState
from app.countries import Country
from app.i18n.translation_service import get_i18n_manager
from app.i18n.types import Locale
from app.server_config import UNSUMMARIZED_WINDOW_SIZE, TO_BE_SUMMARIZED_WINDOW_SIZE
from common_libs.test_utilities import get_random_session_id
from common_libs.test_utilities.guard_caplog import guard_caplog
from evaluation_tests.conversation_libs.conversation_test_function import conversation_test_function, \
    ConversationTestConfig, ScriptedUserEvaluationTestCase, \
    ScriptedSimulatedUser
from evaluation_tests.get_test_cases_to_run_func import get_test_cases_to_run
from evaluation_tests.recommender_advisor_agent.recommender_agent_executors import (
    RecommenderAgentExecutor, 
    RecommenderAgentIsFinished, 
    RecommenderAgentGetConversationContextExecutor
)

test_cases = [
    ScriptedUserEvaluationTestCase(
        name='happy_path_acceptance',
        simulated_user_prompt="Scripted user: Accepts recommendations and moves to action",
        scripted_user=[
            "Hello",
            "I like the first option", # Present -> Exploration
            "I am worried about the cost", # Exploration -> Concerns
            "That makes sense. What are the downsides?", # Concerns -> Tradeoffs
            "I want to apply", # Tradeoffs -> Action
            "I will update my CV", # Action -> Wrapup
            "Thank you, bye" # Wrapup -> Complete
        ],
        evaluations=[]
    ),
]

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.mark.asyncio
@pytest.mark.evaluation_test("gemini-2.0-flash-001/")
@pytest.mark.repeat(3)
@pytest.mark.parametrize('test_case', get_test_cases_to_run(test_cases),
                         ids=[case.name for case in get_test_cases_to_run(test_cases)])
async def test_recommender_agent_scripted_user(max_iterations: int,
                                               test_case: ScriptedUserEvaluationTestCase, caplog):
    
    get_i18n_manager().set_locale(test_case.locale)
    print(f"Running test case {test_case.name}")

    session_id = str(get_random_session_id())
    output_folder = os.path.join(os.getcwd(), 'test_output/recommender_agent/scripted', test_case.name)

    conversation_manager = ConversationMemoryManager(UNSUMMARIZED_WINDOW_SIZE, TO_BE_SUMMARIZED_WINDOW_SIZE)
    conversation_manager.set_state(state=ConversationMemoryManagerState(session_id=session_id))

    # Initialize State
    # Note: We are mocking dependencies so real recommendations aren't needed, 
    # but strictly the state needs valid objects or Nones where appropriate.
    agent_state = RecommenderAdvisorAgentState(
        session_id=session_id,
        youth_id="test_youth",
        country_of_user=Country.SOUTH_AFRICA,
        conversation_phase=ConversationPhase.INTRO
    )

    # Mocks for dependencies
    mock_db6 = MagicMock()
    mock_node2vec = MagicMock()
    mock_search = MagicMock()

    execute_evaluated_agent = RecommenderAgentExecutor(
        state=agent_state, 
        conversation_manager=conversation_manager,
        db6_client=mock_db6,
        node2vec_client=mock_node2vec,
        occupation_search_service=mock_search
    )

    config = ConversationTestConfig(
        max_iterations=len(test_case.scripted_user) + 5, # Give some buffer
        test_case=test_case,
        output_folder=output_folder,
        execute_evaluated_agent=execute_evaluated_agent,
        execute_simulated_user=ScriptedSimulatedUser(script=test_case.scripted_user),
        is_finished=RecommenderAgentIsFinished(),
        get_conversation_context=RecommenderAgentGetConversationContextExecutor(conversation_manager=conversation_manager)
    )

    with caplog.at_level(logging.INFO):
        guard_caplog(execute_evaluated_agent._agent._logger, caplog)

        await conversation_test_function(config=config)

        # Check if completed
        context = await conversation_manager.get_conversation_context()
