
import asyncio
import logging
import os
from datetime import datetime, timezone
import pytest

from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase, Node2VecRecommendations, OccupationRecommendation
)
from app.conversation_memory.conversation_memory_manager import ConversationMemoryManager
from app.conversation_memory.conversation_memory_types import ConversationMemoryManagerState
from app.countries import Country
from app.i18n.translation_service import get_i18n_manager
from app.server_config import UNSUMMARIZED_WINDOW_SIZE, TO_BE_SUMMARIZED_WINDOW_SIZE
from app.vector_search.vector_search_dependencies import SearchServices
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
        name='e2e_user_explains_concern_and_accepts',
        simulated_user_prompt="User is presented with Software Developer role, has concerns about skills, then accepts.",
        scripted_user=[
            "Hello",
            "This sounds interesting, tell me about Software Developer",
            "I am not sure if I have the right skills", # Concerns
            "Okay, what are the training options?", # Tradeoffs/Training
            "I will try to apply", # Action
            "Bye"
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
async def test_recommender_advisor_agent_e2e(
    max_iterations: int,
    test_case: ScriptedUserEvaluationTestCase, 
    caplog,
    setup_search_services: SearchServices # This injects the REAL services (or properly configured environment services)
):
    
    get_i18n_manager().set_locale(test_case.locale)
    print(f"Running E2E test case {test_case.name}")

    session_id = get_random_session_id()
    output_folder = os.path.join(os.getcwd(), 'test_output/recommender_agent/e2e', test_case.name)

    conversation_manager = ConversationMemoryManager(UNSUMMARIZED_WINDOW_SIZE, TO_BE_SUMMARIZED_WINDOW_SIZE)
    conversation_manager.set_state(state=ConversationMemoryManagerState(session_id=session_id))

    # Mock recommendations for initial state
    # We provide a recommendation that aligns with the user script ("Software Developer")
    search_services = await setup_search_services
    
    # Create fake recommendation
    rec = OccupationRecommendation(
        uuid="test_uuid",
        originUuid="test_origin_uuid",
        rank=1,
        occupation_id="2512",
        occupation_code="2512",
        occupation="Software Developer",
        confidence_score=0.9,
        description="Develops software."
    )
    
    recommendations = Node2VecRecommendations(
        youth_id="test_youth",
        occupation_recommendations=[rec]
    )

    agent_state = RecommenderAdvisorAgentState(
        session_id=session_id,
        youth_id="test_youth",
        country_of_user=Country.SOUTH_AFRICA,
        conversation_phase=ConversationPhase.INTRO,
        recommendations=recommendations
    )

    execute_evaluated_agent = RecommenderAgentExecutor(
        state=agent_state, 
        conversation_manager=conversation_manager,
        # We inject the REAL search service here
        occupation_search_service=search_services.occupation_search_service
    )

    config = ConversationTestConfig(
        max_iterations=len(test_case.scripted_user) + 5,
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
