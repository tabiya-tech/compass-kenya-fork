
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional

from app.agent.recommender_advisor_agent.agent import RecommenderAdvisorAgent
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import ConversationPhase
from app.agent.agent_types import AgentInput, AgentType, AgentOutputWithReasoning
from app.conversation_memory.conversation_memory_manager import ConversationContext
from app.agent.recommender_advisor_agent.llm_response_models import ConversationResponse

class TestRecommenderAdvisorAgent:
    """
    Tests for the RecommenderAdvisorAgent
    """

    @pytest.fixture
    def mock_db6_client(self):
        return MagicMock()

    @pytest.fixture
    def mock_node2vec_client(self):
        return MagicMock()

    @pytest.fixture
    def mock_occupation_search_service(self):
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_db6_client, mock_node2vec_client, mock_occupation_search_service):
        with patch('app.agent.recommender_advisor_agent.agent.GeminiGenerativeLLM') as mock_llm_cls:
            agent = RecommenderAdvisorAgent(
                db6_client=mock_db6_client,
                node2vec_client=mock_node2vec_client,
                occupation_search_service=mock_occupation_search_service
            )
            return agent

    @pytest.fixture
    def mock_state(self):
        state = MagicMock(spec=RecommenderAdvisorAgentState)
        state.conversation_phase = ConversationPhase.INTRO
        state.turn_count = 0
        return state

    @pytest.fixture
    def mock_context(self):
        return MagicMock(spec=ConversationContext)

    def test_initialization(self, agent):
        # THEN expected agent type is correct
        assert agent.agent_type == AgentType.RECOMMENDER_ADVISOR_AGENT
        # AND handlers are initialized
        assert agent._intro_handler is not None
        assert agent._present_handler is not None
        assert agent._exploration_handler is not None
        assert agent._concerns_handler is not None
        assert agent._tradeoffs_handler is not None
        assert agent._action_handler is not None
        assert agent._wrapup_handler is not None

    @pytest.mark.asyncio
    async def test_execute_routes_properly_intro(self, agent, mock_state, mock_context):
        # GIVEN the agent is in INTRO phase
        mock_state.conversation_phase = ConversationPhase.INTRO
        agent.set_state(mock_state)

        # AND the intro handler is mocked
        agent._intro_handler.handle = AsyncMock(return_value=(
            ConversationResponse(message="Hello", finished=False, reasoning="Intro"), 
            []
        ))

        # WHEN execute is called
        user_input = AgentInput(message="Hi", is_artificial=False)
        output = await agent.execute(user_input, mock_context)

        # THEN expected the intro handler to be called
        agent._intro_handler.handle.assert_called_once_with("Hi", mock_state, mock_context)
        # AND result matches
        assert isinstance(output, AgentOutputWithReasoning)
        assert output.message_for_user == "Hello"
        assert output.reasoning == "Intro"

    @pytest.mark.asyncio
    async def test_execute_routes_properly_exploration(self, agent, mock_state, mock_context):
        # GIVEN the agent is in CAREER_EXPLORATION phase
        mock_state.conversation_phase = ConversationPhase.CAREER_EXPLORATION
        agent.set_state(mock_state)

        # AND the exploration handler is mocked
        agent._exploration_handler.handle = AsyncMock(return_value=(
            ConversationResponse(message="Exploring", finished=False, reasoning="Explore"), 
            []
        ))

        # WHEN execute is called
        user_input = AgentInput(message="Tell me more", is_artificial=False)
        output = await agent.execute(user_input, mock_context)

        # THEN expected the exploration handler to be called
        agent._exploration_handler.handle.assert_called_once_with("Tell me more", mock_state, mock_context)
        # AND result matches
        assert output.message_for_user == "Exploring"

    @pytest.mark.asyncio
    async def test_error_handling(self, agent, mock_state, mock_context):
        # GIVEN the agent is in INTRO phase
        mock_state.conversation_phase = ConversationPhase.INTRO
        agent.set_state(mock_state)

        # AND the handler raises an exception
        agent._intro_handler.handle = AsyncMock(side_effect=Exception("Something bad happened"))

        # WHEN execute is called
        user_input = AgentInput(message="Hi", is_artificial=False)
        output = await agent.execute(user_input, mock_context)

        # THEN expected an error response
        assert "I apologize" in output.message_for_user
        assert output.finished is False

