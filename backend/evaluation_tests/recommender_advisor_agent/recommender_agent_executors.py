
from typing import Optional, Any
from app.agent.agent_types import AgentInput, AgentOutput
from app.agent.recommender_advisor_agent.agent import RecommenderAdvisorAgent
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.conversation_memory.conversation_memory_manager import ConversationMemoryManager
from app.conversation_memory.conversation_memory_types import ConversationContext
from app.vector_search.similarity_search_service import SimilaritySearchService
from app.vector_search.esco_entities import OccupationEntity

class RecommenderAgentExecutor:
    """
    Executes the Recommender Advisor Agent.
    """

    def __init__(
        self, 
        state: RecommenderAdvisorAgentState, 
        conversation_manager: ConversationMemoryManager,
        db6_client: Optional[Any] = None,
        node2vec_client: Optional[Any] = None,
        occupation_search_service: Optional[SimilaritySearchService[OccupationEntity]] = None
    ):
        self._agent = RecommenderAdvisorAgent(
            db6_client=db6_client,
            node2vec_client=node2vec_client,
            occupation_search_service=occupation_search_service
        )
        self._agent.set_state(state)
        self._conversation_manager = conversation_manager

    async def __call__(self, agent_input: AgentInput) -> AgentOutput:
        """
        Executes the agent with the given input.
        """
        context = await self._conversation_manager.get_conversation_context()
        agent_output = await self._agent.execute(agent_input, context)
        await self._conversation_manager.update_history(agent_input, agent_output)
        return agent_output

class RecommenderAgentGetConversationContextExecutor:
    """
    Returns the conversation context.
    """

    def __init__(self, conversation_manager: ConversationMemoryManager):
        self._conversation_manager = conversation_manager

    async def __call__(self) -> ConversationContext:
        return await self._conversation_manager.get_conversation_context()

class RecommenderAgentIsFinished:
    """
    Checks if the agent is finished.
    """

    def __call__(self, agent_output: AgentOutput) -> bool:
        return agent_output.finished
