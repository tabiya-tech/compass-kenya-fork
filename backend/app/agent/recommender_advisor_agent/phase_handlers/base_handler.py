"""
Base Phase Handler for the Recommender/Advisor Agent.

Provides common functionality for all phase handlers.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

from app.agent.agent_types import LLMStats
from app.agent.llm_caller import LLMCaller
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.llm_response_models import ConversationResponse
from app.conversation_memory.conversation_memory_manager import ConversationContext
from common_libs.llm.generative_models import GeminiGenerativeLLM


class BasePhaseHandler(ABC):
    """
    Abstract base class for phase handlers.
    
    Each phase handler is responsible for a specific conversation phase
    and knows how to process user input and generate responses for that phase.
    """
    
    def __init__(
        self,
        conversation_llm: GeminiGenerativeLLM,
        conversation_caller: LLMCaller[ConversationResponse],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the phase handler.
        
        Args:
            conversation_llm: LLM for generating conversational responses
            conversation_caller: Typed LLM caller for response parsing
            logger: Optional logger instance
        """
        self._conversation_llm = conversation_llm
        self._conversation_caller = conversation_caller
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def handle(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle the current phase.
        
        Args:
            user_input: User's message
            state: Current agent state
            context: Conversation context
            
        Returns:
            Tuple of (ConversationResponse, list of LLMStats)
        """
        pass
    
    def _build_metadata(self, **kwargs) -> dict[str, Any]:
        """
        Build metadata dict for UI rendering.
        
        Args:
            **kwargs: Key-value pairs to include in metadata
            
        Returns:
            Metadata dictionary
        """
        return {k: v for k, v in kwargs.items() if v is not None}
