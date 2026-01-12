"""
Recommender/Advisor Agent module.

This agent presents occupation, opportunity, and training recommendations
to users and motivates them to take action (apply, enroll, explore).

Epic 3 implementation following the conversational agent pattern.

Module Structure:
- agent.py: Main orchestrator (slim, delegates to handlers)
- state.py: Agent state management
- types.py: Data models and enums
- llm_response_models.py: Pydantic models for LLM responses
- recommendation_interface.py: Node2Vec integration + stubs
- phase_handlers/: Phase-specific conversation handlers
"""

from app.agent.recommender_advisor_agent.agent import RecommenderAdvisorAgent
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    OccupationRecommendation,
    OpportunityRecommendation,
    SkillsTrainingRecommendation,
    Node2VecRecommendations,
    ActionCommitment,
    ResistanceType,
    ConversationPhase,
    UserInterestLevel,
    CommitmentLevel,
    ActionType,
)
from app.agent.recommender_advisor_agent.recommendation_interface import RecommendationInterface

__all__ = [
    # Main agent
    "RecommenderAdvisorAgent",
    "RecommenderAdvisorAgentState",
    
    # Recommendation types (from Node2Vec)
    "OccupationRecommendation",
    "OpportunityRecommendation", 
    "SkillsTrainingRecommendation",
    "Node2VecRecommendations",
    
    # User engagement types
    "ActionCommitment",
    "ResistanceType",
    "UserInterestLevel",
    "CommitmentLevel",
    "ActionType",
    
    # Conversation flow
    "ConversationPhase",
    
    # Interface
    "RecommendationInterface",
]
