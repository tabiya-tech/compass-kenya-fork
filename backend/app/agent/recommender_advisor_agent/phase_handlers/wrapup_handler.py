"""
Wrapup Phase Handler for the Recommender/Advisor Agent.

Handles the WRAPUP and COMPLETE phases where we summarize
the session and save to DB6.
"""

from typing import Any, Optional
from datetime import datetime, timezone

from app.agent.agent_types import LLMStats
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    OccupationRecommendation,
    OpportunityRecommendation,
    SkillsTrainingRecommendation,
)
from app.agent.recommender_advisor_agent.llm_response_models import ConversationResponse
from app.agent.recommender_advisor_agent.phase_handlers.base_handler import BasePhaseHandler
from app.conversation_memory.conversation_memory_manager import ConversationContext

# DB6 imports (Epic 1 dependency - optional)
try:
    from app.epic1.db6_youth_database.db6_client import DB6Client, YouthProfile
    DB6_AVAILABLE = True
except ImportError:
    DB6Client = None
    YouthProfile = None
    DB6_AVAILABLE = False


class WrapupPhaseHandler(BasePhaseHandler):
    """
    Handles the WRAPUP and COMPLETE phases.
    
    Responsibilities:
    - Summarize the session and action commitments
    - Save session data to DB6 (Youth Database)
    - Handle final barrier check ("what might stop you?")
    - Close the session gracefully
    """
    
    def __init__(self, *args, db6_client: Optional[Any] = None, **kwargs):
        """
        Initialize the wrapup handler.
        
        Args:
            db6_client: Optional DB6 client for saving session data
        """
        super().__init__(*args, **kwargs)
        self._db6_client = db6_client
    
    async def handle(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle session wrapup and finish conversation.
        """
        # Build summary
        summary = self._build_session_summary(state)

        # Save to DB6
        await self._save_session_to_db6(state)

        # Mark as complete and finish
        state.conversation_phase = ConversationPhase.COMPLETE

        return ConversationResponse(
            reasoning="Session complete, showing summary and finishing",
            message=summary,
            finished=True
        ), []
    
    async def handle_complete(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle session completion (final message).
        """
        return ConversationResponse(
            reasoning="Session complete",
            message="Great talking with you! Good luck with your next steps. Remember - persistence beats perfection. You've got this! 🚀",
            finished=True
        ), []
    
    def _build_session_summary(self, state: RecommenderAdvisorAgentState) -> str:
        """Build the session summary message."""
        parts = ["Perfect! Here's what we've discussed:\n"]

        # Current focus
        if state.current_focus_id:
            focus_title = self._get_focus_title(state)
            parts.append(f"\n**Your top match:** {focus_title}")

        # Action commitment
        if state.action_commitment:
            commitment = state.action_commitment
            action_display = commitment.action_type.value.replace("_", " ").title()
            timeline_display = commitment.commitment_level.value.replace("_", " ").title()

            parts.append(f"\n**Next step:** {action_display}")
            parts.append(f"\n**Timeline:** {timeline_display}")

            if commitment.barriers_mentioned:
                parts.append(f"\n**Potential barriers:** {', '.join(commitment.barriers_mentioned)}")

        parts.append("\n\nI've saved this to your profile so we can follow up.")
        parts.append("\n\nGood luck with your next steps! Remember - persistence beats perfection. You've got this! 🚀")

        return "".join(parts)
    
    def _get_focus_title(self, state: RecommenderAdvisorAgentState) -> str:
        """Get the title of the currently focused recommendation."""
        if state.current_focus_id is None:
            return "Unknown"
        
        rec = state.get_recommendation_by_id(state.current_focus_id)
        
        if rec is None:
            return "Unknown"
        
        if isinstance(rec, OccupationRecommendation):
            return rec.occupation
        elif isinstance(rec, OpportunityRecommendation):
            return rec.opportunity_title
        elif isinstance(rec, SkillsTrainingRecommendation):
            return rec.training_title
        
        return "Unknown"
    
    async def _save_session_to_db6(self, state: RecommenderAdvisorAgentState) -> None:
        """
        Save the session summary to DB6 (Youth Database).
        
        If DB6 is not available, logs a warning but does not fail.
        """
        if not self._db6_client or not DB6_AVAILABLE:
            self.logger.info("DB6 client not available, skipping session save")
            return
        
        try:
            session_summary = state.get_session_summary()
            
            # Fetch existing profile or create new
            profile = await self._db6_client.get_youth_profile(state.youth_id)
            
            if not profile:
                self.logger.info(f"Creating new youth profile for {state.youth_id}")
                profile = YouthProfile(youth_id=state.youth_id)
            
            # Add recommender session to interaction history
            profile.interaction_history.append({
                "agent": "RecommenderAdvisorAgent",
                "action": "recommender_session_completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "session_summary": session_summary
            })
            
            # Save to DB6
            await self._db6_client.save_youth_profile(profile)
            self.logger.info(f"Saved recommender session to DB6 for youth {state.youth_id}")
        
        except Exception as e:
            # Don't fail the conversation - just log the error
            self.logger.error(f"Failed to save session to DB6: {e}", exc_info=True)
