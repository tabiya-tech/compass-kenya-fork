"""
Skills Pivot Phase Handler for the Recommender/Advisor Agent.

Handles the SKILLS_UPGRADE_PIVOT phase where we present training
recommendations after user has rejected occupation options.
"""

from app.agent.agent_types import LLMStats
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    SkillsTrainingRecommendation,
    UserInterestLevel,
)
from app.agent.recommender_advisor_agent.llm_response_models import ConversationResponse
from app.agent.recommender_advisor_agent.phase_handlers.base_handler import BasePhaseHandler
from app.conversation_memory.conversation_memory_manager import ConversationContext


class SkillsPivotPhaseHandler(BasePhaseHandler):
    """
    Handles the SKILLS_UPGRADE_PIVOT phase.
    
    Responsibilities:
    - Present training/skill-building recommendations
    - Frame training as a path to future opportunities
    - Connect each training to specific occupations it unlocks
    - Keep door open for deeper conversation about barriers
    
    Trigger: User has rejected >= 3 occupations.
    """
    
    async def handle(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle presenting training recommendations after occupation rejections.
        """
        # Mark that we've pivoted
        state.pivoted_to_training = True
        
        if state.recommendations is None:
            return ConversationResponse(
                reasoning="No recommendations available for pivot",
                message="Let me find some alternative paths for you. What kind of skills would you like to develop?",
                finished=False
            ), []
        
        # Get training recommendations
        trainings = state.recommendations.skillstraining_recommendations[:5]
        
        if not trainings:
            # No training recommendations - explore why they rejected everything
            return await self._handle_no_trainings(user_input, state, context)
        
        # Build training presentation
        message = self._build_training_presentation(trainings, state)
        
        # Build metadata for UI
        metadata = self._build_metadata(
            interaction_type="training_presentation",
            trainings=[
                {
                    "uuid": trn.uuid,
                    "skill": trn.skill,
                    "training_title": trn.training_title,
                    "provider": trn.provider,
                    "estimated_hours": trn.estimated_hours,
                    "cost": trn.cost,
                    "target_occupations": trn.target_occupations,
                    "justification": trn.justification,
                }
                for trn in trainings
            ]
        )
        
        return ConversationResponse(
            reasoning="Pivoting to training recommendations after occupation rejections",
            message=message,
            finished=False,
            metadata=metadata
        ), []
    
    def _build_training_presentation(
        self,
        trainings: list[SkillsTrainingRecommendation],
        state: RecommenderAdvisorAgentState
    ) -> str:
        """Build the training recommendations message."""
        parts = [
            "I understand none of those career paths felt right. That's completely okay - let's approach this differently.\n\n",
            "Looking at your interests, here are skill-building opportunities that could open up new options:\n\n",
            "**Training Recommendations:**\n"
        ]
        
        for i, trn in enumerate(trainings, 1):
            # Build header
            provider_info = f" ({trn.provider})" if trn.provider else ""
            hours_info = f", {trn.estimated_hours} hours" if trn.estimated_hours else ""
            parts.append(f"\n**{i}. {trn.training_title}**{provider_info}{hours_info}")
            
            # Cost info
            if trn.cost:
                parts.append(f"\n   - Cost: {trn.cost}")
            
            # Justification
            parts.append(f"\n   - {trn.justification}")
            
            # What occupations it opens
            if trn.target_occupations:
                targets = ", ".join(trn.target_occupations[:3])
                parts.append(f"\n   - Opens doors to: {targets}")
            
            # Delivery mode
            if trn.delivery_mode:
                mode_display = trn.delivery_mode.replace("_", " ").capitalize()
                parts.append(f"\n   - Format: {mode_display}")
            
            # Track as presented
            if trn.uuid not in state.presented_trainings:
                state.presented_trainings.append(trn.uuid)
        
        parts.append("\n\nWould building these skills make you feel more confident about career options?")
        parts.append("\n\nOr is there something else holding you back that we should talk about?")
        
        return "".join(parts)
    
    async def _handle_no_trainings(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Handle case where no training recommendations are available."""
        
        # This often means deeper issues - explore barriers
        message = """I understand none of those career paths felt right.

Before I suggest more options, I'd like to understand what's really driving your decisions. Sometimes it's not about the careers themselves, but about other things:

- **Confidence**: Feeling unsure if you can succeed?
- **External pressures**: Family expectations, financial constraints?
- **Clarity**: Still unsure what you actually want?

What's the main thing that makes these options feel wrong for you?"""
        
        # Move to concerns phase to explore deeper
        state.conversation_phase = ConversationPhase.ADDRESS_CONCERNS
        
        return ConversationResponse(
            reasoning="No training recommendations available, exploring deeper barriers",
            message=message,
            finished=False
        ), []
    
    async def handle_training_interest(
        self,
        user_input: str,
        training_id: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Handle when user expresses interest in a specific training."""
        
        # Mark interest
        state.mark_interest(training_id, UserInterestLevel.INTERESTED)
        state.current_focus_id = training_id
        state.current_recommendation_type = "training"
        
        # Get the training details
        training = None
        if state.recommendations:
            for trn in state.recommendations.skillstraining_recommendations:
                if trn.uuid == training_id:
                    training = trn
                    break
        
        if training is None:
            state.conversation_phase = ConversationPhase.ACTION_PLANNING
            return ConversationResponse(
                reasoning="Could not find training details",
                message="Great! Let's plan how you'll get started with that training.",
                finished=False
            ), []
        
        # Build detailed training info and transition to action
        message = f"""Great choice! Let me tell you more about **{training.training_title}**:

**Provider:** {training.provider or "Various"}
**Duration:** {training.estimated_hours or "Varies"} hours
**Format:** {(training.delivery_mode or "online").replace("_", " ").capitalize()}
**Cost:** {training.cost or "Contact provider"}

**What you'll learn:**
{training.justification}

**Career doors this opens:**
{', '.join(training.target_occupations[:4]) if training.target_occupations else "Multiple career paths"}

Would you like to start this training? I can help you take the next step."""
        
        state.conversation_phase = ConversationPhase.ACTION_PLANNING
        
        return ConversationResponse(
            reasoning=f"User interested in training: {training.training_title}",
            message=message,
            finished=False
        ), []
