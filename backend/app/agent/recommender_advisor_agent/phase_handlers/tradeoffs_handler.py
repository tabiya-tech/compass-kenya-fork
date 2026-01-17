"""
Tradeoffs Phase Handler for the Recommender/Advisor Agent.

Handles the DISCUSS_TRADEOFFS phase where we balance user preferences
against labor market realities.
"""

from typing import Optional

from app.agent.agent_types import LLMStats
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    OccupationRecommendation,
)
from app.agent.recommender_advisor_agent.llm_response_models import ConversationResponse
from app.agent.recommender_advisor_agent.phase_handlers.base_handler import BasePhaseHandler
from app.agent.simple_llm_agent.prompt_response_template import get_json_response_instructions
from app.conversation_memory.conversation_formatter import ConversationHistoryFormatter
from app.conversation_memory.conversation_memory_manager import ConversationContext


class TradeoffsPhaseHandler(BasePhaseHandler):
    """
    Handles the DISCUSS_TRADEOFFS phase.
    
    Responsibilities:
    - Present tradeoffs between user preferences and labor demand
    - Help user understand "stepping stone" concept
    - Avoid pushing high-demand options manipulatively
    - Let user make informed choice
    
    Trigger: User prefers a low-demand occupation over a high-demand one.
    """
    
    async def handle(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle discussing preference vs demand tradeoffs.
        """
        all_llm_stats: list[LLMStats] = []

        # Get the user's preferred occupation and alternatives
        preferred_occ = self._get_user_preferred_occupation(state)
        high_demand_alt = self._get_high_demand_alternative(state)

        if preferred_occ is None:
            # No clear preference, go back to exploration
            state.conversation_phase = ConversationPhase.CAREER_EXPLORATION
            return ConversationResponse(
                reasoning="No clear preference for tradeoff discussion",
                message="Let me understand - which of these career paths interests you most?",
                finished=False
            ), []

        # If no high-demand alternative, skip tradeoffs entirely
        if high_demand_alt is None or high_demand_alt.uuid == preferred_occ.uuid:
            state.conversation_phase = ConversationPhase.CAREER_EXPLORATION
            return ConversationResponse(
                reasoning="No tradeoff needed - proceeding with user's choice",
                message=f"Great choice with **{preferred_occ.occupation}**! Let's talk about your next steps.",
                finished=False
            ), []

        # Generate tradeoff discussion
        response, llm_stats = await self._generate_tradeoff_message(
            preferred_occ, high_demand_alt, state, context, user_input
        )
        all_llm_stats.extend(llm_stats)

        # After presenting the tradeoff, transition to CAREER_EXPLORATION
        # so the user can explore their chosen option
        state.conversation_phase = ConversationPhase.CAREER_EXPLORATION

        return response, all_llm_stats
    
    def _get_user_preferred_occupation(
        self,
        state: RecommenderAdvisorAgentState
    ) -> Optional[OccupationRecommendation]:
        """Get the occupation the user seems to prefer."""
        if state.current_focus_id is None:
            return None
        
        rec = state.get_recommendation_by_id(state.current_focus_id)
        if isinstance(rec, OccupationRecommendation):
            return rec
        return None
    
    def _get_high_demand_alternative(
        self,
        state: RecommenderAdvisorAgentState
    ) -> Optional[OccupationRecommendation]:
        """Get the highest-demand occupation that's different from user's preference."""
        if state.recommendations is None:
            return None
        
        # Sort by demand (high > medium > low)
        demand_priority = {"high": 0, "medium": 1, "low": 2, None: 3}
        
        occupations = sorted(
            state.recommendations.occupation_recommendations,
            key=lambda x: (demand_priority.get(x.labor_demand_category, 3), x.rank)
        )
        
        # Return first high-demand occupation that's not the current focus
        for occ in occupations:
            if occ.labor_demand_category == "high" and occ.uuid != state.current_focus_id:
                return occ
        
        return None
    
    async def _generate_tradeoff_message(
        self,
        preferred: OccupationRecommendation,
        alternative: OccupationRecommendation,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext,
        user_input: str
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Generate the tradeoff discussion message."""
        
        # Build context about the tradeoff
        preferred_demand = preferred.labor_demand_category or "unknown"
        alt_demand = alternative.labor_demand_category or "high"
        
        prompt = f"""
Generate a tradeoff discussion between two career options.

**User's Preference:** {preferred.occupation}
- Labor Demand: {preferred_demand}
- Why they like it: {preferred.justification}
- Salary: {preferred.salary_range or "Unknown"}

**Alternative (Higher Demand):** {alternative.occupation}
- Labor Demand: {alt_demand}
- Match reason: {alternative.justification}
- Salary: {alternative.salary_range or "Unknown"}

Generate a message that:
1. Acknowledges their preference is VALID (don't dismiss it)
2. Presents the tradeoff clearly (not as "you should do X instead")
3. Introduces the "stepping stone" concept (start with B, transition to A later)
4. Asks genuinely: which path would they prefer?

HARD RULES:
- Don't push the high-demand option as the "right" choice
- Don't imply their preference is wrong
- Present information, let THEM decide
- Use language like "Here's the tradeoff to consider..."
- Set finished=false (the conversation continues after this)

{get_json_response_instructions()}
"""
        
        response, llm_stats = await self._conversation_caller.call_llm(
            llm=self._conversation_llm,
            llm_input=ConversationHistoryFormatter.format_for_agent_generative_prompt(
                model_response_instructions=prompt,
                context=context,
                user_input=user_input
            ),
            logger=self.logger
        )
        
        return response, llm_stats
