"""
Intro Phase Handler for the Recommender/Advisor Agent.

Handles the introduction phase where we set expectations
and prepare recommendations.
"""

from app.agent.agent_types import LLMStats
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import ConversationPhase
from app.agent.recommender_advisor_agent.llm_response_models import ConversationResponse
from app.agent.recommender_advisor_agent.phase_handlers.base_handler import BasePhaseHandler
from app.agent.recommender_advisor_agent.recommendation_interface import RecommendationInterface
from app.conversation_memory.conversation_memory_manager import ConversationContext


class IntroPhaseHandler(BasePhaseHandler):
    """
    Handles the INTRO phase.
    
    Responsibilities:
    - Set expectations about what's coming
    - Generate/load recommendations if not already set
    - Transition to PRESENT_RECOMMENDATIONS
    """
    
    def __init__(self, *args, recommendation_interface: RecommendationInterface, **kwargs):
        """
        Initialize the intro handler.
        
        Args:
            recommendation_interface: Interface for generating recommendations
            *args, **kwargs: Passed to BasePhaseHandler
        """
        super().__init__(*args, **kwargs)
        self._recommendation_interface = recommendation_interface
    
    async def handle(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle the intro phase.
        
        Generates recommendations if needed and presents introduction message.
        """
        # Generate recommendations if not already set
        if state.recommendations is None:
            recommendations = await self._recommendation_interface.generate_recommendations(
                youth_id=state.youth_id,
                city=state.city,
                province=state.province,
                preference_vector=state.preference_vector,
                skills_vector=state.skills_vector,
                bws_scores=state.bws_scores,
                top_10_bws=state.top_10_bws,
                education_experiences=state.education_experiences,
            )

            # An empty result (matching service returned nothing, or had a transient
            # issue) must NOT be cached: leaving state.recommendations as None means the
            # next user turn re-enters this branch and calls the matching service again,
            # rather than freezing on a "no recommendations" screen. Stay in INTRO and
            # invite the user to try again so the retry actually happens.
            if recommendations.is_empty():
                self.logger.warning(
                    f"No recommendations returned for {state.youth_id}; staying in INTRO to retry on next turn"
                )
                return ConversationResponse(
                    reasoning="Matching service returned no recommendations; not caching so the next turn retries",
                    message=(
                        "I wasn't able to pull up matching options for you just yet — this can "
                        "happen while our job database is updating. Send me a message to try again, "
                        "or check back a little later and I'll have a fresh set for you."
                    ),
                    finished=False,
                ), []

            state.recommendations = recommendations
            self.logger.info(
                f"Generated recommendations for {state.youth_id}: "
                f"{len(state.recommendations.occupation_recommendations)} occupations, "
                f"{len(state.recommendations.opportunity_recommendations)} opportunities, "
                f"{len(state.recommendations.skillstraining_recommendations)} trainings"
            )

        # Build intro message — ask up front whether they want career paths, actual job
        # openings, or both, so everyone can reach jobs quickly without having to ask.
        intro_message = """Great! Based on what you've told me, I've got some options for you.

            Would you like to see career paths to consider, actual job openings you could apply to right now, or both?
        """

        response = ConversationResponse(
            reasoning="Introducing the recommendation session and asking whether the user wants careers, jobs, or both",
            message=intro_message,
            finished=False
        )

        # The user's answer is classified in the PRESENT phase, which then renders the
        # chosen view. Flag that we're awaiting that choice.
        state.awaiting_view_choice = True

        # Transition to presenting recommendations
        state.conversation_phase = ConversationPhase.PRESENT_RECOMMENDATIONS
        
        return response, []
