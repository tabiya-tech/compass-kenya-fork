"""
Present Phase Handler for the Recommender/Advisor Agent.

Handles the PRESENT_RECOMMENDATIONS phase where we show
occupation recommendations to the user using LLM for natural presentation.
"""

from typing import Optional

from app.agent.agent_types import LLMStats
from app.agent.llm_caller import LLMCaller
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    UserInterestLevel,
    OccupationRecommendation
)
from app.agent.recommender_advisor_agent.llm_response_models import (
    ConversationResponse,
    UserIntentClassification
)
from app.agent.recommender_advisor_agent.phase_handlers.base_handler import BasePhaseHandler
from app.agent.recommender_advisor_agent.prompts import (
    PRESENT_RECOMMENDATIONS_PROMPT,
    build_context_block
)
from app.agent.recommender_advisor_agent.intent_classifier import IntentClassifier
from app.conversation_memory.conversation_memory_manager import ConversationContext
from app.conversation_memory.conversation_formatter import ConversationHistoryFormatter
from app.agent.simple_llm_agent.prompt_response_template import get_json_response_instructions
from common_libs.llm.generative_models import GeminiGenerativeLLM
from app.vector_search.esco_entities import OccupationEntity
from app.vector_search.similarity_search_service import SimilaritySearchService


class PresentPhaseHandler(BasePhaseHandler):
    """
    Handles the PRESENT_RECOMMENDATIONS phase.

    Responsibilities:
    - Present top 3-5 occupation recommendations
    - Track which recommendations have been shown
    - Parse user interest signals to route to appropriate next phase
    - Detect user intent (explore, reject, question)
    - Update state with engagement signals
    - Transition to appropriate next phase
    """

    def __init__(
        self,
        conversation_llm: GeminiGenerativeLLM,
        conversation_caller: LLMCaller[ConversationResponse],
        intent_classifier: IntentClassifier = None,
        exploration_handler: 'ExplorationPhaseHandler' = None,
        concerns_handler: 'ConcernsPhaseHandler' = None,
        tradeoffs_handler: 'TradeoffsPhaseHandler' = None,
        occupation_search_service: Optional[SimilaritySearchService[OccupationEntity]] = None,
        **kwargs
    ):
        """
        Initialize the present handler.

        Args:
            conversation_llm: LLM for generating responses
            conversation_caller: LLM caller for conversation responses
            intent_classifier: Optional intent classifier for detecting user intent
            exploration_handler: Optional exploration handler for immediate transitions
            concerns_handler: Optional concerns handler for immediate transitions
            tradeoffs_handler: Optional tradeoffs handler for immediate transitions
            occupation_search_service: Optional occupation search service for finding occupations not in recommendations
        """
        super().__init__(conversation_llm, conversation_caller, **kwargs)
        self._intent_classifier = intent_classifier
        self._exploration_handler = exploration_handler
        self._concerns_handler = concerns_handler
        self._tradeoffs_handler = tradeoffs_handler
        self._occupation_search_service = occupation_search_service
    
    async def handle(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle presenting occupation recommendations using LLM for natural presentation.
        Also detects user intent and transitions to appropriate phase.
        """
        all_llm_stats: list[LLMStats] = []

        if state.recommendations is None:
            return ConversationResponse(
                reasoning="No recommendations available",
                message="I don't have any recommendations ready yet. Let me prepare some for you.",
                finished=False
            ), []

        # Get top occupations to present (strict rank order)
        occupations = state.recommendations.occupation_recommendations[:5]

        if not occupations:
            return ConversationResponse(
                reasoning="No occupation recommendations",
                message="I couldn't find suitable occupation recommendations. Let me try a different approach.",
                finished=False
            ), []

        # Check if we've already presented recommendations before
        # (more reliable than turn counting)
        has_presented_before = len(state.presented_occupations) > 0

        # Track as presented (only on first presentation)
        for occ in occupations:
            if occ.uuid not in state.presented_occupations:
                state.presented_occupations.append(occ.uuid)

        # If this is not the initial presentation (user has responded), detect their intent
        is_initial_presentation = user_input.strip() == "" or not has_presented_before

        if not is_initial_presentation and self._intent_classifier:
            # Classify user intent using centralized classifier
            self.logger.info(f"Classifying user intent for: '{user_input}' (turn_count={state.conversation_turn_count})")
            intent, intent_stats = await self._intent_classifier.classify_intent(
                user_input=user_input,
                state=state,
                context=context,
                phase=ConversationPhase.PRESENT_RECOMMENDATIONS,
                llm=self._conversation_llm,
                logger=self.logger
            )
            all_llm_stats.extend(intent_stats)

            # Check if intent classification succeeded
            if intent is None:
                self.logger.error("Intent classification failed after all retries, falling back to conversational response")
                # Fall through to regular conversation response generation below
            else:
                self.logger.info(f"Intent classified as: {intent.intent}, reasoning: {intent.reasoning}")

                # Handle the intent and potentially transition phases
                phase_transition = await self._handle_user_intent(intent, user_input, state, context)

                if phase_transition:
                    # Intent handled, phase transition will occur
                    all_llm_stats.extend(phase_transition[1])
                    return phase_transition[0], all_llm_stats

        # Check if there's a pending out-of-list occupation that user might be persisting on
        # This handles cases where intent classification failed but user is still talking about the pending occupation
        if state.pending_out_of_list_occupation:
            # Check if user might be persisting using LLM
            is_persistence = await self._is_user_persisting_on_pending_occupation(
                user_input=user_input,
                pending_occupation=state.pending_out_of_list_occupation,
                context=context
            )

            if is_persistence:
                # User is persisting on out-of-list occupation → Handle it
                self.logger.info(
                    f"User persisting on pending out-of-list occupation '{state.pending_out_of_list_occupation}' "
                    f"(intent classification didn't catch it)"
                )
                response, llm_stats = await self._handle_request_outside_recommendations(
                    requested_occupation_name=state.pending_out_of_list_occupation,
                    user_input=user_input,
                    state=state,
                    context=context
                )
                all_llm_stats.extend(llm_stats)
                return response, all_llm_stats

        # Build recommendations summary for LLM context
        recs_summary = self._build_detailed_recommendations_summary(occupations)

        # Build full context for LLM
        skills_list = self._extract_skills_list(state)
        pref_vec_dict = state.preference_vector.model_dump() if state.preference_vector else {}
        conv_history = ConversationHistoryFormatter.format_to_string(context)

        context_block = build_context_block(
            skills=skills_list,
            preference_vector=pref_vec_dict,
            recommendations_summary=recs_summary,
            conversation_history=conv_history,
            country_of_user=state.country_of_user
        )

        # Build prompt for LLM
        full_prompt = context_block + PRESENT_RECOMMENDATIONS_PROMPT + get_json_response_instructions()

        # Call LLM to generate natural presentation
        response, llm_stats = await self._call_llm(full_prompt, user_input, context)
        all_llm_stats.extend(llm_stats)

        # Build metadata for structured UI rendering
        metadata = self._build_metadata(
            interaction_type="occupation_presentation",
            occupations=[
                {
                    "uuid": occ.uuid,
                    "occupation": occ.occupation,
                    "rank": occ.rank,
                    "confidence_score": occ.confidence_score,
                    "labor_demand_category": occ.labor_demand_category,
                    "salary_range": occ.salary_range,
                    "justification": occ.justification,
                    "skills_match_score": occ.skills_match_score,
                    "preference_match_score": occ.preference_match_score,
                    "labor_demand_score": occ.labor_demand_score,
                }
                for occ in occupations
            ]
        )

        response.metadata = metadata

        return response, all_llm_stats

    def _build_detailed_recommendations_summary(self, occupations: list) -> str:
        """Build a detailed summary of occupation recommendations for LLM context."""
        lines = []
        for i, occ in enumerate(occupations, 1):
            lines.append(f"{i}. **{occ.occupation}** (Rank: {occ.rank}, Confidence: {occ.confidence_score:.0%})")
            lines.append(f"   - Labor Demand: {occ.labor_demand_category or 'Unknown'}")
            lines.append(f"   - Salary Range: {occ.salary_range or 'Not specified'}")
            lines.append(f"   - Justification: {occ.justification or 'N/A'}")

            # Add score breakdowns if available
            if occ.skills_match_score is not None:
                lines.append(f"   - Skills Match: {occ.skills_match_score:.0%}")
            if occ.preference_match_score is not None:
                lines.append(f"   - Preference Match: {occ.preference_match_score:.0%}")
            if occ.labor_demand_score is not None:
                lines.append(f"   - Labor Demand Score: {occ.labor_demand_score:.0%}")

            lines.append("")  # Blank line between occupations

        return '\n'.join(lines)

    async def _handle_user_intent(
        self,
        intent: UserIntentClassification,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]] | None:
        """
        Handle classified user intent and update state accordingly.

        Returns:
            Tuple of (response, llm_stats) if intent triggers phase transition, None otherwise
        """
        self.logger.info(f"Classified intent: {intent.intent} (reasoning: {intent.reasoning})")

        # GUARDRAIL: Check for off-recommendation requests first
        if intent.intent == "request_outside_recommendations":
            self.logger.warning(f"GUARDRAIL TRIGGERED: User requested occupation outside recommendations: {intent.requested_occupation_name}")
            # Use strict guardrail to redirect back to recommendations
            return await self._handle_request_outside_recommendations(
                requested_occupation_name=intent.requested_occupation_name or "that occupation",
                user_input=user_input,
                state=state,
                context=context
            )

        # Handle EXPLORE intent - user wants to learn more about a specific occupation
        elif intent.intent == "explore_occupation":
            return await self._handle_explore_intent(intent, user_input, state, context)

        # Handle REJECT intent - user is rejecting recommendations
        elif intent.intent == "reject":
            return await self._handle_reject_intent(intent, state, context)

        # Handle CONCERN intent - user expressing worries
        elif intent.intent == "express_concern":
            return await self._handle_concern_intent(intent, user_input, state, context)

        # Handle ACCEPT intent - user likes a recommendation
        elif intent.intent == "accept":
            return await self._handle_accept_intent(intent, user_input, state, context)

        # For other intents (questions, unclear), let LLM handle conversationally
        # The LLM has instructions in the base prompt to handle user-suggested occupations
        return None

    async def _handle_explore_intent(
        self,
        intent: UserIntentClassification,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Handle user wanting to explore a specific occupation."""
        # Determine which occupation they want to explore
        target_occ = None

        if intent.target_occupation_index and state.recommendations:
            idx = intent.target_occupation_index - 1  # Convert to 0-indexed
            occupations = state.recommendations.occupation_recommendations
            if 0 <= idx < len(occupations):
                target_occ = occupations[idx]
                self.logger.info(f"Identified occupation by index {intent.target_occupation_index}: {target_occ.occupation}")

        elif intent.target_recommendation_id:
            target_occ = state.get_recommendation_by_id(intent.target_recommendation_id)
            if target_occ:
                self.logger.info(f"Identified occupation by UUID {intent.target_recommendation_id}: {target_occ.occupation}")

        if target_occ:
            # Set focus and mark as exploring
            state.current_focus_id = target_occ.uuid
            state.current_recommendation_type = "occupation"
            state.mark_interest(target_occ.uuid, UserInterestLevel.EXPLORING)

            # Check if we should discuss tradeoffs BEFORE exploring
            # (user interested in lower-demand occupation when high-demand alternatives exist)
            if self._should_discuss_tradeoffs(target_occ, state):
                state.conversation_phase = ConversationPhase.DISCUSS_TRADEOFFS
                
                self.logger.info(f"User interested in lower-demand {target_occ.occupation}, transitioning to DISCUSS_TRADEOFFS")
                
                # If we have a tradeoffs handler, immediately invoke it
                if self._tradeoffs_handler:
                    self.logger.info("Immediately invoking tradeoffs handler for seamless experience")
                    return await self._tradeoffs_handler.handle(user_input, state, context)
                
                # Fallback: skip tradeoffs if no handler, proceed to exploration
                self.logger.warning("No tradeoffs handler available, proceeding to EXPLORATION")
                state.conversation_phase = ConversationPhase.CAREER_EXPLORATION

            # Transition to CAREER_EXPLORATION phase
            if state.conversation_phase != ConversationPhase.DISCUSS_TRADEOFFS:
                state.conversation_phase = ConversationPhase.CAREER_EXPLORATION

            self.logger.info(f"Transitioning to CAREER_EXPLORATION for {target_occ.occupation}")

            # If we have an exploration handler, immediately invoke it for seamless transition
            if self._exploration_handler:
                self.logger.info("Immediately invoking exploration handler for seamless experience")
                return await self._exploration_handler.handle(user_input, state, context)

            # Fallback: just return transition message (requires another user turn)
            return ConversationResponse(
                reasoning=f"User wants to explore {target_occ.occupation}, transitioning to EXPLORATION phase",
                message=f"Great! Let me tell you more about {target_occ.occupation}.",
                finished=False
            ), []

        # Couldn't identify which occupation - stay in PRESENT phase and let LLM handle
        # This can happen if user says general things like "tell me more" without specifying which one
        # OR if they mention an occupation not in our recommendations (e.g., "I want to be a DJ")
        # The LLM has instructions in the base prompt to handle user-suggested occupations appropriately
        self.logger.warning(f"Could not identify target occupation from intent. target_occupation_index={intent.target_occupation_index}, target_recommendation_id={intent.target_recommendation_id}")
        return None


    async def _handle_reject_intent(
        self,
        intent: UserIntentClassification,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Handle user rejecting recommendations."""
        # Mark rejection in state
        # If they rejected a specific occupation, mark it
        if intent.target_recommendation_id:
            state.mark_interest(intent.target_recommendation_id, UserInterestLevel.REJECTED)
            self.logger.info(f"User rejected specific recommendation: {intent.target_recommendation_id}")
        else:
            # General rejection - mark the most recently discussed or all presented
            # For now, just increment the counter
            state.rejected_occupations += 1
            self.logger.info(f"User rejected recommendations generally. Total rejections: {state.rejected_occupations}")

        # Check if we should pivot to training (3+ rejections)
        if state.should_pivot_to_training():
            state.conversation_phase = ConversationPhase.SKILLS_UPGRADE_PIVOT
            state.pivoted_to_training = True

            return ConversationResponse(
                reasoning="User rejected 3+ occupations, pivoting to training recommendations",
                message="I understand these occupations aren't quite right. Let me show you some training opportunities that could open up new career paths for you.",
                finished=False
            ), []

        # If multiple rejections but not ready to pivot, transition to ADDRESS_CONCERNS
        if state.rejected_occupations >= 2:
            state.conversation_phase = ConversationPhase.ADDRESS_CONCERNS

            return ConversationResponse(
                reasoning="User has rejected multiple recommendations, transitioning to address concerns",
                message="I'm noticing these recommendations aren't resonating with you. Can you help me understand what's not working? What's most important to you in a career?",
                finished=False
            ), []

        # First rejection - stay in PRESENT but acknowledge
        return None

    async def _handle_concern_intent(
        self,
        intent: UserIntentClassification,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Handle user expressing concerns."""
        # Transition to ADDRESS_CONCERNS phase
        state.conversation_phase = ConversationPhase.ADDRESS_CONCERNS

        self.logger.info("User expressed concern, transitioning to ADDRESS_CONCERNS phase")

        # If we have a concerns handler, immediately invoke it for seamless transition
        if self._concerns_handler:
            self.logger.info("Immediately invoking concerns handler for seamless experience")
            return await self._concerns_handler.handle(user_input, state, context)

        # Fallback: just return transition message (requires another user turn)
        return ConversationResponse(
            reasoning="User expressed a concern, transitioning to CONCERNS phase to address it",
            message="I hear you. Let's talk through that concern.",
            finished=False
        ), []

    async def _handle_accept_intent(
        self,
        intent: UserIntentClassification,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle user accepting/liking a recommendation in PRESENT phase.

        In PRESENT phase, "accept" means "I want to explore this more",
        NOT "I'm ready to commit". So we transition to EXPLORATION, not ACTION.
        But first, we check if we should discuss tradeoffs (user interested in 
        lower-demand occupation when high-demand alternatives exist).
        """
        # Find which occupation they're interested in
        target_occ = None
        if intent.target_recommendation_id:
            target_occ = state.get_recommendation_by_id(intent.target_recommendation_id)
        elif intent.target_occupation_index and state.recommendations:
            idx = intent.target_occupation_index - 1
            occupations = state.recommendations.occupation_recommendations
            if 0 <= idx < len(occupations):
                target_occ = occupations[idx]

        if target_occ:
            # Mark as exploring (not yet fully interested/committed)
            state.current_focus_id = target_occ.uuid
            state.current_recommendation_type = "occupation"
            state.mark_interest(target_occ.uuid, UserInterestLevel.EXPLORING)

            # Check if we should discuss tradeoffs BEFORE exploring
            # (user interested in lower-demand occupation when high-demand alternatives exist)
            if self._should_discuss_tradeoffs(target_occ, state):
                state.conversation_phase = ConversationPhase.DISCUSS_TRADEOFFS
                
                self.logger.info(f"User interested in lower-demand {target_occ.occupation}, transitioning to DISCUSS_TRADEOFFS")
                
                # If we have a tradeoffs handler, immediately invoke it
                if self._tradeoffs_handler:
                    self.logger.info("Immediately invoking tradeoffs handler for seamless experience")
                    return await self._tradeoffs_handler.handle(user_input, state, context)
                
                # Fallback: skip tradeoffs if no handler, proceed to exploration
                self.logger.warning("No tradeoffs handler available, proceeding to EXPLORATION")
                state.conversation_phase = ConversationPhase.CAREER_EXPLORATION

            # Transition to CAREER_EXPLORATION (they want to learn more)
            if state.conversation_phase != ConversationPhase.DISCUSS_TRADEOFFS:
                state.conversation_phase = ConversationPhase.CAREER_EXPLORATION

            self.logger.info(f"User expressed interest in {target_occ.occupation}, transitioning to CAREER_EXPLORATION")

            # If we have an exploration handler, immediately invoke it for seamless transition
            if self._exploration_handler:
                self.logger.info("Immediately invoking exploration handler for seamless experience")
                return await self._exploration_handler.handle(user_input, state, context)

            # Fallback: just return transition message
            return ConversationResponse(
                reasoning="User is interested, transitioning to EXPLORATION phase to learn more",
                message=f"Great! Let me tell you more about {target_occ.occupation}.",
                finished=False
            ), []

        # Couldn't identify which occupation - stay in PRESENT
        self.logger.warning("User accepted but couldn't identify which occupation")
        return None


    async def _call_llm(self, prompt: str, user_input: str, context: ConversationContext) -> tuple[ConversationResponse, list[LLMStats]]:
        """Call LLM with the prepared prompt."""
        try:
            result, llm_stats = await self._conversation_caller.call_llm(
                llm=self._conversation_llm,
                llm_input=ConversationHistoryFormatter.format_for_agent_generative_prompt(
                    model_response_instructions=prompt,
                    context=context,
                    user_input=user_input,
                ),
                logger=self.logger
            )
            return result, llm_stats
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            # Fallback to basic presentation
            return ConversationResponse(
                reasoning="LLM call failed, using fallback presentation",
                message="I have some career recommendations for you. Which would you like to explore first?",
                finished=False
            ), []

    def _should_discuss_tradeoffs(
        self,
        target_occ: OccupationRecommendation,
        state: RecommenderAdvisorAgentState
    ) -> bool:
        """
        Determine if we should discuss tradeoffs before proceeding.

        Returns True if:
        1. User's preferred occupation has lower demand than alternatives
        2. There's a high-demand alternative available
        3. We haven't already discussed tradeoffs for this occupation

        Args:
            target_occ: The occupation the user is interested in
            state: Current agent state

        Returns:
            True if tradeoffs discussion is warranted, False otherwise
        """
        # Must have recommendations to compare
        if not state.recommendations:
            return False

        # If target occupation is already high demand, no tradeoff needed
        if target_occ.labor_demand_category == "high":
            self.logger.debug(f"Target {target_occ.occupation} is high-demand, no tradeoff needed")
            return False

        # Check if we've already discussed tradeoffs for this occupation
        if target_occ.uuid in state.tradeoffs_discussed_for:
            self.logger.debug(f"Already discussed tradeoffs for {target_occ.occupation}")
            return False

        # Check if there's a high-demand alternative with better demand score
        has_better_alternative = False
        preferred_demand = target_occ.labor_demand_score or 0.0

        for occ in state.recommendations.occupation_recommendations:
            # Skip the current target
            if occ.uuid == target_occ.uuid:
                continue

            # Check if this is a high-demand occupation with better score
            if occ.labor_demand_category == "high":
                alt_demand = occ.labor_demand_score or 0.0

                if alt_demand > preferred_demand:
                    has_better_alternative = True
                    self.logger.info(
                        f"Found high-demand alternative: {occ.occupation} "
                        f"(demand: {alt_demand:.2f}) vs preferred {target_occ.occupation} "
                        f"(demand: {preferred_demand:.2f})"
                    )
                    break

        if has_better_alternative:
            # Mark that we'll discuss tradeoffs for this occupation
            state.tradeoffs_discussed_for.append(target_occ.uuid)
            return True

        return False

