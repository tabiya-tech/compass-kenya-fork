"""
Exploration Phase Handler for the Recommender/Advisor Agent.

Handles the CAREER_EXPLORATION phase where we deep-dive into
a specific occupation the user is interested in, using LLM to generate
missing content and create natural, motivational exploration.
"""

from app.agent.agent_types import LLMStats
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    OccupationRecommendation,
    UserInterestLevel
)
from app.agent.recommender_advisor_agent.llm_response_models import (
    ConversationResponse,
    UserIntentClassification
)
from app.agent.recommender_advisor_agent.phase_handlers.base_handler import BasePhaseHandler
from app.agent.recommender_advisor_agent.prompts import (
    CAREER_EXPLORATION_PROMPT,
    build_context_block
)
from app.agent.recommender_advisor_agent.intent_classifier import IntentClassifier
from app.conversation_memory.conversation_memory_manager import ConversationContext
from app.conversation_memory.conversation_formatter import ConversationHistoryFormatter
from app.agent.simple_llm_agent.prompt_response_template import get_json_response_instructions


class ExplorationPhaseHandler(BasePhaseHandler):
    """
    Handles the CAREER_EXPLORATION phase.

    Responsibilities:
    - Show detailed information about the selected occupation
    - Connect occupation to user's preferences
    - Identify skill gaps and career progression
    - Prompt for concerns or interest
    - Detect intent and transition to appropriate phases (ADDRESS_CONCERNS, ACTION_PLANNING, etc.)
    """

    def __init__(
        self,
        conversation_llm,
        conversation_caller,
        intent_classifier: IntentClassifier = None,
        concerns_handler: 'ConcernsPhaseHandler' = None,
        action_handler: 'ActionPhaseHandler' = None,
        tradeoffs_handler: 'TradeoffsPhaseHandler' = None,
        **kwargs
    ):
        """
        Initialize the exploration handler.

        Args:
            conversation_llm: LLM for generating responses
            conversation_caller: LLM caller for conversation responses
            intent_classifier: Optional intent classifier for detecting user intent
            concerns_handler: Optional concerns handler for immediate transitions
            action_handler: Optional action handler for immediate transitions
            tradeoffs_handler: Optional tradeoffs handler for immediate transitions
        """
        super().__init__(conversation_llm, conversation_caller, **kwargs)
        self._intent_classifier = intent_classifier
        self._concerns_handler = concerns_handler
        self._action_handler = action_handler
        self._tradeoffs_handler = tradeoffs_handler

    async def handle(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle deep exploration of a specific occupation using LLM.
        Also detects user intent and transitions to appropriate phase.
        """
        all_llm_stats: list[LLMStats] = []

        if state.current_focus_id is None:
            # No occupation selected, go back to presentation
            self.logger.warning("No occupation selected for exploration")
            state.conversation_phase = ConversationPhase.PRESENT_RECOMMENDATIONS
            return ConversationResponse(
                reasoning="No occupation selected, returning to presentation",
                message="Which occupation would you like to learn more about?",
                finished=False
            ), []

        # Get the focused occupation
        rec = state.get_recommendation_by_id(state.current_focus_id)

        if rec is None or not isinstance(rec, OccupationRecommendation):
            return ConversationResponse(
                reasoning="Could not find selected occupation",
                message="I couldn't find that occupation. Would you like to see the options again?",
                finished=False
            ), []

        # Mark as explored (only on first exploration)
        is_initial_exploration = rec.uuid not in state.explored_items
        if is_initial_exploration:
            state.explored_items.append(rec.uuid)

        # If this is not the initial exploration, classify user intent
        if not is_initial_exploration and user_input.strip() != "" and self._intent_classifier:
            self.logger.info(f"Classifying user intent during exploration: '{user_input}'")
            intent, intent_stats = await self._intent_classifier.classify_intent(
                user_input=user_input,
                state=state,
                context=context,
                phase=ConversationPhase.CAREER_EXPLORATION,
                llm=self._conversation_llm,
                logger=self.logger
            )
            all_llm_stats.extend(intent_stats)

            # Check if intent classification succeeded
            if intent is not None:
                self.logger.info(f"Intent classified as: {intent.intent}, reasoning: {intent.reasoning}")
                self.logger.info(f"Intent target_recommendation_id: {intent.target_recommendation_id}")
                self.logger.info(f"Intent target_occupation_index: {intent.target_occupation_index}")
                self.logger.info(f"Intent requested_occupation_name: {intent.requested_occupation_name}")

                # Also print to console for debugging
                print(f"\n{'='*80}")
                print(f"[INTENT CLASSIFIER DEBUG]")
                print(f"User Input: {user_input}")
                print(f"Current Focus ID: {state.current_focus_id}")
                print(f"Classified Intent: {intent.intent}")
                print(f"Reasoning: {intent.reasoning}")
                print(f"Target Recommendation ID: {intent.target_recommendation_id}")
                print(f"Target Occupation Index: {intent.target_occupation_index}")
                print(f"Requested Occupation Name: {intent.requested_occupation_name}")
                print(f"{'='*80}\n")

                # Handle the intent and potentially transition phases
                phase_transition = await self._handle_user_intent(intent, user_input, state, context)

                if phase_transition:
                    # Intent handled, phase transition will occur
                    all_llm_stats.extend(phase_transition[1])
                    return phase_transition[0], all_llm_stats

        # Build occupation summary for LLM
        occ_summary = self._build_occupation_summary(rec)
        
        # Build list of ALL available occupations with UUIDs for occupation switching
        all_occupations_list = ""
        if state.recommendations:
            occ_lines = ["\n\n**ALL AVAILABLE OCCUPATIONS (with UUIDs for discussed_occupation_id)**:"]
            for i, occ in enumerate(state.recommendations.occupation_recommendations[:5], 1):
                occ_lines.append(f"{i}. {occ.occupation} (uuid: {occ.uuid})")
            all_occupations_list = "\n".join(occ_lines)

        # Build full context for LLM
        skills_list = self._extract_skills_list(state)
        pref_vec_dict = state.preference_vector.model_dump() if state.preference_vector else {}
        conv_history = ConversationHistoryFormatter.format_to_string(context)

        # Add occupation details to context
        context_block = build_context_block(
            skills=skills_list,
            preference_vector=pref_vec_dict,
            recommendations_summary=f"Currently exploring: {occ_summary}{all_occupations_list}",
            conversation_history=conv_history,
            country_of_user=state.country_of_user
        )

        # Create example response with discussed_occupation_id to ensure LLM knows about this field
        example_response = ConversationResponse(
            reasoning="The user expressed interest in Fundi wa Stima (Electrician), so I will discuss that occupation and set discussed_occupation_id to track the switch.",
            message="Let's explore what being a Fundi wa Stima (Electrician) involves...",
            finished=False,
            discussed_occupation_id="occ_001_uuid"  # Example UUID - LLM should use actual UUID from list
        )

        # Build prompt for LLM
        full_prompt = context_block + CAREER_EXPLORATION_PROMPT + get_json_response_instructions(examples=[example_response])

        # Call LLM to generate exploration
        response, llm_stats = await self._call_llm(full_prompt, user_input, context, rec)
        all_llm_stats.extend(llm_stats)

        # === RESPONSE-BASED FOCUS UPDATE ===
        # If the LLM discussed a different occupation than our current focus,
        # update the state to reflect the actual occupation being discussed.
        # This handles occupation switches that the intent classifier might have missed.
        if response.discussed_occupation_id and response.discussed_occupation_id != state.current_focus_id:
            new_focus = state.get_recommendation_by_id(response.discussed_occupation_id)
            if new_focus:
                self.logger.info(
                    f"LLM discussed different occupation: {new_focus.occupation} (uuid: {response.discussed_occupation_id}). "
                    f"Updating focus from {state.current_focus_id} to {response.discussed_occupation_id}"
                )
                state.current_focus_id = response.discussed_occupation_id
                state.current_recommendation_type = "occupation"
                state.mark_interest(response.discussed_occupation_id, UserInterestLevel.EXPLORING)
                
                # Update metadata to reflect the actual occupation discussed
                rec = new_focus
            else:
                self.logger.warning(f"Could not find occupation with UUID: {response.discussed_occupation_id}")

        # Add metadata
        response.metadata = self._build_metadata(
            interaction_type="occupation_exploration",
            occupation_id=rec.uuid,
            occupation=rec.occupation,
            confidence_score=rec.confidence_score,
            skills_match_score=rec.skills_match_score,
            preference_match_score=rec.preference_match_score,
            labor_demand_score=rec.labor_demand_score,
        )

        return response, all_llm_stats

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
        self.logger.info(f"Handling intent during exploration: {intent.intent}")

        print(f"\n{'='*80}")
        print(f"[_handle_user_intent DEBUG]")
        print(f"Intent: {intent.intent}")
        print(f"User Input: {user_input}")
        print(f"Current Focus: {state.current_focus_id}")
        print(f"Target Rec ID: {intent.target_recommendation_id}")
        print(f"Target Occ Index: {intent.target_occupation_index}")
        print(f"{'='*80}\n")

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

        # Handle CONCERN intent - user expressing worries/doubts
        elif intent.intent == "express_concern":
            # Transition to ADDRESS_CONCERNS phase
            state.conversation_phase = ConversationPhase.ADDRESS_CONCERNS

            self.logger.info("User expressed concern during exploration, transitioning to ADDRESS_CONCERNS")

            # If we have a concerns handler, immediately invoke it for seamless transition
            if self._concerns_handler:
                self.logger.info("Immediately invoking concerns handler for seamless experience")
                return await self._concerns_handler.handle(user_input, state, context)

            # Fallback: just return transition message
            return ConversationResponse(
                reasoning="User expressed concern during exploration, transitioning to CONCERNS phase",
                message="I hear your concern. Let's talk through that.",
                finished=False
            ), []

        # Handle ACCEPT intent - user is ready to move forward
        elif intent.intent == "accept":
            # Mark as interested
            if state.current_focus_id:
                state.mark_interest(state.current_focus_id, UserInterestLevel.INTERESTED)

            # Transition directly to ACTION_PLANNING
            # Note: Tradeoffs are already discussed in PRESENT_RECOMMENDATIONS phase
            # No need to discuss again here
            state.conversation_phase = ConversationPhase.ACTION_PLANNING
            self.logger.info("User accepted occupation, transitioning to ACTION_PLANNING")

            # Invoke action handler if available
            if self._action_handler:
                self.logger.info("Immediately invoking action handler for seamless experience")
                return await self._action_handler.handle(user_input, state, context)

            # Fallback: just return transition message
            return ConversationResponse(
                reasoning="User is ready to move forward, transitioning to ACTION phase",
                message="That's great! Let's talk about your next steps and how to move forward with this path.",
                finished=False
            ), []

        # Handle REJECT intent - user doesn't want this occupation
        elif intent.intent == "reject":
            # Mark as rejected
            if state.current_focus_id:
                state.mark_interest(state.current_focus_id, UserInterestLevel.REJECTED)

            # Go back to PRESENT_RECOMMENDATIONS
            state.conversation_phase = ConversationPhase.PRESENT_RECOMMENDATIONS
            state.current_focus_id = None
            state.current_recommendation_type = None

            self.logger.info("User rejected occupation, returning to PRESENT_RECOMMENDATIONS")

            return ConversationResponse(
                reasoning="User rejected occupation, returning to presentation",
                message="No problem. Let's look at the other options.",
                finished=False
            ), []

        # Handle EXPLORE_DIFFERENT - user wants to explore a different occupation
        elif intent.intent == "explore_different":
            print(f"\n{'='*80}")
            print(f"[EXPLORE_DIFFERENT HANDLER]")
            print(f"target_recommendation_id: {intent.target_recommendation_id}")
            print(f"target_occupation_index: {intent.target_occupation_index}")
            print(f"{'='*80}\n")

            if intent.target_recommendation_id or intent.target_occupation_index:
                # Find the new occupation
                target_occ = None
                if intent.target_occupation_index and state.recommendations:
                    idx = intent.target_occupation_index - 1
                    occupations = state.recommendations.occupation_recommendations
                    print(f"[DEBUG] Trying to find occupation at index {idx} (0-based)")
                    if 0 <= idx < len(occupations):
                        target_occ = occupations[idx]
                        print(f"[DEBUG] Found occupation: {target_occ.occupation} (uuid: {target_occ.uuid})")
                    else:
                        print(f"[DEBUG] Index {idx} out of range (total: {len(occupations)})")

                elif intent.target_recommendation_id:
                    print(f"[DEBUG] Looking up by UUID: {intent.target_recommendation_id}")
                    target_occ = state.get_recommendation_by_id(intent.target_recommendation_id)
                    if target_occ:
                        print(f"[DEBUG] Found occupation: {target_occ.occupation}")
                    else:
                        print(f"[DEBUG] No occupation found with UUID: {intent.target_recommendation_id}")

                if target_occ:
                    # Switch focus to new occupation
                    print(f"[DEBUG] SWITCHING FOCUS from {state.current_focus_id} to {target_occ.uuid}")
                    state.current_focus_id = target_occ.uuid
                    state.mark_interest(target_occ.uuid, UserInterestLevel.EXPLORING)

                    self.logger.info(f"User wants to explore different occupation: {target_occ.occupation}")

                    # Re-invoke this handler with empty input to trigger exploration of new occupation
                    return await self.handle("", state, context)
                else:
                    print(f"[DEBUG] target_occ is None - could not find occupation!")
            else:
                print(f"[DEBUG] Neither target_recommendation_id nor target_occupation_index was set!")

            # Couldn't identify which occupation in our recommendations
            # Try searching the occupation taxonomy for occupations not in our list
            self.logger.info(f"Could not identify target occupation in recommendations. Searching taxonomy for: '{user_input}'")

            # Search for the occupation in the taxonomy
            mentioned_occ = await self._search_occupation_by_name(user_input)

            if mentioned_occ:
                # Found an occupation in taxonomy that's not in our recommendations
                # Generate LLM-driven response to handle this gracefully (using base class method)
                occ_summary = "Currently exploring occupations" if not state.recommendations else self._build_occupation_summary_for_context(state)
                return await self._handle_out_of_list_occupation(mentioned_occ, user_input, state, context, occ_summary)

            # Couldn't find the occupation anywhere - go back to presentation
            state.conversation_phase = ConversationPhase.PRESENT_RECOMMENDATIONS
            state.current_focus_id = None

            return ConversationResponse(
                reasoning="User wants different occupation but unclear which, returning to presentation",
                message="Which occupation would you like to explore instead?",
                finished=False
            ), []

        # For CONTINUE_EXPLORING, ASK_QUESTION, or OTHER - stay in exploration, return None
        # to let the LLM handle it conversationally
        # The LLM has instructions in the base prompt to handle user-suggested occupations
        return None
    
    def _build_occupation_summary(self, occ: OccupationRecommendation) -> str:
        """Build a detailed summary of the occupation for LLM context."""
        lines = [f"**{occ.occupation}**"]
        lines.append(f"UUID: {occ.uuid}")  # Include UUID for occupation tracking
        lines.append(f"Occupation Code: {occ.occupation_code}")
        lines.append(f"Confidence Score: {occ.confidence_score:.0%}")

        # Score breakdowns
        if occ.skills_match_score is not None:
            lines.append(f"Skills Match: {occ.skills_match_score:.0%}")
        if occ.preference_match_score is not None:
            lines.append(f"Preference Match: {occ.preference_match_score:.0%}")
        if occ.labor_demand_score is not None:
            lines.append(f"Labor Demand Score: {occ.labor_demand_score:.0%}")

        # Labor demand category
        if occ.labor_demand_category:
            lines.append(f"Labor Demand Category: {occ.labor_demand_category}")

        # Salary
        if occ.salary_range:
            lines.append(f"Salary Range: {occ.salary_range}")

        # Description
        if occ.description:
            lines.append(f"Description: {occ.description}")

        # Typical tasks (if provided)
        if occ.typical_tasks:
            lines.append(f"Typical Tasks (PROVIDED): {', '.join(occ.typical_tasks)}")
        else:
            lines.append("Typical Tasks: NOT PROVIDED - LLM should generate realistic tasks")

        # Career path (if provided)
        if occ.career_path_next_steps:
            lines.append(f"Career Path (PROVIDED): {' → '.join(occ.career_path_next_steps)}")
        else:
            lines.append("Career Path: NOT PROVIDED - LLM should generate realistic progression")

        # Skills
        lines.append(f"Essential Skills: {', '.join(occ.essential_skills)}")

        if occ.skill_gaps:
            lines.append(f"Skill Gaps: {', '.join(occ.skill_gaps)}")

        if occ.user_skill_coverage > 0:
            lines.append(f"User Skill Coverage: {occ.user_skill_coverage:.0%}")

        # Justification
        if occ.justification:
            lines.append(f"Justification: {occ.justification}")

        return '\n'.join(lines)

    def _build_occupation_summary_for_context(self, state: RecommenderAdvisorAgentState) -> str:
        """Build occupation summary for context when handling out-of-list occupations."""
        if not state.recommendations:
            return "No recommendations available"

        # Get current occupation if exploring
        if state.current_focus_id:
            rec = state.get_recommendation_by_id(state.current_focus_id)
            if rec and isinstance(rec, OccupationRecommendation):
                return f"Currently exploring: {self._build_occupation_summary(rec)}"

        # Fall back to list of recommendations
        lines = ["Available occupations:"]
        for i, occ in enumerate(state.recommendations.occupation_recommendations[:5], 1):
            lines.append(f"{i}. {occ.occupation}")
        return '\n'.join(lines)

    async def _call_llm(
        self,
        prompt: str,
        user_input: str,
        context: ConversationContext,
        occ: OccupationRecommendation
    ) -> tuple[ConversationResponse, list[LLMStats]]:
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
            # Fallback to basic exploration
            fallback_message = f"""Let's explore **{occ.occupation}**:

**What this involves**: {occ.description if occ.description else 'A role focused on ' + occ.occupation.lower()}

**Why this matches you**: {occ.justification if occ.justification else 'Based on your skills and preferences'}

**What concerns do you have about this path?**"""

            return ConversationResponse(
                reasoning="LLM call failed, using fallback exploration",
                message=fallback_message,
                finished=False
            ), []

    def _should_discuss_tradeoffs(self, state: RecommenderAdvisorAgentState) -> bool:
        """
        Determine if we should discuss tradeoffs before proceeding to action.

        Returns True if:
        1. User's preferred occupation has lower demand than alternatives
        2. There's a high-demand alternative available
        3. We haven't already discussed tradeoffs for this occupation

        Args:
            state: Current agent state

        Returns:
            True if tradeoffs discussion is warranted, False otherwise
        """
        # Must have current focus and recommendations
        if not state.current_focus_id or not state.recommendations:
            return False

        # Get the user's preferred occupation
        preferred_occ = state.get_recommendation_by_id(state.current_focus_id)
        if not preferred_occ or not isinstance(preferred_occ, OccupationRecommendation):
            return False

        # If preferred occupation is already high demand, no tradeoff needed
        if preferred_occ.labor_demand_category == "high":
            self.logger.debug(f"Target {preferred_occ.occupation} is high-demand, no tradeoff needed")
            return False

        # Check if we've already discussed tradeoffs for this occupation
        if preferred_occ.uuid in state.tradeoffs_discussed_for:
            self.logger.debug(f"Already discussed tradeoffs for {preferred_occ.occupation}")
            return False

        # Check if there's a high-demand alternative with better demand score
        has_better_alternative = False
        preferred_demand = preferred_occ.labor_demand_score or 0.0

        for occ in state.recommendations.occupation_recommendations:
            # Skip the current focus
            if occ.uuid == state.current_focus_id:
                continue

            # Check if this is a high-demand occupation
            if occ.labor_demand_category == "high":
                alt_demand = occ.labor_demand_score or 0.0

                if alt_demand > preferred_demand:
                    has_better_alternative = True
                    self.logger.info(
                        f"Found high-demand alternative: {occ.occupation} "
                        f"(demand: {alt_demand:.2f}) vs preferred {preferred_occ.occupation} "
                        f"(demand: {preferred_demand:.2f})"
                    )
                    break

        if has_better_alternative:
            # Mark that we'll discuss tradeoffs for this occupation
            state.tradeoffs_discussed_for.append(preferred_occ.uuid)
            return True

        return False

