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
        **kwargs
    ):
        """
        Initialize the exploration handler.

        Args:
            conversation_llm: LLM for generating responses
            conversation_caller: LLM caller for conversation responses
            intent_classifier: Optional intent classifier for detecting user intent
            concerns_handler: Optional concerns handler for immediate transitions
        """
        super().__init__(conversation_llm, conversation_caller, **kwargs)
        self._intent_classifier = intent_classifier
        self._concerns_handler = concerns_handler

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

                # Handle the intent and potentially transition phases
                phase_transition = await self._handle_user_intent(intent, user_input, state, context)

                if phase_transition:
                    # Intent handled, phase transition will occur
                    all_llm_stats.extend(phase_transition[1])
                    return phase_transition[0], all_llm_stats

        # Build occupation summary for LLM
        occ_summary = self._build_occupation_summary(rec)

        # Build full context for LLM
        skills_list = self._extract_skills_list(state)
        pref_vec_dict = state.preference_vector.model_dump() if state.preference_vector else {}
        conv_history = ConversationHistoryFormatter.format_to_string(context)

        # Add occupation details to context
        context_block = build_context_block(
            skills=skills_list,
            preference_vector=pref_vec_dict,
            recommendations_summary=f"Currently exploring: {occ_summary}",
            conversation_history=conv_history,
            country_of_user=state.country_of_user
        )

        # Build prompt for LLM
        full_prompt = context_block + CAREER_EXPLORATION_PROMPT + get_json_response_instructions()

        # Call LLM to generate exploration
        response, llm_stats = await self._call_llm(full_prompt, user_input, context, rec)
        all_llm_stats.extend(llm_stats)

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

        # Handle CONCERN intent - user expressing worries/doubts
        if intent.intent == "express_concern":
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
            # Mark as interested and transition to ACTION_PLANNING
            if state.current_focus_id:
                state.mark_interest(state.current_focus_id, UserInterestLevel.INTERESTED)

            state.conversation_phase = ConversationPhase.ACTION_PLANNING

            self.logger.info("User accepted occupation, transitioning to ACTION_PLANNING")

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
            if intent.target_recommendation_id or intent.target_occupation_index:
                # Find the new occupation
                target_occ = None
                if intent.target_occupation_index and state.recommendations:
                    idx = intent.target_occupation_index - 1
                    occupations = state.recommendations.occupation_recommendations
                    if 0 <= idx < len(occupations):
                        target_occ = occupations[idx]

                elif intent.target_recommendation_id:
                    target_occ = state.get_recommendation_by_id(intent.target_recommendation_id)

                if target_occ:
                    # Switch focus to new occupation
                    state.current_focus_id = target_occ.uuid
                    state.mark_interest(target_occ.uuid, UserInterestLevel.EXPLORING)

                    self.logger.info(f"User wants to explore different occupation: {target_occ.occupation}")

                    # Re-invoke this handler with empty input to trigger exploration of new occupation
                    return await self.handle("", state, context)

            # Couldn't identify which occupation - go back to presentation
            state.conversation_phase = ConversationPhase.PRESENT_RECOMMENDATIONS
            state.current_focus_id = None

            return ConversationResponse(
                reasoning="User wants different occupation but unclear which, returning to presentation",
                message="Which occupation would you like to explore instead?",
                finished=False
            ), []

        # For CONTINUE_EXPLORING, ASK_QUESTION, or OTHER - stay in exploration, return None
        # to let the LLM handle it conversationally
        return None
    
    def _build_occupation_summary(self, occ: OccupationRecommendation) -> str:
        """Build a detailed summary of the occupation for LLM context."""
        lines = [f"**{occ.occupation}**"]
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

    def _extract_skills_list(self, state: RecommenderAdvisorAgentState) -> list[str]:
        """Extract list of skills from state.skills_vector."""
        if not state.skills_vector:
            return []

        # Handle different possible structures
        if isinstance(state.skills_vector, dict):
            # Could be {"skill_name": proficiency_level} or {"skills": [...]}
            if "skills" in state.skills_vector:
                return state.skills_vector["skills"]
            elif "top_skills" in state.skills_vector:
                # Handle ExperienceEntity-like structure
                skills = state.skills_vector.get("top_skills", [])
                if isinstance(skills, list) and skills:
                    # Extract skill names
                    return [s.get("preferredLabel", s.get("name", str(s))) if isinstance(s, dict) else str(s) for s in skills]
            else:
                # Assume keys are skill names
                return list(state.skills_vector.keys())
        elif isinstance(state.skills_vector, list):
            return state.skills_vector

        return []

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
