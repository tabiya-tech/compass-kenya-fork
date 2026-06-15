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
from app.conversation_memory.conversation_formatter import ConversationHistoryFormatter
from app.agent.simple_llm_agent.prompt_response_template import get_json_response_instructions
from common_libs.llm.generative_models import GeminiGenerativeLLM
from app.vector_search.esco_entities import OccupationEntity
from app.vector_search.similarity_search_service import SimilaritySearchService


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
        occupation_search_service: Optional[SimilaritySearchService[OccupationEntity]] = None,
        skills_pivot_handler: Optional['SkillsPivotPhaseHandler'] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the phase handler.

        Args:
            conversation_llm: LLM for generating conversational responses
            conversation_caller: Typed LLM caller for response parsing
            occupation_search_service: Optional occupation search service for finding occupations not in recommendations
            skills_pivot_handler: Optional skills pivot handler for immediate delegation when user persists on out-of-list occupation
            logger: Optional logger instance
        """
        self._conversation_llm = conversation_llm
        self._conversation_caller = conversation_caller
        self._occupation_search_service = occupation_search_service
        self._skills_pivot_handler = skills_pivot_handler
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

    async def _search_occupation_by_name(self, occupation_name: str) -> Optional[OccupationEntity]:
        """
        Search occupation taxonomy for an occupation mentioned by the user.

        Uses vector search to find semantically similar occupations in the database.
        This is a shared method used by all phase handlers when users mention
        occupations that aren't in the recommendations list.

        Args:
            occupation_name: The occupation name mentioned by the user (e.g., "DJ", "pilot")

        Returns:
            OccupationEntity if found, None if not found or search fails
        """
        if not self._occupation_search_service:
            self.logger.warning("Occupation search service not available - cannot search for out-of-list occupations")
            return None

        try:
            self.logger.info(f"Searching occupation taxonomy for: '{occupation_name}'")
            results = await self._occupation_search_service.search(
                query=occupation_name,
                k=1  # Just get the top match
            )

            if results and len(results) > 0:
                found_occupation = results[0]
                self.logger.info(
                    f"Found occupation in taxonomy: {found_occupation.preferredLabel} "
                    f"(code: {found_occupation.code}, score: {found_occupation.score:.2f})"
                )
                return found_occupation
            else:
                self.logger.info(f"No occupation found in taxonomy for: '{occupation_name}'")
                return None

        except Exception as e:
            self.logger.error(f"Occupation search failed for '{occupation_name}': {e}")
            return None

    async def _handle_out_of_list_occupation(
        self,
        found_occupation: OccupationEntity,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext,
        recommendations_summary: str
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle case where user mentioned an occupation not in our recommendations.

        Uses LLM to generate a contextual, conversational response that (per Change 2d):
        1. Acknowledges their interest genuinely
        2. Gives an honest, grounded assessment of the gap
        3. Is transparent about scope (no plans/training for off-list) and invites them back
           to the recommendations - it does NOT offer to "explore it anyway"

        This is a shared method used by all phase handlers.

        Args:
            found_occupation: The occupation found in the taxonomy
            user_input: The user's original message
            state: Current agent state
            context: Conversation context
            recommendations_summary: Summary of current recommendations for context

        Returns:
            Tuple of (ConversationResponse, LLMStats)
        """
        all_llm_stats: list[LLMStats] = []

        self.logger.info(f"Generating LLM response for out-of-list occupation: {found_occupation.preferredLabel}")

        # Import here to avoid circular dependency
        from app.agent.recommender_advisor_agent.prompts import (
            build_context_block,
            BASE_RECOMMENDER_PROMPT,
        )

        # Build context for LLM
        skills_list = self._extract_skills_list(state)
        pref_vec_dict = state.preference_vector.model_dump() if state.preference_vector else {}
        conv_history = ConversationHistoryFormatter.format_to_string(context)

        context_block = build_context_block(
            skills=skills_list,
            preference_vector=pref_vec_dict,
            recommendations_summary=recommendations_summary,
            conversation_history=conv_history,
            country_of_user=state.country_of_user
        )

        # Prepend BASE so the "HANDLING USER-SUGGESTED OCCUPATIONS & JOB OPENINGS" protocol
        # (Change 2d) governs this response: honest assessment + transparent scope, NOT an
        # offer to build plans for the off-list occupation.
        prompt = context_block + BASE_RECOMMENDER_PROMPT + f"""
## OUT-OF-LIST OCCUPATION REQUEST

**USER REQUEST**: The user mentioned "{found_occupation.preferredLabel}" which is NOT in our top recommendations.

**OCCUPATION FOUND IN DATABASE**:
- Name: {found_occupation.preferredLabel}
- Code: {found_occupation.code}
- Description: {found_occupation.description or 'No description available'}

**YOUR TASK**:
Follow the "HANDLING USER-SUGGESTED OCCUPATIONS & JOB OPENINGS" protocol above for
"{found_occupation.preferredLabel}":
1. **Acknowledge their interest genuinely** - it's real information about what they value.
2. **Give an honest, grounded assessment** of the gap: compare their skills/preferences (above)
   to what this occupation needs; state market reality if you know it, say so plainly if you
   don't. Don't exaggerate the negatives, don't soften real ones.
3. **Be transparent about your scope**: you help them act on the recommended paths; you do NOT
   build action plans, training paths, or research steps for off-list occupations. If they want
   to pursue it on their own, that's their call, said respectfully.
4. **Invite them back** to the recommendations without pressure - point to any recommended path
   that genuinely shares what draws them to "{found_occupation.preferredLabel}".

Do NOT offer to "explore {found_occupation.preferredLabel} anyway", and do NOT promise to plan
or build steps toward it.

**CRITICAL**:
- Your response must be a JSON object matching ConversationResponse schema
- Set `finished` to `false` - the conversation continues
- Keep the message conversational (2-4 sentences)

""" + get_json_response_instructions()

        # Call LLM to generate contextual response
        try:
            response, llm_stats = await self._conversation_caller.call_llm(
                llm=self._conversation_llm,
                llm_input=ConversationHistoryFormatter.format_for_agent_generative_prompt(
                    model_response_instructions=prompt,
                    context=context,
                    user_input=user_input,
                ),
                logger=self.logger
            )
            all_llm_stats.extend(llm_stats)

            self.logger.info(f"Generated LLM response for out-of-list occupation: {response.message[:100]}...")
            return response, all_llm_stats

        except Exception as e:
            self.logger.error(f"LLM call failed for out-of-list occupation: {e}")
            # Fallback to basic acknowledgment
            return ConversationResponse(
                reasoning=f"User mentioned {found_occupation.preferredLabel} (not in recommendations), LLM failed - using fallback",
                message=f"I can see why {found_occupation.preferredLabel} interests you. It wasn't among my recommendations because those are built on your current skills and the local job market - that's where I can genuinely help you take the next step. Would you like to look at which of the recommended paths comes closest to what draws you to it?",
                finished=False
            ), all_llm_stats

    async def _handle_request_outside_recommendations(
        self,
        requested_occupation_name: str,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle when user requests an occupation outside our recommendations.

        Flow (per Change 2d - the agent stays on-list and never plans toward off-list goals):
        1. First request: honest assessment + transparent scope, invite back to recommendations
        2. If user persists: restate scope once and offer a choice (keep exploring the
           recommended paths, or wrap up) - NO pivot to off-list training/gap analysis

        No vector search needed - we use LLM to explain why based on current recommendations.

        Args:
            requested_occupation_name: Name of occupation user mentioned (e.g., "DJ")
            user_input: User's message
            state: Current agent state
            context: Conversation context

        Returns:
            Tuple of (ConversationResponse, list of LLMStats)
        """
        all_llm_stats: list[LLMStats] = []

        # Check if this is persistence using LLM (not keyword matching!)
        # If there's a pending occupation, we already asked them about it
        # Use LLM to determine if current input is confirming they want to explore it
        if state.pending_out_of_list_occupation:
            is_persistence = await self._is_user_persisting_on_pending_occupation(
                user_input=user_input,
                pending_occupation=state.pending_out_of_list_occupation,
                context=context
            )
        else:
            is_persistence = False

        if is_persistence:
            # Change 2d: the agent stays ON-LIST. On persistence it does NOT build plans or
            # training for the off-list occupation - it restates its scope once and offers a
            # clear choice (keep exploring the recommended paths, or wrap up). The previous
            # behaviour (pivot to SKILLS_UPGRADE_PIVOT gap analysis for the off-list occupation)
            # is intentionally removed; _handle_out_of_list_occupation_gap_analysis in
            # skills_pivot_handler is now unreachable from this path.
            occupation_name = state.pending_out_of_list_occupation
            self.logger.info(
                f"User persisting on out-of-list occupation '{occupation_name}' "
                f"→ restating scope and offering choice (no off-list planning, per Change 2d)"
            )

            # Clear the pending marker so we don't loop on the same occupation.
            state.pending_out_of_list_occupation = None
            state.pending_out_of_list_occupation_entity = None

            return ConversationResponse(
                reasoning=f"User persisted on off-list '{occupation_name}'; restating scope and "
                          f"offering choice (continue with recommendations or wrap up) per Change 2d",
                message=f"I hear you - {occupation_name} clearly matters to you, and pursuing it "
                        f"on your own is completely your call. What I can genuinely help with is "
                        f"acting on the paths built from your skills and the local job market, so "
                        f"I won't build a plan for {occupation_name} here. Would you like to keep "
                        f"exploring the recommended paths together, or wrap up for now?",
                finished=False
            ), all_llm_stats

        # First request - explain why not recommended using LLM
        self.logger.info(f"First out-of-list request: '{requested_occupation_name}'")

        state.pending_out_of_list_occupation = requested_occupation_name

        # Use LLM to explain why it's not recommended (compared to current recs)
        response, llm_stats = await self._explain_why_not_recommended(
            requested_occupation_name=requested_occupation_name,
            user_input=user_input,
            state=state,
            context=context
        )
        all_llm_stats.extend(llm_stats)

        return response, all_llm_stats

    async def _is_user_persisting_on_pending_occupation(
        self,
        user_input: str,
        pending_occupation: str,
        context: ConversationContext
    ) -> bool:
        """
        Use LLM to determine if user is persisting on the pending out-of-list occupation.

        This is better than keyword matching because it understands variations like:
        - "DJ" → "I want to pursue DJing"
        - "DJ, MD" → "show me what it takes for DJing/MD roles"
        - "pilot" → "yes, I want to be a pilot"

        Args:
            user_input: User's current message
            pending_occupation: The occupation we previously asked them about
            context: Conversation context (includes our previous question)

        Returns:
            True if user is confirming/persisting on pending occupation, False otherwise
        """
        # Build a simple prompt for LLM to classify
        from pydantic import BaseModel, Field

        class PersistenceCheck(BaseModel):
            is_persisting: bool = Field(
                description="True if user is confirming they want to explore the pending occupation, False if this is a different/new request"
            )
            reasoning: str = Field(description="Brief explanation of the decision")

        prompt = f"""
You previously asked the user about exploring **"{pending_occupation}"** which was not in their recommendations.

**USER'S RESPONSE**: "{user_input}"

**YOUR TASK**: Determine if the user is confirming/persisting on wanting to explore "{pending_occupation}", or if this is a different/new request.

**EXAMPLES OF PERSISTENCE (return is_persisting=true)**:
- Pending: "DJ" → User: "I want to pursue DJing" → TRUE (same occupation, variation)
- Pending: "DJ, MD" → User: "show me what it takes for DJ/MD roles" → TRUE (same occupations)
- Pending: "pilot" → User: "yes, I want to be a pilot" → TRUE (affirmative about same occupation)
- Pending: "software engineer" → User: "I want to see what software engineering requires" → TRUE

**EXAMPLES OF NEW REQUEST (return is_persisting=false)**:
- Pending: "DJ" → User: "actually, I want to be a teacher" → FALSE (different occupation)
- Pending: "DJ" → User: "tell me about option 1" → FALSE (referring to recommendations)
- Pending: "pilot" → User: "what about the Electrician job?" → FALSE (different occupation)

**REQUIRED OUTPUT FORMAT** (JSON):
{{
    "is_persisting": true,
    "reasoning": "User is confirming interest in DJ/MD roles with variation in phrasing (DJing/MD)"
}}

OR

{{
    "is_persisting": false,
    "reasoning": "User is asking about a different occupation (teacher) not related to pending DJ"
}}
"""

        try:
            # Create a simple LLM caller for this classification
            from app.agent.llm_caller import LLMCaller

            classifier = LLMCaller[PersistenceCheck](
                model_response_type=PersistenceCheck
            )

            result, _ = await classifier.call_llm(
                llm=self._conversation_llm,
                llm_input=prompt,
                logger=self.logger
            )

            self.logger.info(
                f"Persistence check: pending='{pending_occupation}', "
                f"user_input='{user_input}', is_persisting={result.is_persisting}, "
                f"reasoning={result.reasoning}"
            )

            return result.is_persisting

        except Exception as e:
            self.logger.error(f"LLM-based persistence check failed: {e}, defaulting to False")
            # Fallback: assume not persisting (safer than false positive)
            return False

    async def _explain_why_not_recommended(
        self,
        requested_occupation_name: str,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Use LLM to explain why requested occupation isn't in recommendations.

        Compares user's skills/preferences with what the occupation likely requires, then
        (per Change 2d) gives an honest assessment, is transparent about scope, and invites
        the user back to the recommendations - no "binary choice to explore it anyway".

        Args:
            requested_occupation_name: Occupation user mentioned (e.g., "DJ")
            user_input: User's message
            state: Current agent state
            context: Conversation context

        Returns:
            Tuple of (ConversationResponse, list of LLMStats)
        """
        all_llm_stats: list[LLMStats] = []

        # Import here to avoid circular dependency
        from app.agent.recommender_advisor_agent.prompts import (
            build_context_block,
            BASE_RECOMMENDER_PROMPT,
        )

        # Build context for LLM
        skills_list = self._extract_skills_list(state)
        pref_vec_dict = state.preference_vector.model_dump() if state.preference_vector else {}
        conv_history = ConversationHistoryFormatter.format_to_string(context)
        recommendations_summary = self._build_recommendations_summary(state)

        context_block = build_context_block(
            skills=skills_list,
            preference_vector=pref_vec_dict,
            recommendations_summary=recommendations_summary,
            conversation_history=conv_history,
            country_of_user=state.country_of_user
        )

        # Prepend BASE so the off-list protocol (Change 2d) governs this response: honest
        # assessment + transparent scope, NOT a "binary choice to explore it anyway".
        prompt = context_block + BASE_RECOMMENDER_PROMPT + f"""
## OUT-OF-RECOMMENDATIONS REQUEST

The user asked about **"{requested_occupation_name}"** which is NOT in our top recommendations.

**YOUR TASK**:
Follow the "HANDLING USER-SUGGESTED OCCUPATIONS & JOB OPENINGS" protocol above for
"{requested_occupation_name}":

1. **Acknowledge their interest genuinely** - it's real information about what they value.

2. **Give an honest, grounded assessment** of why it's not a recommended path: compare their
   skills/preferences (above) to what "{requested_occupation_name}" needs; state market reality
   if you know it, say so plainly if you don't. Honest but not discouraging.

3. **Be transparent about your scope**: you help them act on the recommended paths; you do NOT
   build action plans, training paths, or research steps for off-list occupations. If they want
   to pursue it independently, that's their call - say so respectfully.

4. **Invite them back** to the recommendations without pressure - point to a recommended path
   that genuinely shares what draws them to "{requested_occupation_name}", if one does.

Do NOT offer a "binary choice to explore {requested_occupation_name} anyway", and do NOT
promise to plan, train for, or build steps toward it.

**REQUIREMENTS**:
- Conversational and supportive; 4-6 sentences maximum
- Response must be JSON matching ConversationResponse schema
- Set `finished` to `false`
- NEVER provide contact information, specific URLs, or addresses - focus only on career guidance

""" + get_json_response_instructions()

        # Call LLM
        try:
            response, llm_stats = await self._conversation_caller.call_llm(
                llm=self._conversation_llm,
                llm_input=ConversationHistoryFormatter.format_for_agent_generative_prompt(
                    model_response_instructions=prompt,
                    context=context,
                    user_input=user_input,
                ),
                logger=self.logger
            )
            all_llm_stats.extend(llm_stats)

            self.logger.info(f"Generated explanation for why '{requested_occupation_name}' not recommended")
            return response, all_llm_stats

        except Exception as e:
            self.logger.error(f"LLM call failed for explaining '{requested_occupation_name}': {e}")
            # Fallback to simple template
            return ConversationResponse(
                reasoning=f"User requested '{requested_occupation_name}' outside recommendations, LLM failed - using fallback",
                message=f"I can see why {requested_occupation_name} interests you. "
                        f"It wasn't in my recommendations because those are built on your current skills and the local job market - that's where I can genuinely help you take a next step. "
                        f"Want to look at which of the recommended paths comes closest to what draws you to it?",
                finished=False
            ), all_llm_stats

    def _build_recommendations_summary(self, state: RecommenderAdvisorAgentState) -> str:
        """
        Build a summary of current recommendations for LLM context.

        Args:
            state: Current agent state

        Returns:
            Summary string of recommendations
        """
        if not state.recommendations:
            return "No recommendations available"

        lines = []
        for occ in state.recommendations.occupation_recommendations[:5]:  # Top 5
            lines.append(f"- {occ.occupation} (rank {occ.rank})")

        return "Current recommendations:\n" + "\n".join(lines)

    def _extract_skills_list(self, state: RecommenderAdvisorAgentState) -> list[str]:
        """
        Extract list of skills from state.skills_vector.

        Shared helper method for building context across all phase handlers.

        Args:
            state: Current agent state

        Returns:
            List of skill names
        """
        if not state.skills_vector:
            return []

        def _as_label(s) -> str:
            # A skill may be a plain string or a dict (e.g. {"preferredLabel": ...}).
            # Always coerce to a string label — downstream context building does
            # ', '.join(skills), which raises if any item is a dict.
            if isinstance(s, dict):
                return s.get("preferredLabel") or s.get("name") or s.get("label") or str(s)
            return str(s)

        # Handle different possible structures
        if isinstance(state.skills_vector, dict):
            # Could be {"skill_name": proficiency_level} or {"skills": [...]}
            if "skills" in state.skills_vector:
                skills = state.skills_vector["skills"]
                return [_as_label(s) for s in skills] if isinstance(skills, list) else []
            elif "top_skills" in state.skills_vector:
                # Handle ExperienceEntity-like structure
                skills = state.skills_vector.get("top_skills", [])
                if isinstance(skills, list) and skills:
                    # Extract skill names
                    return [_as_label(s) for s in skills]
            else:
                # Assume keys are skill names
                return list(state.skills_vector.keys())
        elif isinstance(state.skills_vector, list):
            return [_as_label(s) for s in state.skills_vector]

        return []
