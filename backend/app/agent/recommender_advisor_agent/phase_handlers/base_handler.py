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

        Uses LLM to generate a contextual, conversational response that:
        1. Acknowledges we found the occupation
        2. Explains why it wasn't in recommendations (if possible)
        3. Offers choices: explore it anyway, understand alternatives, or see similar options

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
        from app.agent.recommender_advisor_agent.prompts import build_context_block

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

        # Build prompt for handling out-of-list occupation
        prompt = context_block + f"""
## OUT-OF-LIST OCCUPATION REQUEST

**USER REQUEST**: The user mentioned "{found_occupation.preferredLabel}" which is NOT in our top recommendations.

**OCCUPATION FOUND IN DATABASE**:
- Name: {found_occupation.preferredLabel}
- Code: {found_occupation.code}
- Description: {found_occupation.description or 'No description available'}

**YOUR TASK**:
The user wants to explore "{found_occupation.preferredLabel}", but it's not among our top recommendations based on their skills and preferences.

Generate a response that:
1. **Acknowledges** you found this occupation in our database
2. **Briefly explains** why it likely wasn't in the top recommendations:
   - Check if their skills seem relevant (you have their skills list above)
   - Consider if it matches their preferences
   - You can infer general market demand if you have knowledge of this occupation
   - Be honest but not discouraging
3. **Offers a choice**:
   - "Would you like to explore {found_occupation.preferredLabel} anyway?" (respect their autonomy)
   - "Would you like to understand why I recommended these alternatives instead?"
   - "There might be similar occupations in my recommendations - want to see if any overlap?"

**TONE GUIDELINES**:
- Be conversational and natural, not robotic
- Don't make them feel bad for asking about this occupation
- Frame it as expanding their options, not shutting them down
- Acknowledge their interest: "I can see why {found_occupation.preferredLabel} appeals to you"
- Be truthful but supportive
- Respect their autonomy - if they want to explore it, that's valid

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
                message=f"I found {found_occupation.preferredLabel} in our database. While it wasn't among my top recommendations based on your profile, I'm happy to explore it with you if you're interested. Would you like to learn more about it, or hear why I suggested the alternatives instead?",
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

        Flow:
        1. First request: Explain why it's not recommended, offer controlled binary choice
        2. If user persists: Transition to SKILLS_UPGRADE_PIVOT for gap analysis

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
            # User is persisting → Transition to SKILLS_UPGRADE_PIVOT for gap analysis
            self.logger.info(
                f"User persisting on out-of-list occupation '{state.pending_out_of_list_occupation}' "
                f"→ Transitioning to SKILLS_UPGRADE_PIVOT for gap analysis"
            )

            from app.agent.recommender_advisor_agent.types import ConversationPhase

            state.conversation_phase = ConversationPhase.SKILLS_UPGRADE_PIVOT
            state.pivoted_to_training = True

            # Immediately delegate to skills_pivot_handler for seamless experience
            if self._skills_pivot_handler:
                self.logger.info("Immediately invoking skills_pivot_handler for seamless experience")
                return await self._skills_pivot_handler.handle(user_input, state, context)

            # Fallback: just return transition message (requires another user turn)
            occupation_name = state.pending_out_of_list_occupation
            return ConversationResponse(
                reasoning=f"User persisted on '{occupation_name}', pivoting to skills gap analysis (no handler available)",
                message=f"I understand {occupation_name} is important to you. "
                        f"Let me help you understand what it would take to pursue this path and show you relevant training options.",
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

        Compares user's skills/preferences with what the occupation likely requires,
        then offers controlled binary choice.

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
        from app.agent.recommender_advisor_agent.prompts import build_context_block

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

        # Build prompt for explaining why occupation is not recommended
        prompt = context_block + f"""
## OUT-OF-RECOMMENDATIONS REQUEST

The user asked about **"{requested_occupation_name}"** which is NOT in our top recommendations.

**YOUR TASK**:
Generate a response that:

1. **Acknowledges their interest** (1 sentence)
   - "I understand {requested_occupation_name} interests you."

2. **Briefly explains why it's not in the top recommendations** (2-3 sentences max)
   - Compare their current skills to what {requested_occupation_name} likely requires
   - Consider if it matches their stated preferences (stable income, etc.)
   - You can use your knowledge of {requested_occupation_name} to infer skill/market gaps
   - Be honest but respectful - don't discourage them

3. **Offer controlled binary choice** (1 sentence) - CRITICAL: Must be a clear either/or choice
   - "Would you still like to explore {requested_occupation_name}, or shall we dive deeper into these recommendations?"
   - OR "Would you like to see what it would take to pursue {requested_occupation_name}, or explore why I recommended these alternatives?"

**TONE**:
- Conversational and supportive, not robotic
- Respectful of their autonomy
- Honest about gaps but not discouraging
- Total length: 4-6 sentences maximum

**CRITICAL REQUIREMENTS**:
- End with a BINARY CHOICE question (not open-ended)
- Do NOT ask "What appeals to you about DJ?" (too open-ended, allows derailing)
- Do NOT offer 3+ options (keeps it simple)
- Response must be JSON matching ConversationResponse schema
- Set `finished` to `false`
- NEVER provide contact information, specific URLs, or addresses - focus only on career guidance

**REQUIRED OUTPUT FORMAT** (JSON):
{{
    "reasoning": "User requested DJ which requires different skills than their electrical background and has variable income",
    "message": "I understand DJ interests you. However, it requires music production and sound engineering skills, which are quite different from your current electrical and manual labor experience. Also, DJ work typically has irregular income, which may not align with your preference for stability. Would you still like to explore what it takes to become a DJ, or shall we dive deeper into these recommendations?",
    "finished": false
}}

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
                message=f"I understand you're interested in {requested_occupation_name}. "
                        f"It wasn't in my top recommendations because your current skills and preferences align better with the options I showed you. "
                        f"Would you still like to explore {requested_occupation_name}, or shall we dive deeper into these recommendations?",
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
