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
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the phase handler.

        Args:
            conversation_llm: LLM for generating conversational responses
            conversation_caller: Typed LLM caller for response parsing
            occupation_search_service: Optional occupation search service for finding occupations not in recommendations
            logger: Optional logger instance
        """
        self._conversation_llm = conversation_llm
        self._conversation_caller = conversation_caller
        self._occupation_search_service = occupation_search_service
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
