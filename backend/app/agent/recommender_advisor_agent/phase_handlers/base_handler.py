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

    async def _handle_request_outside_recommendations(
        self,
        requested_occupation_name: str,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        STRICT GUARDRAIL: Handle when user requests occupation outside recommendations.

        This guardrail politely redirects users back to data-driven recommendations
        without offering to explore the off-list occupation. The response:
        1. Acknowledges their interest
        2. Explains recommendations are based on their skills/preferences
        3. Redirects to comparing requested occupation with available options
        4. Encourages exploration of recommended paths

        Args:
            requested_occupation_name: Name of occupation user requested (e.g., "DJ", "pilot")
            user_input: The user's original message
            state: Current agent state
            context: Conversation context

        Returns:
            Tuple of (ConversationResponse, list of LLMStats)
        """
        all_llm_stats: list[LLMStats] = []

        self.logger.info(f"GUARDRAIL: User requested occupation outside recommendations: '{requested_occupation_name}'")

        # Search for the occupation in taxonomy (optional - to personalize response)
        found_occupation = await self._search_occupation_by_name(requested_occupation_name)

        # Import here to avoid circular dependency
        from app.agent.recommender_advisor_agent.prompts import build_context_block

        # Build context for LLM
        skills_list = self._extract_skills_list(state)
        pref_vec_dict = state.preference_vector.model_dump() if state.preference_vector else {}
        conv_history = ConversationHistoryFormatter.format_to_string(context)

        # Build recommendations summary
        recs_summary = ""
        if state.recommendations and state.recommendations.occupation_recommendations:
            recs_summary = "\n".join([
                f"- {occ.occupation} (match: {occ.confidence_score:.0%})"
                for occ in state.recommendations.occupation_recommendations[:5]
            ])

        context_block = build_context_block(
            skills=skills_list,
            preference_vector=pref_vec_dict,
            recommendations_summary=recs_summary,
            conversation_history=conv_history,
            country_of_user=state.country_of_user
        )

        # Build occupation info if found in taxonomy
        occupation_info = ""
        if found_occupation:
            occupation_info = f"""
**OCCUPATION INFO** (found in database):
- Name: {found_occupation.preferredLabel}
- Description: {found_occupation.description or 'No description available'}
"""

        # Build detailed user profile summary
        user_skills_summary = ", ".join(skills_list) if skills_list else "No specific skills identified"

        # Build preferences summary
        preferences_summary = ""
        if pref_vec_dict:
            key_prefs = []
            for pref_key, pref_value in pref_vec_dict.items():
                if isinstance(pref_value, (int, float)) and pref_value > 0.7:
                    key_prefs.append(f"{pref_key}: {pref_value:.2f}")
            preferences_summary = ", ".join(key_prefs) if key_prefs else "No strong preferences identified"
        else:
            preferences_summary = "No preferences data available"

        prompt = context_block + f"""
## GUARDRAIL: OFF-RECOMMENDATION REQUEST

**USER REQUEST**: The user wants to explore "{requested_occupation_name}" which is NOT in our data-driven recommendations.

{occupation_info}

**USER PROFILE SUMMARY**:
- **Current Skills**: {user_skills_summary}
- **Key Preferences**: {preferences_summary}

**RECOMMENDED OCCUPATIONS** (based on data-driven matching):
{recs_summary}

**YOUR TASK - COMPREHENSIVE COMPARISON & REDIRECT**:
Generate a response that:
1. **Acknowledges their interest warmly** in {requested_occupation_name}
2. **Provides honest, unbiased comparison**:
   - Analyze if {requested_occupation_name} requires skills they DON'T currently have (skills mismatch)
   - Analyze if {requested_occupation_name} aligns with their stated preferences (preferences mismatch)
   - Compare {requested_occupation_name} with the recommended occupations above
   - Be truthful - if there's a mismatch, explain it clearly without bias
3. **Redirect to recommended options**: Encourage exploring the data-driven recommendations that DO match their profile
4. **Keep door open for growth**: Acknowledge they can build toward {requested_occupation_name} in the future

**ANALYSIS REQUIREMENTS**:
- **Skills Mismatch**: If {requested_occupation_name} needs skills like [X, Y, Z] that aren't in their current skillset, SAY SO clearly
  Example: "Being a DJ typically requires skills in music production, mixing, and event promotion - which are different from your current electrical and manual labor skills."

- **Preferences Mismatch**: If {requested_occupation_name} doesn't align with their preferences (e.g., they value stability but DJ work is unstable), SAY SO clearly
  Example: "DJ work tends to be gig-based and unpredictable, which might not align with your preference for job security (0.85)."

- **Be Balanced**: Don't be discouraging, but don't be overly optimistic either. Give them the facts.

**TONE GUIDELINES**:
- Warm and understanding, never dismissive
- Data-driven and honest - show the comparison
- Supportive but realistic
- Frame as "helping you make an informed choice"
- 3-5 sentences (more detail than before)

**EXAMPLE STRUCTURE**:
"I hear you're interested in being a DJ. Being a DJ typically requires [skills they don't have], and the work tends to be [mismatch with preferences]. The careers I recommended - [occupation 1], [occupation 2] - align more closely with your current skills in [X, Y] and your preference for [Z]. These paths offer a stronger foundation for your immediate career growth. If DJ work is still appealing long-term, you could consider building those skills over time."

**CRITICAL**:
- Response must be JSON matching ConversationResponse schema
- Set `finished` to `false` - conversation continues
- Be comprehensive (3-5 sentences) - show the comparison clearly
- Don't offer to explore the off-list occupation now
- Do acknowledge it as a potential future path

""" + get_json_response_instructions()

        # Call LLM to generate guardrail response
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

            self.logger.info(f"GUARDRAIL response generated: {response.message[:100]}...")
            return response, all_llm_stats

        except Exception as e:
            self.logger.error(f"LLM call failed for guardrail response: {e}")
            # Fallback to comprehensive hardcoded guardrail
            skills_str = ", ".join(skills_list[:3]) if skills_list else "your current skills"
            recs_str = ""
            if state.recommendations and state.recommendations.occupation_recommendations:
                top_recs = state.recommendations.occupation_recommendations[:2]
                recs_str = " and ".join([occ.occupation for occ in top_recs])

            return ConversationResponse(
                reasoning=f"GUARDRAIL: User requested {requested_occupation_name} (not in recommendations), LLM failed - using fallback",
                message=f"I hear you're interested in {requested_occupation_name}. {requested_occupation_name} may require different skills and work conditions than what aligns with your current background in {skills_str}. The careers I recommended - {recs_str} - match more closely with your existing skills and preferences. These paths offer a stronger foundation for immediate opportunities. We can explore how the recommended options align with what you're looking for.",
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
