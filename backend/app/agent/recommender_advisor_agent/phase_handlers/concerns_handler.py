"""
Concerns Phase Handler for the Recommender/Advisor Agent.

Handles the ADDRESS_CONCERNS phase where we respond to user
resistance and objections using appropriate strategies.

Uses two-step LLM process:
1. Classify resistance type
2. Generate appropriate response based on classification
"""

from app.agent.agent_types import LLMStats
from app.agent.llm_caller import LLMCaller
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    ConcernRecord,
    ResistanceType,
)
from app.agent.recommender_advisor_agent.llm_response_models import (
    ConversationResponse,
    ResistanceClassification,
)
from app.agent.recommender_advisor_agent.phase_handlers.base_handler import BasePhaseHandler
from app.agent.recommender_advisor_agent.prompts import (
    ADDRESS_CONCERNS_PROMPT_CLASSIFICATION,
    ADDRESS_CONCERNS_PROMPT_RESPONSE,
    build_context_block
)
from app.agent.simple_llm_agent.prompt_response_template import get_json_response_instructions
from app.conversation_memory.conversation_formatter import ConversationHistoryFormatter
from app.conversation_memory.conversation_memory_manager import ConversationContext
from common_libs.llm.generative_models import GeminiGenerativeLLM


class ConcernsPhaseHandler(BasePhaseHandler):
    """
    Handles the ADDRESS_CONCERNS phase.
    
    Responsibilities:
    - Classify the type of resistance (belief, salience, effort, etc.)
    - Generate appropriate responses without manipulation
    - Track concerns raised and addressed
    - Know when to transition to action planning or pivot
    """
    
    def __init__(
        self,
        conversation_llm: GeminiGenerativeLLM,
        conversation_caller: LLMCaller[ConversationResponse],
        resistance_caller: LLMCaller[ResistanceClassification],
        **kwargs
    ):
        """
        Initialize the concerns handler.
        
        Args:
            resistance_caller: LLM caller for resistance classification
        """
        super().__init__(conversation_llm, conversation_caller, **kwargs)
        self._resistance_caller = resistance_caller
    
    async def handle(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle addressing user concerns/resistance.
        """
        all_llm_stats: list[LLMStats] = []

        # Classify the resistance type
        classification, llm_stats = await self._classify_resistance(user_input, context)
        all_llm_stats.extend(llm_stats)

        # Check if classification failed
        if classification is None:
            self.logger.error("Resistance classification failed after all retries, using fallback")
            # Fallback: treat as generic concern
            concern = ConcernRecord(
                item_id=state.current_focus_id or "unknown",
                item_type=state.current_recommendation_type,
                concern=user_input,
                resistance_type=ResistanceType.BELIEF_BASED  # Default to belief-based
            )
            state.add_concern(concern)

            return ConversationResponse(
                reasoning="Classification failed, providing generic supportive response",
                message="I hear your concern. Can you tell me more about what specifically worries you? That will help me address it better.",
                finished=False
            ), all_llm_stats

        # If no resistance, transition to action planning
        if classification.resistance_type == "none":
            state.conversation_phase = ConversationPhase.ACTION_PLANNING
            return ConversationResponse(
                reasoning="No resistance detected, moving to action planning",
                message="Great! It sounds like this path interests you. Let's talk about next steps.",
                finished=False
            ), all_llm_stats
        
        # Record the concern
        concern = ConcernRecord(
            item_id=state.current_focus_id or "unknown",
            item_type=state.current_recommendation_type,
            concern=classification.concern_summary,
            resistance_type=ResistanceType(classification.resistance_type)
        )
        state.add_concern(concern)
        
        # Generate response based on resistance type
        response, llm_stats = await self._generate_response(
            classification, user_input, state, context
        )
        all_llm_stats.extend(llm_stats)
        
        return response, all_llm_stats
    
    async def _classify_resistance(
        self,
        user_input: str,
        context: ConversationContext
    ) -> tuple[ResistanceClassification, list[LLMStats]]:
        """
        Classify the type of user resistance using comprehensive prompt.

        Step 1 of 2-step process.
        """
        # Use the comprehensive classification prompt with proper schema instructions
        schema_instructions = """
Your response must be a JSON object with the following schema:
{
    "reasoning": "Step by step explanation of what type of resistance this is",
    "resistance_type": "One of: belief, salience, effort, financial, circumstantial, none",
    "concern_summary": "Brief summary of the user's concern"
}

Always return a valid JSON object matching this exact schema.
"""

        full_prompt = ADDRESS_CONCERNS_PROMPT_CLASSIFICATION + "\n\n" + schema_instructions

        return await self._resistance_caller.call_llm(
            llm=self._conversation_llm,
            llm_input=ConversationHistoryFormatter.format_for_agent_generative_prompt(
                model_response_instructions=full_prompt,
                context=context,
                user_input=user_input,
            ),
            logger=self.logger
        )
    
    async def _generate_response(
        self,
        classification: ResistanceClassification,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Generate a response to the user's resistance.

        Step 2 of 2-step process. Uses full context and comprehensive response strategies.
        """
        # Build full context for LLM
        skills_list = self._extract_skills_list(state)
        pref_vec_dict = state.preference_vector.model_dump() if state.preference_vector else {}
        conv_history = ConversationHistoryFormatter.format_to_string(context)

        # Get current recommendation being discussed
        current_rec_summary = "Unknown recommendation"
        if state.current_focus_id:
            rec = state.get_recommendation_by_id(state.current_focus_id)
            if rec:
                current_rec_summary = f"{rec.occupation if hasattr(rec, 'occupation') else rec.opportunity_title if hasattr(rec, 'opportunity_title') else 'Recommendation'}"

        context_block = build_context_block(
            skills=skills_list,
            preference_vector=pref_vec_dict,
            recommendations_summary=f"User is concerned about: {current_rec_summary}",
            conversation_history=conv_history,
            country_of_user=state.country_of_user
        )

        # Add classification context
        classification_context = f"""
## CLASSIFIED RESISTANCE

**Resistance Type**: {classification.resistance_type.upper()}
**User's Concern**: {classification.concern_summary}
**Reasoning**: {classification.reasoning}

---
"""

        # Build full prompt
        full_prompt = context_block + classification_context + ADDRESS_CONCERNS_PROMPT_RESPONSE + "\n\n" + get_json_response_instructions()

        return await self._conversation_caller.call_llm(
            llm=self._conversation_llm,
            llm_input=ConversationHistoryFormatter.format_for_agent_generative_prompt(
                model_response_instructions=full_prompt,
                context=context,
                user_input=user_input
            ),
            logger=self.logger
        )

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
    
