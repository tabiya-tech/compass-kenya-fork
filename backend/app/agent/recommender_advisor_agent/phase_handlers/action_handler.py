"""
Action Phase Handler for the Recommender/Advisor Agent.

Handles the ACTION_PLANNING phase where we guide users
toward concrete next steps.
"""

from app.agent.agent_types import LLMStats
from app.agent.llm_caller import LLMCaller
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    ActionCommitment,
    ActionType,
    CommitmentLevel,
    OccupationRecommendation,
    OpportunityRecommendation,
    SkillsTrainingRecommendation,
)
from app.agent.recommender_advisor_agent.llm_response_models import (
    ConversationResponse,
    ActionExtractionResult,
)
from app.agent.recommender_advisor_agent.phase_handlers.base_handler import BasePhaseHandler
from app.agent.simple_llm_agent.prompt_response_template import get_json_response_instructions
from app.conversation_memory.conversation_formatter import ConversationHistoryFormatter
from app.conversation_memory.conversation_memory_manager import ConversationContext
from common_libs.llm.generative_models import GeminiGenerativeLLM


class ActionPhaseHandler(BasePhaseHandler):
    """
    Handles the ACTION_PLANNING phase.
    
    Responsibilities:
    - Extract action commitments from user responses
    - Guide users toward concrete next steps
    - Track commitment level and barriers
    - Transition to wrapup when commitment is made
    """
    
    def __init__(
        self,
        conversation_llm: GeminiGenerativeLLM,
        conversation_caller: LLMCaller[ConversationResponse],
        action_caller: LLMCaller[ActionExtractionResult],
        **kwargs
    ):
        """
        Initialize the action handler.
        
        Args:
            action_caller: LLM caller for action extraction
        """
        super().__init__(conversation_llm, conversation_caller, **kwargs)
        self._action_caller = action_caller
    
    async def handle(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """
        Handle action planning phase.
        """
        all_llm_stats: list[LLMStats] = []
        
        # Extract action commitment from user input
        extraction, llm_stats = await self._extract_action(user_input, context)
        all_llm_stats.extend(llm_stats)
        
        # If user made a commitment, record it
        if extraction.has_commitment and extraction.action_type:
            try:
                action_type = ActionType(extraction.action_type)
                commitment_level = CommitmentLevel(extraction.commitment_level or "interested")
                
                commitment = ActionCommitment(
                    recommendation_id=state.current_focus_id or "unknown",
                    recommendation_type=state.current_recommendation_type,
                    recommendation_title=self._get_focus_title(state),
                    action_type=action_type,
                    commitment_level=commitment_level,
                    barriers_mentioned=extraction.barriers_mentioned
                )
                state.set_action_commitment(commitment)
                
                # Strong commitment → move to wrapup
                if commitment_level in [CommitmentLevel.WILL_DO_THIS_WEEK, CommitmentLevel.WILL_DO_THIS_MONTH]:
                    state.conversation_phase = ConversationPhase.WRAPUP
                    return ConversationResponse(
                        reasoning="User made strong commitment, moving to wrapup",
                        message=self._build_commitment_acknowledgment(commitment),
                        finished=False
                    ), all_llm_stats
            except ValueError as e:
                self.logger.warning(f"Invalid action/commitment type: {e}")
        
        # Generate action-focused response
        response, llm_stats = await self._generate_action_prompt(state, context)
        all_llm_stats.extend(llm_stats)
        
        return response, all_llm_stats
    
    async def _extract_action(
        self,
        user_input: str,
        context: ConversationContext
    ) -> tuple[ActionExtractionResult, list[LLMStats]]:
        """Extract action commitment from user input."""
        prompt = f"""
Analyze this user response to determine if they're committing to an action:

User said: "{user_input}"

Context: They've been exploring career/occupation recommendations.

Determine:
1. Did they make a clear commitment to take action?
2. What type of action? (apply_to_job, enroll_in_training, explore_occupation, research_employer, update_cv, network)
3. How strong is their commitment? (will_do_this_week, will_do_this_month, interested, maybe_later, not_interested)
4. Did they mention any barriers?

{get_json_response_instructions()}
"""
        
        return await self._action_caller.call_llm(
            llm=self._conversation_llm,
            llm_input=ConversationHistoryFormatter.format_for_agent_generative_prompt(
                model_response_instructions=prompt,
                conversation_context=context,
            ),
            logger=self.logger
        )
    
    async def _generate_action_prompt(
        self,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Generate an action-focused response."""
        focus_title = self._get_focus_title(state)
        
        prompt = f"""
The user has been exploring career recommendations and we're now in action planning.

Current focus: {focus_title}

Generate a response that:
1. Acknowledges their interest
2. Offers specific next steps they could take:
   - Apply to a job posting
   - Enroll in relevant training
   - Research the field more
   - Update their CV
   - Connect with people in the field
3. Asks for a commitment with a timeline ("When will you...?" or "Would you like to do X this week?")
4. Keep it supportive and action-oriented, but not pushy

If they haven't shown clear interest yet, first try to understand what's holding them back.

{get_json_response_instructions()}
"""
        
        return await self._conversation_caller.call_llm(
            llm=self._conversation_llm,
            llm_input=ConversationHistoryFormatter.format_for_agent_generative_prompt(
                model_response_instructions=prompt,
                conversation_context=context,
            ),
            logger=self.logger
        )
    
    def _get_focus_title(self, state: RecommenderAdvisorAgentState) -> str:
        """Get the title of the currently focused recommendation."""
        if state.current_focus_id is None:
            return "Unknown"
        
        rec = state.get_recommendation_by_id(state.current_focus_id)
        
        if rec is None:
            return "Unknown"
        
        if isinstance(rec, OccupationRecommendation):
            return rec.occupation
        elif isinstance(rec, OpportunityRecommendation):
            return rec.opportunity_title
        elif isinstance(rec, SkillsTrainingRecommendation):
            return rec.training_title
        
        return "Unknown"
    
    def _build_commitment_acknowledgment(self, commitment: ActionCommitment) -> str:
        """Build acknowledgment message for a commitment."""
        action_verbs = {
            ActionType.APPLY_TO_JOB: "apply",
            ActionType.ENROLL_IN_TRAINING: "enroll",
            ActionType.EXPLORE_OCCUPATION: "explore",
            ActionType.RESEARCH_EMPLOYER: "research",
            ActionType.UPDATE_CV: "update your CV",
            ActionType.NETWORK: "reach out to contacts",
        }
        
        action_verb = action_verbs.get(commitment.action_type, "take action")
        timeline = commitment.commitment_level.value.replace("_", " ")
        
        return f"""Excellent! So you're going to {action_verb} for **{commitment.recommendation_title}**, and you'll do that {timeline}.

That's a great step. Let me summarize what we've discussed..."""
