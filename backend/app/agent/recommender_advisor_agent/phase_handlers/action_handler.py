"""
Action Phase Handler for the Recommender/Advisor Agent.

Handles the ACTION_PLANNING phase where we guide users
toward concrete next steps.
"""
from textwrap import dedent

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
from app.agent.recommender_advisor_agent.intent_classifier import IntentClassifier
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
    - Route user choices from ConcernsHandler menu
    """

    # Max times we hold a committed-but-vague user in ACTION_PLANNING to elicit a concrete
    # plan (when/how) before accepting whatever exists. Prevents interrogation loops.
    MAX_PLAN_NUDGES = 2

    def __init__(
        self,
        conversation_llm: GeminiGenerativeLLM,
        conversation_caller: LLMCaller[ConversationResponse],
        action_caller: LLMCaller[ActionExtractionResult],
        action_llm: GeminiGenerativeLLM = None,
        intent_classifier: IntentClassifier = None,
        present_handler: 'PresentPhaseHandler' = None,
        concerns_handler: 'ConcernsPhaseHandler' = None,
        wrapup_handler: 'WrapupPhaseHandler' = None,
        **kwargs
    ):
        """
        Initialize the action handler.

        Args:
            action_caller: LLM caller for action extraction
            action_llm: LLM configured with build_request_schema(ActionExtractionResult) for structured extraction
            intent_classifier: Optional intent classifier for routing user choices
            present_handler: Optional present handler for immediate delegation
            concerns_handler: Optional concerns handler for immediate delegation
            wrapup_handler: Optional wrapup handler for immediate delegation
        """
        super().__init__(conversation_llm, conversation_caller, **kwargs)
        self._action_caller = action_caller
        self._action_llm = action_llm
        self._intent_classifier = intent_classifier
        self._present_handler = present_handler
        self._concerns_handler = concerns_handler
        self._wrapup_handler = wrapup_handler
    
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

        # Check if user is responding to choice menu (from ConcernsHandler)
        # Use intent classification to route appropriately
        if self._intent_classifier:
            intent, llm_stats = await self._intent_classifier.classify_intent(
                user_input=user_input,
                state=state,
                context=context,
                phase=ConversationPhase.ACTION_PLANNING,
                llm=self._conversation_llm,
                logger=self.logger
            )
            all_llm_stats.extend(llm_stats)

            # Handle semantic intents with immediate delegation

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

            elif intent.intent == "accept":
                # User accepted/understood the previous explanation (likely post-guardrail)
                # Continue with action planning for the ORIGINAL recommendation they were exploring
                self.logger.info("User accepted explanation, continuing with action planning for current focus")
                # Fall through to normal action planning flow below

            elif intent.intent == "ask_question":
                # User has questions (likely about the comparison or mismatch)
                # Generate response that answers their question while staying in ACTION_PLANNING
                self.logger.info("User has questions, generating response while staying in ACTION_PLANNING")
                response, llm_stats = await self._generate_action_prompt(user_input, state, context)
                all_llm_stats.extend(llm_stats)
                return response, all_llm_stats

            elif intent.intent == "discuss_next_steps":
                # User wants to continue with action planning
                # Stay in ACTION_PLANNING and continue normally
                self.logger.info("User chose to discuss next steps, continuing in ACTION_PLANNING")
                # Fall through to normal action planning flow below

            elif intent.intent == "explore_alternatives":
                # User wants to see other recommendations
                # Delegate immediately to PresentHandler
                self.logger.info("User chose to explore alternatives, delegating to PresentHandler")
                state.conversation_phase = ConversationPhase.PRESENT_RECOMMENDATIONS

                if self._present_handler:
                    return await self._present_handler.handle(user_input, state, context)

                # Fallback if no present handler available
                return ConversationResponse(
                    reasoning="User wants to explore other recommendations",
                    message="Let me show you your other career recommendations.",
                    finished=False
                ), all_llm_stats

            elif intent.intent == "address_more_concerns":
                # User still has concerns
                # Delegate immediately to ConcernsHandler
                self.logger.info("User chose to address more concerns, delegating to ConcernsHandler")
                state.conversation_phase = ConversationPhase.ADDRESS_CONCERNS

                if self._concerns_handler:
                    return await self._concerns_handler.handle(user_input, state, context)

                # Fallback if no concerns handler available
                return ConversationResponse(
                    reasoning="User wants to address more concerns",
                    message="Of course! What's on your mind?",
                    finished=False
                ), all_llm_stats

        # Normal action planning flow: Extract action commitment from user input
        extraction, llm_stats = await self._extract_action(user_input, context)
        all_llm_stats.extend(llm_stats)

        # If extraction failed, fall back to generating a response without recording commitment
        if extraction is None:
            self.logger.warning("Action extraction failed, continuing with general action prompt")
            response, llm_stats = await self._generate_action_prompt(user_input, state, context)
            all_llm_stats.extend(llm_stats)
            return response, all_llm_stats

        # If user made a commitment, record it (merging plan slots across turns - a user may
        # give the day on one turn and the method on the next).
        if extraction.has_commitment and extraction.action_type:
            try:
                action_type = ActionType(extraction.action_type)
                commitment_level = CommitmentLevel(extraction.commitment_level or "interested")

                prev = state.action_commitment
                plan_when = extraction.plan_when or (prev.plan_when if prev else None)
                plan_where = extraction.plan_where or (prev.plan_where if prev else None)
                plan_how = extraction.plan_how or (prev.plan_how if prev else None)
                plan_backup = extraction.plan_backup or (prev.plan_backup if prev else None)

                commitment = ActionCommitment(
                    recommendation_id=state.current_focus_id or "unknown",
                    recommendation_type=state.current_recommendation_type,
                    recommendation_title=self._get_focus_title(state),
                    action_type=action_type,
                    commitment_level=commitment_level,
                    barriers_mentioned=extraction.barriers_mentioned,
                    plan_when=plan_when,
                    plan_where=plan_where,
                    plan_how=plan_how,
                    plan_backup=plan_backup,
                )
                state.set_action_commitment(commitment)

                # The user leaves ACTION_PLANNING only once the plan is concrete enough -
                # a commitment alone ("I'll apply this week") is NOT a plan. Require at least
                # a WHEN and a HOW. The nudge cap prevents interrogation: after
                # MAX_PLAN_NUDGES attempts we accept whatever plan exists and move on.
                strong = commitment_level in [CommitmentLevel.WILL_DO_THIS_WEEK, CommitmentLevel.WILL_DO_THIS_MONTH]
                plan_concrete = bool(plan_when) and bool(plan_how)

                if strong and (plan_concrete or state.action_planning_nudges >= self.MAX_PLAN_NUDGES):
                    state.conversation_phase = ConversationPhase.WRAPUP
                    self.logger.info(
                        f"Strong commitment with {'concrete' if plan_concrete else 'partial'} "
                        f"plan (nudges={state.action_planning_nudges}) → wrapup"
                    )

                    # Immediately invoke wrapup handler for seamless transition
                    if self._wrapup_handler:
                        return await self._wrapup_handler.handle(user_input, state, context)

                    # Fallback if no wrapup handler
                    return ConversationResponse(
                        reasoning="User made strong commitment with a plan, moving to wrapup (no handler available)",
                        message=self._build_commitment_acknowledgment(commitment),
                        finished=False
                    ), all_llm_stats

                if strong and not plan_concrete:
                    # Committed but vague - stay in ACTION_PLANNING and elicit the missing plan
                    # slot(s). The plan-making sequence in ACTION_PLANNING_PROMPT does the asking.
                    state.action_planning_nudges += 1
                    self.logger.info(
                        f"Strong commitment but plan not concrete (when={bool(plan_when)}, "
                        f"how={bool(plan_how)}); nudging for specifics (nudge {state.action_planning_nudges})"
                    )
                    response, llm_stats = await self._generate_action_prompt(user_input, state, context)
                    all_llm_stats.extend(llm_stats)
                    return response, all_llm_stats
            except ValueError as e:
                self.logger.warning(f"Invalid action/commitment type: {e}")
        
        # Generate action-focused response
        response, llm_stats = await self._generate_action_prompt(user_input, state, context)
        all_llm_stats.extend(llm_stats)

        return response, all_llm_stats
    
    async def _extract_action(
        self,
        user_input: str,
        context: ConversationContext
    ) -> tuple[ActionExtractionResult | None, list[LLMStats]]:
        """Extract action commitment from user input."""

        # Provide examples for structured output
        examples = [
            ActionExtractionResult(
                reasoning="User committed and gave a concrete plan with day, place, method, and a backup",
                has_commitment=True,
                action_type="apply_to_job",
                commitment_level="will_do_this_week",
                barriers_mentioned=["fare money"],
                plan_when="Thursday morning",
                plan_where="the depot",
                plan_how="matatu, ask the supervisor about the opening",
                plan_backup="Saturday if the fare is short"
            ),
            ActionExtractionResult(
                reasoning="User committed to a timeline but gave no concrete plan details yet",
                has_commitment=True,
                action_type="apply_to_job",
                commitment_level="will_do_this_week",
                barriers_mentioned=[],
                plan_when=None,
                plan_where=None,
                plan_how=None,
                plan_backup=None
            ),
            ActionExtractionResult(
                reasoning="User mentioned concerns about cost and time",
                has_commitment=False,
                action_type=None,
                commitment_level=None,
                barriers_mentioned=["cost concerns", "time constraints"]
            )
        ]

        prompt = dedent(f"""
            Analyze this user response to determine if they're committing to an action:
            
            User said: "{user_input}"
            
            Context: They've been exploring career/occupation recommendations.
            
            Determine:
            1. has_commitment: Did they make a clear commitment to take action? (true/false)
            2. action_type: What type of action? Must be one of: apply_to_job, enroll_in_training, explore_occupation, research_employer, network
            3. commitment_level: How strong is their commitment? Must be one of: will_do_this_week, will_do_this_month, interested, maybe_later, not_interested
            4. barriers_mentioned: List any barriers or concerns mentioned (empty list if none)
            5. plan_when: When they'll act, in their words (e.g., "Thursday morning"). Null if not stated.
            6. plan_where: Where / the specific target (e.g., "the depot", "the Brookside posting"). Null if not stated.
            7. plan_how: How - transport, what to bring or say (e.g., "matatu, ask the supervisor"). Null if not stated.
            8. plan_backup: The if-then backup for the most likely obstacle (e.g., "Saturday if the fare is short"). Null if not stated.
            
            IMPORTANT:
            - Use exact field names: has_commitment, action_type, commitment_level, barriers_mentioned, plan_when, plan_where, plan_how, plan_backup
            - action_type and commitment_level must use exact enum values listed above
            - If has_commitment is false, set action_type and commitment_level to null
            - Only fill a plan_* field if the user actually stated it THIS turn - never invent details. Leave it null otherwise.
            """)

        try:
            result = await self._action_caller.call_llm(
                llm=self._action_llm or self._conversation_llm,
                llm_input=ConversationHistoryFormatter.format_for_agent_generative_prompt(
                    model_response_instructions=prompt,
                    context=context,
                    user_input=user_input
                ),
                logger=self.logger
            )
            return result
        except Exception as e:
            self.logger.error(f"Failed to extract action after all retries: {e}")
            # Return None to indicate extraction failure
            return None, []
    
    async def _generate_action_prompt(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Generate an action-focused response."""
        focus_title = self._get_focus_title(state)

        # Provide examples for structured output
        examples = [
            ConversationResponse(
                reasoning="User expressed interest in moving forward, so I'll suggest concrete next steps and ask for commitment with a timeline. Set finished=False because we're waiting for their commitment.",
                message="That's fantastic! I'm glad this role appeals to you. Here are some concrete next steps you could take: 1) Research companies hiring for this role, 2) Look for relevant job postings, 3) Explore training opportunities to strengthen your skills. Would you like to start applying to jobs this week?",
                finished=False
            ),
            ConversationResponse(
                reasoning="User is hesitant and mentioned concerns. I need to understand what's holding them back before pushing for commitment. Set finished=False to continue the conversation.",
                message="I hear some hesitation in your response. What concerns do you have about taking these next steps? I'm here to help you work through them.",
                finished=False
            )
        ]

        # Prepend the activation toolkit + plan-making guidance (BASE + ACTION_PLANNING_PROMPT)
        # so the action phase actually deploys the toolkit, grounded in the user's own data.
        from app.agent.recommender_advisor_agent.prompts import (
            ACTION_PLANNING_PROMPT,
            build_context_block,
        )

        skills_list = self._extract_skills_list(state)
        pref_vec_dict = state.preference_vector.model_dump() if state.preference_vector else {}
        conv_history = ConversationHistoryFormatter.format_to_string(context)
        recs_summary = self._build_recommendations_summary(state)
        context_block = build_context_block(
            skills=skills_list,
            preference_vector=pref_vec_dict,
            recommendations_summary=recs_summary,
            conversation_history=conv_history,
            country_of_user=state.country_of_user,
        )

        prompt = context_block + ACTION_PLANNING_PROMPT + f"""

## CURRENT FOCUS
{focus_title}

IMPORTANT - Stay focused on core mission:
- Your role is to provide career guidance and recommend next steps
- DO NOT suggest creating or updating CVs (handled by a separate skills agent)
- DO NOT suggest tasks outside the scope of career exploration, job applications, training enrollment, or networking
- Focus ONLY on the 5 allowed action types: apply_to_job, enroll_in_training, explore_occupation, research_employer, network

IMPORTANT - When to set finished:
- Set finished=False when asking for commitment or waiting for user response
- ONLY set finished=True if the user has already made a strong commitment and you're providing a final summary/farewell
- In ACTION_PLANNING phase, finished should almost always be False unless wrapping up

{get_json_response_instructions(examples=examples)}
"""

        return await self._conversation_caller.call_llm(
            llm=self._conversation_llm,
            llm_input=ConversationHistoryFormatter.format_for_agent_generative_prompt(
                model_response_instructions=prompt,
                context=context,
                user_input=user_input
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
            ActionType.NETWORK: "reach out to contacts",
        }
        
        action_verb = action_verbs.get(commitment.action_type, "take action")
        timeline = commitment.commitment_level.value.replace("_", " ")
        
        return f"""Excellent! So you're going to {action_verb} for **{commitment.recommendation_title}**, and you'll do that {timeline}.

That's a great step. Let me summarize what we've discussed..."""
