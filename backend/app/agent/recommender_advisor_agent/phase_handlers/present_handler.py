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
    OccupationRecommendation,
    OpportunityRecommendation
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
from app.agent.recommender_advisor_agent.intent_classifier import IntentClassifier, _batch_size
from app.conversation_memory.conversation_memory_manager import ConversationContext
from app.conversation_memory.conversation_formatter import ConversationHistoryFormatter
from app.agent.simple_llm_agent.prompt_response_template import get_json_response_instructions
from common_libs.llm.generative_models import GeminiGenerativeLLM
from app.vector_search.esco_entities import OccupationEntity
from app.vector_search.similarity_search_service import SimilaritySearchService


# Instruction prompts for the careers/jobs/both views chosen at INTRO. They tell the LLM
# to present only the fields present in the data (every job field is optional and guarded
# upstream), so missing salary/employer/deadline simply don't appear rather than being
# invented.
_PRESENT_JOBS_PROMPT = """
You are a warm, encouraging career advisor. The user asked to see actual job openings.
Present the job opportunities from the context above clearly and concisely.

For each job, include only the fields that are present in the data (skip any that are missing):
the job title, location, why it matches the user, the contract type, the salary range (only if it
appears in the data), the expected labor demand (only if present), and the application link (label
it "Apply here:"). Do NOT invent salary, employer, demand, deadlines, or any field not in the data —
if a field is missing, simply leave it out.

End with one short, encouraging sentence that invites the user to dig into these openings — for
example, ask if they'd like to know more about any one of them (how it matches them, what it
involves, how to apply). As secondary options, mention they can also see career paths to consider,
or check back another day as new listings are added regularly. Lead with exploring the jobs, not
with leaving them.
"""

_PRESENT_BOTH_PROMPT = """
You are a warm, encouraging career advisor. The user asked to see both job openings and career paths.
Present the JOB OPENINGS FIRST, then the CAREER PATHS, using the data in the context above.

For each job opening, include only the fields present in the data (skip missing ones): title, location,
why it matches, contract type, salary range (only if present in the data), expected labor demand (only
if present), and the application link (label it "Apply here:"). Do NOT invent salary, employer, demand,
deadlines, or any field not in the data.

For each career path, give its title and a brief, encouraging description of why it could fit.

Keep it scannable. End with one encouraging sentence inviting the user to tell you which job or career
path interests them, or to ask any questions.
"""


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

        # View-choice gate: the INTRO phase asked whether the user wants career paths,
        # job openings, or both. Classify their answer once, record it, and render the
        # chosen view. "careers" falls through to the occupation presentation below.
        if state.awaiting_view_choice:
            state.awaiting_view_choice = False
            view, choice_stats = await self._classify_view_choice(user_input, context)
            state.recommendation_view = view
            self.logger.info(f"Recommendation view choice classified as: {view}")
            if view == "jobs":
                response, stats = await self._present_jobs_view(user_input, state, context)
                return response, choice_stats + stats
            if view == "both":
                response, stats = await self._present_both_view(user_input, state, context)
                return response, choice_stats + stats
            # view == "careers": fall through to the occupation presentation below
            all_llm_stats.extend(choice_stats)

        # Jobs view requested from elsewhere (e.g. show_opportunities routed here from
        # FOLLOW_UP sets recommendation_view="jobs"). Render the openings directly instead
        # of the occupation presentation; afterwards presented_opportunities is populated so
        # later turns flow through the jobs follow-up gate below.
        if (state.recommendation_view == "jobs"
                and not state.awaiting_view_choice
                and not state.presented_opportunities):
            response, stats = await self._present_jobs_view(user_input, state, context)
            all_llm_stats.extend(stats)
            return response, all_llm_stats

        # Follow-up after a jobs/both view: opportunities were already shown, so classify
        # the user's reply in an opportunity-aware way. Without this, a follow-up about a
        # specific opening falls into the initial occupation presentation below and wrongly
        # dumps career paths (the post-jobs-view bug). Returns None only for "show me the
        # careers", which deliberately falls through to the occupation presentation.
        if (state.recommendation_view in ("jobs", "both")
                and state.presented_opportunities
                and not state.awaiting_view_choice
                and user_input.strip()
                and self._intent_classifier):
            handled = await self._handle_jobs_followup(user_input, state, context)
            if handled is not None:
                response, stats = handled
                all_llm_stats.extend(stats)
                return response, all_llm_stats

        # Get the next unpresented occupations in rank order. Using _next_unpresented
        # instead of a raw slice ensures pagination works correctly when occupations
        # were already shown via a "both" view before the user switched to the careers view.
        occupations = self._next_unpresented(
            state.recommendations.occupation_recommendations,
            state.presented_occupations,
        )

        if not occupations:
            # Fall back to opportunity recommendations if no occupations are available
            opportunities = self._next_unpresented(
                state.recommendations.opportunity_recommendations,
                state.presented_opportunities,
            )
            if opportunities:
                self.logger.info(
                    f"No occupation recommendations available; presenting {len(opportunities)} opportunity recommendations instead"
                )
                return await self._present_opportunities(user_input, state, context, opportunities)
            # Nothing presentable. Don't freeze here: drop the (empty) cached
            # recommendations and route back to INTRO so the next user turn re-calls the
            # matching service rather than re-entering this dead-end every turn.
            state.recommendations = None
            state.conversation_phase = ConversationPhase.INTRO
            return ConversationResponse(
                reasoning="No occupation or opportunity recommendations available; clearing cache and returning to INTRO so the next turn retries",
                message=(
                    "I couldn't find suitable recommendations just now — this can happen while "
                    "our job database is updating. Send me a message to try again, or check back "
                    "a little later and I'll have a fresh set for you."
                ),
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

    async def _present_opportunities(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext,
        opportunities: list
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Present job opportunity recommendations when no occupation recommendations are available."""
        recs_summary = self._build_opportunities_summary(opportunities)

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

        full_prompt = context_block + PRESENT_RECOMMENDATIONS_PROMPT + get_json_response_instructions()
        response, llm_stats = await self._call_llm(full_prompt, user_input, context)
        return response, llm_stats

    async def _classify_view_choice(
        self,
        user_input: str,
        context: ConversationContext,
    ) -> tuple[str, list[LLMStats]]:
        """Classify the user's careers/jobs/both choice. Defaults to 'both' when unclear."""
        if self._intent_classifier:
            intent, stats = await self._intent_classifier.classify_view_choice(
                user_input=user_input,
                context=context,
                llm=self._conversation_llm,
                logger=self.logger,
            )
            if intent and intent.intent in ("careers", "jobs", "both"):
                return intent.intent, stats
            return self._keyword_view_choice(user_input), stats
        return self._keyword_view_choice(user_input), []

    @staticmethod
    def _keyword_view_choice(user_input: str) -> str:
        """Keyword fallback for the careers/jobs/both choice. Defaults to 'both' when unclear."""
        text = (user_input or "").lower()
        wants_jobs = any(k in text for k in (
            "job", "opening", "vacanc", "opportunit", "kazi", "apply", "hiring", "posting"))
        wants_careers = any(k in text for k in ("career", "path", "occupation", "profession"))
        if wants_jobs and not wants_careers:
            return "jobs"
        if wants_careers and not wants_jobs:
            return "careers"
        return "both"

    async def _present_jobs_view(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext,
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Present job openings, keeping the conversation open (jobs-only view)."""
        opportunities = self._next_unpresented(
            state.recommendations.opportunity_recommendations,
            state.presented_opportunities,
        )
        if not opportunities:
            return ConversationResponse(
                reasoning="User chose jobs but no job openings are available",
                message=("I don't have job openings matching your profile right now. "
                         "The listings refresh regularly, so it's worth checking back another day. "
                         "In the meantime, would you like to see some career paths to consider?"),
                finished=False
            ), []

        for opp in opportunities:
            if opp.uuid not in state.presented_opportunities:
                state.presented_opportunities.append(opp.uuid)

        recs_summary = self._build_opportunities_summary(opportunities)
        full_prompt = self._build_view_prompt(state, context, recs_summary, _PRESENT_JOBS_PROMPT)
        response, llm_stats = await self._call_llm(full_prompt, user_input, context)
        # Jobs view stays open (the user can keep talking) — never trust the LLM to end the turn.
        response.finished = False
        return response, llm_stats

    async def _present_both_view(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext,
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Present job openings first, then career paths, in one message (both view)."""
        opportunities = self._next_unpresented(
            state.recommendations.opportunity_recommendations,
            state.presented_opportunities,
        )
        occupations = self._next_unpresented(
            state.recommendations.occupation_recommendations,
            state.presented_occupations,
        )

        if not opportunities and not occupations:
            return ConversationResponse(
                reasoning="User chose both but nothing is available",
                message=("I couldn't find matches for you right now. The listings refresh "
                         "regularly, so please check back another day."),
                finished=False
            ), []

        for opp in opportunities:
            if opp.uuid not in state.presented_opportunities:
                state.presented_opportunities.append(opp.uuid)
        for occ in occupations:
            if occ.uuid not in state.presented_occupations:
                state.presented_occupations.append(occ.uuid)

        jobs_block = (self._build_opportunities_summary(opportunities)
                      if opportunities else "No job openings available right now.")
        careers_block = (self._build_detailed_recommendations_summary(occupations)
                         if occupations else "No career paths available right now.")
        recs_summary = (
            f"{jobs_block}\n\n"
            f"**Career Paths** (present these after the jobs):\n{careers_block}"
        )
        full_prompt = self._build_view_prompt(state, context, recs_summary, _PRESENT_BOTH_PROMPT)
        response, llm_stats = await self._call_llm(full_prompt, user_input, context)
        # Both view stays open so the user can keep discussing.
        response.finished = False
        return response, llm_stats

    def _build_view_prompt(
        self,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext,
        recs_summary: str,
        instruction_prompt: str,
    ) -> str:
        """Assemble the context block + a view-specific instruction prompt."""
        skills_list = self._extract_skills_list(state)
        pref_vec_dict = state.preference_vector.model_dump() if state.preference_vector else {}
        conv_history = ConversationHistoryFormatter.format_to_string(context)
        context_block = build_context_block(
            skills=skills_list,
            preference_vector=pref_vec_dict,
            recommendations_summary=recs_summary,
            conversation_history=conv_history,
            country_of_user=state.country_of_user,
        )
        return context_block + instruction_prompt + get_json_response_instructions()

    def _build_opportunities_summary(self, opportunities: list) -> str:
        """Build a summary of job opportunity recommendations for LLM context."""
        lines = [
            "**Job Opportunities** (present these job postings to the user):"
        ]
        for i, opp in enumerate(opportunities, 1):
            lines.append(f"{i}. **{opp.opportunity_title}** (Rank: {opp.rank})")
            if opp.employer:
                lines.append(f"   - Employer: {opp.employer}")
            if opp.location:
                lines.append(f"   - Location: {opp.location}")
            if opp.demand_label:
                lines.append(f"   - Labor demand: {opp.demand_label}")
            if opp.salary_range:
                lines.append(f"   - Salary: {opp.salary_range}")
            if opp.justification:
                lines.append(f"   - Why it matches you: {opp.justification}")
            if opp.application_deadline:
                lines.append(f"   - Closing date: {opp.application_deadline}")
            if opp.final_score is not None:
                lines.append(f"   - Match Score: {opp.final_score:.0%}")
            eligible_label = "Yes" if opp.is_eligible else "No"
            lines.append(f"   - Eligible: {eligible_label}")
            if opp.posting_url:
                lines.append(f"   - Apply here: {opp.posting_url}")
            lines.append("")
        return '\n'.join(lines)

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

        # Handle SHOW_MORE intent - user wants the next batch of recommendations
        elif intent.intent == "show_more":
            return await self._handle_show_more_intent(user_input, state, context)

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

    @staticmethod
    def _next_unpresented(
        all_recs: list, presented_ids: list[str], k: Optional[int] = None
    ) -> list:
        """Next ``k`` recommendations in rank order that the user has not seen yet.

        ``presented_ids`` is treated as an ordered log (initial batch, then any prior
        show_more batches), so callers can compute both what to render next and
        whether the underlying list is exhausted. ``k`` defaults to the configured
        recommendation batch size (``COMPASS_RECOMMENDATION_BATCH_SIZE``).
        """
        seen = set(presented_ids)
        limit = _batch_size() if k is None else k
        return [r for r in all_recs if r.uuid not in seen][:limit]

    async def _handle_show_more_intent(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext,
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Present the next batch of recommendations in the user's current view.

        - ``careers`` view (or default) → next 5 occupations.
        - ``jobs`` view → next 5 opportunities.
        - ``both`` view → next 5 opportunities followed by next 5 occupations.

        When the underlying list is exhausted, returns a graceful "that's all"
        message that invites the user to go deeper on one of the ones already shown
        (rather than looping back to page 1). Stays in PRESENT_RECOMMENDATIONS.
        """
        if state.recommendations is None:
            return ConversationResponse(
                reasoning="show_more requested but no recommendations are cached",
                message="Let me pull up your recommendations first — say the word when you're ready.",
                finished=False,
            ), []

        view = state.recommendation_view or "careers"
        all_llm_stats: list[LLMStats] = []

        if view == "jobs":
            next_opps = self._next_unpresented(
                state.recommendations.opportunity_recommendations,
                state.presented_opportunities,
            )
            if not next_opps:
                return self._exhausted_response("opportunity", state), []
            for opp in next_opps:
                state.presented_opportunities.append(opp.uuid)
            recs_summary = self._build_opportunities_summary(next_opps)
            full_prompt = self._build_view_prompt(state, context, recs_summary, _PRESENT_JOBS_PROMPT)
            response, llm_stats = await self._call_llm(full_prompt, user_input, context)
            response.finished = False
            all_llm_stats.extend(llm_stats)
            return response, all_llm_stats

        if view == "both":
            next_opps = self._next_unpresented(
                state.recommendations.opportunity_recommendations,
                state.presented_opportunities,
            )
            next_occs = self._next_unpresented(
                state.recommendations.occupation_recommendations,
                state.presented_occupations,
            )
            if not next_opps and not next_occs:
                return self._exhausted_response("both", state), []
            for opp in next_opps:
                state.presented_opportunities.append(opp.uuid)
            for occ in next_occs:
                state.presented_occupations.append(occ.uuid)
            jobs_block = (self._build_opportunities_summary(next_opps)
                          if next_opps else "No more job openings to show.")
            careers_block = (self._build_detailed_recommendations_summary(next_occs)
                             if next_occs else "No more career paths to show.")
            recs_summary = (
                f"{jobs_block}\n\n"
                f"**Career Paths** (present these after the jobs):\n{careers_block}"
            )
            full_prompt = self._build_view_prompt(state, context, recs_summary, _PRESENT_BOTH_PROMPT)
            response, llm_stats = await self._call_llm(full_prompt, user_input, context)
            response.finished = False
            all_llm_stats.extend(llm_stats)
            return response, all_llm_stats

        # Default: careers view.
        next_occs = self._next_unpresented(
            state.recommendations.occupation_recommendations,
            state.presented_occupations,
        )
        if not next_occs:
            return self._exhausted_response("occupation", state), []
        for occ in next_occs:
            state.presented_occupations.append(occ.uuid)
        recs_summary = self._build_detailed_recommendations_summary(next_occs)
        skills_list = self._extract_skills_list(state)
        pref_vec_dict = state.preference_vector.model_dump() if state.preference_vector else {}
        conv_history = ConversationHistoryFormatter.format_to_string(context)
        context_block = build_context_block(
            skills=skills_list,
            preference_vector=pref_vec_dict,
            recommendations_summary=recs_summary,
            conversation_history=conv_history,
            country_of_user=state.country_of_user,
        )
        full_prompt = context_block + PRESENT_RECOMMENDATIONS_PROMPT + get_json_response_instructions()
        response, llm_stats = await self._call_llm(full_prompt, user_input, context)
        response.finished = False
        all_llm_stats.extend(llm_stats)

        response.metadata = self._build_metadata(
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
                for occ in next_occs
            ],
        )
        return response, all_llm_stats

    def _exhausted_response(
        self,
        kind: str,
        state: RecommenderAdvisorAgentState,
    ) -> ConversationResponse:
        """Reply when the user has already seen every available recommendation of a kind.

        We deliberately do not loop back to page 1 — surfacing repeats without saying so
        would erode trust. Instead, invite the user to go deeper on something already shown
        (or, in the jobs case, to check back later for new listings)."""
        if kind == "opportunity":
            message = (
                "Those are all the job openings I have for you right now. New listings come in "
                "regularly, so it's worth checking back another day. In the meantime, would you "
                "like to dig into any of the openings we've already looked at, or switch to career "
                "paths to consider?"
            )
        elif kind == "both":
            message = (
                "That's everything I have on both sides for now — jobs and career paths. New "
                "openings come in regularly, so check back another day. For now, would you like "
                "to go deeper on any of the options we've already looked at?"
            )
        else:
            message = (
                "Those are all the career paths I have that fit your profile. Would you like to "
                "go deeper on any of the ones we've already looked at, or shall we look at actual "
                "job openings you could apply to?"
            )
        return ConversationResponse(
            reasoning=f"No more {kind} recommendations to present; offering to explore shown items",
            message=message,
            finished=False,
        )


    async def _handle_jobs_followup(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]] | None:
        """
        Route a user's reply after a jobs/both view.

        Returns the (response, stats) to send, or None to fall through to the occupation
        presentation (used only when the user asks to switch to career paths).
        """
        all_llm_stats: list[LLMStats] = []
        intent, stats = await self._intent_classifier.classify_jobs_followup(
            user_input=user_input,
            state=state,
            context=context,
            llm=self._conversation_llm,
            logger=self.logger,
        )
        all_llm_stats.extend(stats)

        if intent is not None:
            self.logger.info(f"Jobs follow-up intent: {intent.intent}")

            if intent.intent == "explore_opportunity":
                result = await self._handle_explore_opportunity_intent(intent, user_input, state, context)
                if result is not None:
                    response, s = result
                    return response, all_llm_stats + s
                # Couldn't identify which opening → answer conversationally below

            elif intent.intent == "explore_occupation" and (
                intent.target_recommendation_id or intent.target_occupation_index
            ):
                result = await self._handle_explore_intent(intent, user_input, state, context)
                if result is not None:
                    response, s = result
                    return response, all_llm_stats + s

            elif intent.intent == "show_careers":
                # Switch to the careers view; fall through to the occupation presentation.
                state.recommendation_view = "careers"
                return None

            elif intent.intent == "show_more":
                # Paginate the next batch in the same view (jobs, or jobs+careers for "both").
                response, s = await self._handle_show_more_intent(user_input, state, context)
                return response, all_llm_stats + s

            elif intent.intent == "express_concern":
                response, s = await self._handle_concern_intent(intent, user_input, state, context)
                return response, all_llm_stats + s

        # Ambiguous / question / accept / classification failed → keep the user in the
        # opportunity context with a conversational answer rather than dumping careers.
        response, s = await self._present_jobs_view(user_input, state, context)
        return response, all_llm_stats + s

    async def _handle_explore_opportunity_intent(
        self,
        intent: UserIntentClassification,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]] | None:
        """Handle the user wanting to explore a specific job opening (parity with occupations)."""
        target_opp = None

        if intent.target_recommendation_id:
            rec = state.get_recommendation_by_id(intent.target_recommendation_id)
            if isinstance(rec, OpportunityRecommendation):
                target_opp = rec

        if target_opp is None and intent.target_occupation_index and state.recommendations:
            idx = intent.target_occupation_index - 1  # Convert to 0-indexed
            opportunities = state.recommendations.opportunity_recommendations
            if 0 <= idx < len(opportunities):
                target_opp = opportunities[idx]

        if target_opp is None:
            self.logger.warning(
                f"Could not identify target opening. index={intent.target_occupation_index}, "
                f"id={intent.target_recommendation_id}"
            )
            return None

        # Set focus and explore via the shared exploration machinery. Opportunities skip the
        # tradeoffs check (it compares occupation labor-demand, which a posting doesn't have).
        state.current_focus_id = target_opp.uuid
        state.current_recommendation_type = "opportunity"
        state.mark_interest(target_opp.uuid, UserInterestLevel.EXPLORING)
        state.conversation_phase = ConversationPhase.CAREER_EXPLORATION

        self.logger.info(f"Transitioning to CAREER_EXPLORATION for opening {target_opp.opportunity_title}")

        if self._exploration_handler:
            return await self._exploration_handler.handle(user_input, state, context)

        return ConversationResponse(
            reasoning=f"User wants to explore opening {target_opp.opportunity_title}, transitioning to EXPLORATION",
            message=f"Great! Let me tell you more about the {target_opp.opportunity_title} opening.",
            finished=False
        ), []

    async def _handle_reject_intent(
        self,
        intent: UserIntentClassification,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext
    ) -> tuple[ConversationResponse, list[LLMStats]]:
        """Handle user rejecting recommendations."""
        # A rejection means the user is moving on from the current jobs/careers view, so clear
        # it — otherwise the jobs follow-up gate could re-fire on the next turn.
        state.recommendation_view = None

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

