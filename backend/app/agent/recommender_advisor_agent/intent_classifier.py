"""
Intent Classifier for the Recommender/Advisor Agent.

Provides centralized, phase-aware intent classification to avoid code duplication
and ensure consistent intent detection across all conversation phases.
"""

import logging
from typing import Tuple

from app.agent.agent_types import LLMStats
from app.agent.llm_caller import LLMCaller
from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import ConversationPhase
from app.agent.recommender_advisor_agent.llm_response_models import UserIntentClassification
from app.conversation_memory.conversation_memory_manager import ConversationContext
from app.conversation_memory.conversation_formatter import ConversationHistoryFormatter
from common_libs.llm.generative_models import GeminiGenerativeLLM


class IntentClassifier:
    """
    Centralized intent classification for the Recommender/Advisor Agent.

    Provides phase-aware intent classification that adapts prompts based on
    the current conversation phase to accurately detect user intent.
    """

    def __init__(self, intent_caller: LLMCaller[UserIntentClassification]):
        """
        Initialize the intent classifier.

        Args:
            intent_caller: LLM caller for UserIntentClassification responses
        """
        self._intent_caller = intent_caller

    async def classify_intent(
        self,
        user_input: str,
        state: RecommenderAdvisorAgentState,
        context: ConversationContext,
        phase: ConversationPhase,
        llm: GeminiGenerativeLLM,
        logger: logging.Logger
    ) -> Tuple[UserIntentClassification | None, list[LLMStats]]:
        """
        Classify user intent based on their message and current conversation phase.

        Args:
            user_input: The user's message
            state: Current agent state
            context: Conversation context
            phase: Current conversation phase
            llm: LLM instance to use
            logger: Logger for debug/error messages

        Returns:
            Tuple of (UserIntentClassification, LLMStats) or (None, LLMStats) if classification fails
        """
        # Build phase-specific prompt
        if phase == ConversationPhase.PRESENT_RECOMMENDATIONS:
            prompt = self._build_present_phase_prompt(user_input, state)
        elif phase == ConversationPhase.CAREER_EXPLORATION:
            prompt = self._build_exploration_phase_prompt(user_input, state)
        elif phase == ConversationPhase.FOLLOW_UP:
            prompt = self._build_followup_phase_prompt(user_input, state)
        else:
            # Generic prompt for other phases
            prompt = self._build_generic_prompt(user_input, state)

        # Call LLM for intent classification
        return await self._intent_caller.call_llm(
            llm=llm,
            llm_input=ConversationHistoryFormatter.format_for_agent_generative_prompt(
                model_response_instructions=prompt,
                context=context,
                user_input=user_input,
            ),
            logger=logger
        )

    def _build_present_phase_prompt(self, user_input: str, state: RecommenderAdvisorAgentState) -> str:
        """
        Build intent classification prompt for PRESENT_RECOMMENDATIONS phase.

        In this phase, we detect:
        - explore_occupation: User wants details on a specific occupation
        - reject: User doesn't like the recommendations
        - express_concern: User has worries/doubts
        - ask_question: Factual questions
        - accept: User likes a recommendation
        """
        # Build occupation list for reference
        occ_list = ""
        if state.recommendations:
            for i, occ in enumerate(state.recommendations.occupation_recommendations[:5], 1):
                occ_list += f"{i}. {occ.occupation} (uuid: {occ.uuid})\n"

        return f"""
Classify what the user wants to do based on their message.

User said: "{user_input}"

Available occupations:
{occ_list}

Current rejection count: {state.rejected_occupations}

IMPORTANT: If the user mentions an occupation by name (e.g., "Data Analyst"), match it to the list above and set:
- target_occupation_index to the number (1-based)
- target_recommendation_id to the uuid
Do a fuzzy match - "Data Analyst", "data analyst", "analyst" should all match "Data Analyst".

Possible intents:
- "explore_occupation": They want to learn more about a specific occupation (e.g., "tell me more about Data Analyst", "I'm interested in #1", "what would I do day-to-day?")
- "reject": They're explicitly rejecting the current recommendations (e.g., "not interested", "I don't like these", "none of these appeal to me", "I reject that")
- "express_concern": They're expressing worry, doubt, hesitation, or identifying problems with a recommendation
  Examples: "I'm worried about...", "what if...", "but...", "seems like it would...", "this takes too much...", "I don't think I can...", "sounds too...", "not enough...", "too boring", "doesn't sound adventurous"
  Key: ANY statement that raises an objection, barrier, or negative aspect about a recommendation
- "ask_question": ONLY factual/clarification questions that don't express concern (e.g., "what does M&E mean?", "which one pays more?")
- "accept": They're accepting/interested in an option (e.g., "I like that one", "sounds good", "let's go with that")
- "other": Unclear or off-topic

CRITICAL DISTINCTION:
- "this role takes a lot from work-life balance" = EXPRESS_CONCERN (identifying a problem)
- "what is work-life balance?" = ASK_QUESTION (factual clarification)
- "too boring" = EXPRESS_CONCERN (objection)
- "not good enough" = EXPRESS_CONCERN (objection)

If they mentioned a number (1, 2, 3...) or occupation name, identify which one in target_occupation_index (1-based).
If they mentioned a specific occupation by name, set target_recommendation_id to the uuid.

Your response must be a JSON object with the following schema:
{{
    "reasoning": "A step by step explanation of why you classified this intent",
    "intent": "One of: explore_occupation, reject, express_concern, ask_question, accept, other",
    "target_recommendation_id": "The UUID of the recommendation if identified, or null",
    "target_occupation_index": "The 1-based index number if identified, or null"
}}

Always return a valid JSON object matching this exact schema.
"""

    def _build_exploration_phase_prompt(self, user_input: str, state: RecommenderAdvisorAgentState) -> str:
        """
        Build intent classification prompt for CAREER_EXPLORATION phase.

        In this phase, we detect:
        - express_concern: User has doubts/worries about the occupation being explored
        - accept: User is ready to move forward with this occupation
        - reject: User doesn't like this occupation anymore
        - explore_different: User wants to explore a different occupation
        - ask_question: Clarifying questions about the current occupation
        """
        current_occ = "Unknown"
        if state.current_focus_id:
            rec = state.get_recommendation_by_id(state.current_focus_id)
            if rec:
                current_occ = rec.occupation if hasattr(rec, 'occupation') else "Unknown"

        # Build list of other available occupations
        other_occs = ""
        if state.recommendations:
            for i, occ in enumerate(state.recommendations.occupation_recommendations[:5], 1):
                if occ.uuid != state.current_focus_id:
                    other_occs += f"{i}. {occ.occupation} (uuid: {occ.uuid})\n"

        return f"""
Classify what the user wants to do based on their message.

User said: "{user_input}"

Currently exploring: {current_occ}

Other available occupations:
{other_occs}

Possible intents:
- "express_concern": They're expressing worry, doubt, hesitation, or identifying barriers about the occupation
  Examples: "I feel incapable", "I'm not equipped", "imposter syndrome", "I'm worried about...", "what if I can't...", "seems too hard", "I don't have the skills", "too much education needed"
  Key: ANY statement expressing self-doubt, barriers, worries, or negative feelings about their ability to succeed in this role
- "accept": They're ready to commit or move forward with this occupation (e.g., "this sounds great", "I want to pursue this", "how do I get started?")
- "reject": They don't want this occupation anymore (e.g., "not interested in this one", "let's look at something else", "this isn't for me")
- "explore_different": They want to explore a different occupation instead (mention another occupation by name or number)
- "ask_question": Factual/clarification questions about the current occupation that don't express concern (e.g., "what does M&E stand for?", "how long is the training?")
- "continue_exploring": They want more information about the current occupation (e.g., "tell me more", "what else?", "okay", "go on")
- "other": Unclear or off-topic

CRITICAL DISTINCTION:
- "I feel incapable" = EXPRESS_CONCERN (self-doubt/barrier)
- "I'm not sure I have the skills" = EXPRESS_CONCERN (self-doubt/barrier)
- "What skills are needed?" = ASK_QUESTION (factual)
- "Sounds good" = ACCEPT (positive commitment)
- "Not for me" = REJECT (negative)

If they mentioned a different occupation by name or number, set target_recommendation_id or target_occupation_index.

Your response must be a JSON object with the following schema:
{{
    "reasoning": "A step by step explanation of why you classified this intent",
    "intent": "One of: express_concern, accept, reject, explore_different, ask_question, continue_exploring, other",
    "target_recommendation_id": "The UUID of a different occupation if mentioned, or null",
    "target_occupation_index": "The 1-based index of a different occupation if mentioned, or null"
}}

Always return a valid JSON object matching this exact schema.
"""

    def _build_followup_phase_prompt(self, user_input: str, state: RecommenderAdvisorAgentState) -> str:
        """
        Build intent classification prompt for FOLLOW_UP phase.

        In this phase, we clarify ambiguous responses.
        """
        # Build occupation list for reference
        occ_list = ""
        if state.recommendations:
            for i, occ in enumerate(state.recommendations.occupation_recommendations[:5], 1):
                occ_list += f"{i}. {occ.occupation} (uuid: {occ.uuid})\n"

        return f"""
Classify what the user wants to do based on their message.

User said: "{user_input}"

Available occupations:
{occ_list}

Possible intents:
- "explore_occupation": They want to learn more about a specific occupation
- "show_opportunities": They want to see job postings
- "express_concern": They're expressing worry/doubt about recommendations
- "ask_question": They have a question about something
- "reject": They're rejecting current options
- "accept": They're accepting/interested in an option
- "other": Unclear or off-topic

If they mentioned a number (1, 2, 3...) or occupation name, identify which one.

Your response must be a JSON object with the following schema:
{{
    "reasoning": "A step by step explanation of why you classified this intent",
    "intent": "One of: explore_occupation, show_opportunities, express_concern, ask_question, reject, accept, other",
    "target_recommendation_id": "The UUID of the recommendation if identified, or null",
    "target_occupation_index": "The 1-based index number if identified, or null"
}}

Always return a valid JSON object matching this exact schema.
"""

    def _build_generic_prompt(self, user_input: str, state: RecommenderAdvisorAgentState) -> str:
        """
        Build generic intent classification prompt for other phases.
        """
        return f"""
Classify what the user wants to do based on their message.

User said: "{user_input}"

Possible intents:
- "express_concern": They're expressing worry, doubt, or hesitation
- "ask_question": They have a question
- "accept": They're agreeing or accepting something
- "reject": They're rejecting or declining something
- "other": Unclear or off-topic

Your response must be a JSON object with the following schema:
{{
    "reasoning": "A step by step explanation of why you classified this intent",
    "intent": "One of: express_concern, ask_question, accept, reject, other",
    "target_recommendation_id": null,
    "target_occupation_index": null
}}

Always return a valid JSON object matching this exact schema.
"""
