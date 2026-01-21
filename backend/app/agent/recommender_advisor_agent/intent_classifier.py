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
        elif phase == ConversationPhase.ADDRESS_CONCERNS:
            prompt = self._build_concerns_phase_prompt(user_input, state)
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

CRITICAL OCCUPATION MATCHING RULES - When user mentions ANY occupation name, you MUST extract both fields:

* CASE-INSENSITIVE MATCHING: "fundi wa stima" = "Fundi wa Stima (Electrician)" = "FUNDI WA STIMA"
* PARTIAL NAME MATCHING: "electrician" matches "Fundi wa Stima (Electrician)", "analyst" matches "Data Analyst"
* IGNORE EXTRA TEXT: "Fundi wa stima" in "I love the Fundi wa stima role" still matches
* MATCH ANY SIGNIFICANT WORD: "boda" or "marine" or "port" matches respective occupations
* MATCH MAIN OCCUPATION NAME: Focus on the main title before parentheses

EXTRACTION REQUIREMENTS when user mentions an occupation:
- Set target_occupation_index = the NUMBER from the list (e.g., if matched to "1. Fundi wa Stima...", set to 1)
- Set target_recommendation_id = the UUID from the list (copy the full uuid string)
- You MUST set BOTH fields when you detect an occupation mention

MATCHING EXAMPLES:
- User: "tell me about Fundi wa stima" → Match "Fundi wa Stima (Electrician)" → Extract number 1 and its UUID
- User: "I love the marine role" → Match "Boat/Marine Equipment Fundi" → Extract its number and UUID
- User: "the boda boda option looks good" → Match "Boda-Boda Rider" → Extract its number and UUID
- User: "interested in #3" → Match occupation at position 3 → Extract number 3 and its UUID

Possible intents:
- "explore_occupation": They want to learn more about a specific occupation (e.g., "tell me more about Data Analyst", "I'm interested in #1", "what would I do day-to-day?", "I love the marine role")
  CRITICAL: When this intent is detected, you MUST also extract target_occupation_index AND target_recommendation_id using the matching rules above
- "reject": They're explicitly rejecting the current recommendations (e.g., "not interested", "I don't like these", "none of these appeal to me", "I reject that")
- "express_concern": They're expressing worry, doubt, hesitation, or identifying problems with a recommendation
  Examples: "I'm worried about...", "what if...", "but...", "seems like it would...", "this takes too much...", "I don't think I can...", "sounds too...", "not enough...", "too boring", "doesn't sound adventurous"
  Key: ANY statement that raises an objection, barrier, or negative aspect about a recommendation
- "ask_question": ONLY factual/clarification questions that don't express concern (e.g., "what does M&E mean?", "which one pays more?")
- "accept": They're accepting/interested in an option (e.g., "I like that one", "sounds good", "let's go with that")
- "request_outside_recommendations": They want to pursue an occupation/career that is NOT in the available recommendations above
  Examples: "I want to be a DJ", "What about becoming a pilot?", "I'd rather be a doctor", "Can we look at nursing?"
  Key: They mention a specific occupation/career that is NOT in the list of available occupations above
  IMPORTANT: Set requested_occupation_name to the occupation they mentioned (e.g., "DJ", "pilot", "doctor")
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
    "intent": "One of: explore_occupation, reject, express_concern, ask_question, accept, request_outside_recommendations, other",
    "target_recommendation_id": "The UUID of the recommendation if identified, or null",
    "target_occupation_index": "The 1-based index number if identified, or null",
    "requested_occupation_name": "The occupation name if they requested something outside recommendations, or null"
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
- "accept": They're ready to commit or move forward with this occupation
  Examples: "this sounds great", "I want to pursue this", "how do I get started?", "let's move forward", "I'm ready", "no concerns", "what next", "okay let's do it", "yes"
  Key: ANY statement indicating readiness, agreement to proceed, or asking about next steps

- "express_concern": They're expressing worry, doubt, hesitation, or identifying barriers about the occupation
  Examples: "I feel incapable", "I'm not equipped", "imposter syndrome", "I'm worried about...", "what if I can't...", "seems too hard", "I don't have the skills", "too much education needed"
  Key: ANY statement expressing self-doubt, barriers, worries, or negative feelings about their ability to succeed in this role

- "reject": They don't want this occupation anymore (e.g., "not interested in this one", "let's look at something else", "this isn't for me")

- "explore_different": They want to explore a different occupation instead (mention another occupation by name or number FROM THE LIST)
  CRITICAL MATCHING RULES - You MUST extract target_recommendation_id or target_occupation_index when user mentions an occupation:

  * CASE-INSENSITIVE MATCHING: "fundi wa stima" = "Fundi wa Stima (Electrician)" = "FUNDI WA STIMA"
  * PARTIAL NAME MATCHING: "electrician" matches "Fundi wa Stima (Electrician)"
  * IGNORE EXTRA TEXT: "Fundi wa stima" in "maybe Fundi wa stima inaeza kua poa" still matches
  * MATCH ANY SIGNIFICANT WORD: If user says "boda" or "boda boda", match "Boda-Boda Rider / Delivery Driver"
  * MATCH MAIN OCCUPATION NAME: User says "Fundi wa Stima" → matches list item "1. Fundi wa Stima (Electrician) (uuid: abc123)"

  EXTRACTION REQUIREMENTS:
  - Set target_occupation_index = the NUMBER from the list (e.g., if it's "1. Fundi wa Stima...", set to 1)
  - Set target_recommendation_id = the UUID from the list (e.g., if it's "uuid: occ_001_uuid", set to "occ_001_uuid")
  - You MUST set BOTH fields when you detect an occupation switch

  EXAMPLES OF MATCHING:
  - User: "Fundi wa stima" → Look for "Fundi wa Stima" in list → Extract number and UUID
  - User: "maybe electrician would be better" → Look for "Electrician" or "electrician" in any occupation name → Extract number and UUID
  - User: "what about the boda role?" → Look for "boda" or "Boda" in list → Extract number and UUID
  - User: "okay then Fundi wa stima inaeza kua poa" → Look for "Fundi wa stima" (case-insensitive) → Extract number and UUID

- "ask_question": Factual/clarification questions about the current occupation that don't express concern (e.g., "what does M&E stand for?", "how long is the training?")

- "continue_exploring": They want more detailed information about the current occupation (e.g., "tell me more about the daily tasks", "what else should I know?", "go on", "continue")
  Note: This is ONLY for requesting MORE DETAILS, not for general acknowledgment

- "request_outside_recommendations": They want to pursue an occupation/career that is NOT in the available recommendations or currently being explored
  Examples: "I want to be a DJ actually", "What about becoming a pilot instead?", "I'd rather be a doctor", "Can we look at nursing?"
  Key: They mention a specific occupation/career that is NOT in the list of available occupations above
  IMPORTANT: Set requested_occupation_name to the occupation they mentioned (e.g., "DJ", "pilot", "doctor")

- "other": Unclear or off-topic

CRITICAL DISTINCTIONS:
- "no concerns" / "what next" / "let's move forward" = ACCEPT (ready to proceed with CURRENT occupation)
- "okay" after exploration = ACCEPT (if said without mentioning a DIFFERENT occupation)
- "okay then Fundi wa stima" = EXPLORE_DIFFERENT (switching to different occupation - extract number and UUID!)
- "maybe electrician would be better" = EXPLORE_DIFFERENT (switching to different occupation - extract number and UUID!)
- "I feel incapable" = EXPRESS_CONCERN (self-doubt/barrier)
- "I'm not sure I have the skills" = EXPRESS_CONCERN (self-doubt/barrier)
- "What skills are needed?" = ASK_QUESTION (factual)
- "tell me more about X" = CONTINUE_EXPLORING (wants details about current occupation)
- "Sounds good" = ACCEPT (positive commitment to current occupation)
- "Not for me" = REJECT (negative about current occupation)

PRIORITY RULES:
1. First check: Does user mention ANY occupation name from "Other available occupations" list? If YES → EXPLORE_DIFFERENT (extract number and UUID!)
2. Then check: Are they expressing readiness/acceptance WITHOUT mentioning different occupation? If YES → ACCEPT
3. Otherwise: Follow the intent classification rules above

If they mentioned a different occupation by name or number, you MUST set both target_recommendation_id AND target_occupation_index by looking up the occupation in the "Other available occupations" list above.

Your response must be a JSON object with the following schema:
{{
    "reasoning": "A step by step explanation of why you classified this intent",
    "intent": "One of: express_concern, accept, reject, explore_different, ask_question, continue_exploring, request_outside_recommendations, other",
    "target_recommendation_id": "The UUID of a different occupation if mentioned, or null",
    "target_occupation_index": "The 1-based index of a different occupation if mentioned, or null",
    "requested_occupation_name": "The occupation name if they requested something outside recommendations, or null"
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
- "explore_occupation": They want to learn more about a specific occupation FROM THE LIST
- "show_opportunities": They want to see job postings
- "express_concern": They're expressing worry/doubt about recommendations
- "ask_question": They have a question about something
- "reject": They're rejecting current options
- "accept": They're accepting/interested in an option
- "request_outside_recommendations": They want to pursue an occupation/career that is NOT in the available recommendations above
  Examples: "I want to be a DJ", "What about becoming a pilot?", "I'd rather be a doctor"
  Key: They mention a specific occupation/career that is NOT in the list of available occupations
  IMPORTANT: Set requested_occupation_name to the occupation they mentioned (e.g., "DJ", "pilot", "doctor")
- "other": Unclear or off-topic

If they mentioned a number (1, 2, 3...) or occupation name, identify which one.

Your response must be a JSON object with the following schema:
{{
    "reasoning": "A step by step explanation of why you classified this intent",
    "intent": "One of: explore_occupation, show_opportunities, express_concern, ask_question, reject, accept, request_outside_recommendations, other",
    "target_recommendation_id": "The UUID of the recommendation if identified, or null",
    "target_occupation_index": "The 1-based index number if identified, or null",
    "requested_occupation_name": "The occupation name if they requested something outside recommendations, or null"
}}

Always return a valid JSON object matching this exact schema.
"""

    def _build_concerns_phase_prompt(self, user_input: str, state: RecommenderAdvisorAgentState) -> str:
        """
        Build intent classification prompt for ADDRESS_CONCERNS phase.

        In this phase, we detect if user is:
        - Actually expressing a concern (let resistance classifier handle it)
        - Wanting to switch to a different recommendation
        - Requesting occupation outside recommendations
        - Accepting/understanding and ready to move on
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
                other_occs += f"{i}. {occ.occupation} (uuid: {occ.uuid})\n"

        return f"""
Classify what the user wants to do based on their message.

User said: "{user_input}"

Currently discussing concerns about: {current_occ}

Available occupations:
{other_occs}

Possible intents:
- "express_concern": They're expressing a NEW concern/worry/hesitation about the current occupation
  Examples: "what about X?", "I'm worried about Y", "that sounds hard"
  Key: They're raising a concern/resistance - let the resistance classifier handle this
  NOTE: Return null to let resistance classifier handle it

- "explore_occupation": They want to switch to a DIFFERENT occupation FROM THE LIST
  Examples: "what if i just want to become a boda guy?", "what about the electrician role?", "tell me about option 2"
  Key: They mention a different occupation that IS in the available occupations list

  CRITICAL MATCHING RULES for occupation switching:
  * CASE-INSENSITIVE: "fundi wa stima" = "Fundi wa Stima (Electrician)"
  * PARTIAL NAMES: "electrician" or "boda" or "marine" matches full occupation names
  * IGNORE CONTEXT: "what about Fundi wa stima?" still extracts "Fundi wa stima"
  * EXTRACT BOTH: Set target_occupation_index (number from list) AND target_recommendation_id (UUID)
  * MUST SET BOTH: When you detect occupation switch, you MUST fill both target_occupation_index AND target_recommendation_id

  Examples of extraction:
  - "what about Fundi wa stima?" → Find "Fundi wa Stima" in list → Extract its number and UUID
  - "maybe the boda role" → Find "Boda-Boda" in list → Extract its number and UUID
  - "tell me about option 2" → Extract number 2 and get UUID from list position 2

- "request_outside_recommendations": They want to pursue an occupation NOT in the available recommendations
  Examples: "I want to be a DJ", "What about becoming a pilot?"
  Key: They mention a specific occupation that is NOT in the list above
  IMPORTANT: Set requested_occupation_name

- "accept": They're accepting/understanding the concern discussion and ready to move on
  Examples: "okay", "I understand", "makes sense", "let's continue", "what next"
  Key: ANY statement indicating understanding or readiness to proceed

- "ask_question": They have a clarifying question about current occupation
  Examples: "how long does that take?", "what's the salary?", "where would I work?"

- "other": Unclear

CRITICAL DISTINCTION:
- "what if i just want to become a boda guy?" (boda is in list) = EXPLORE_OCCUPATION
- "I want to be a DJ" (DJ not in list) = REQUEST_OUTSIDE_RECOMMENDATIONS
- "that sounds hard" = EXPRESS_CONCERN (return null to let resistance classifier handle)
- "okay I understand" = ACCEPT

Your response must be a JSON object with the following schema:
{{
    "reasoning": "A step by step explanation of why you classified this intent",
    "intent": "One of: express_concern, explore_occupation, request_outside_recommendations, accept, ask_question, other",
    "target_recommendation_id": "The UUID of a different occupation if mentioned, or null",
    "target_occupation_index": "The 1-based index of a different occupation if mentioned, or null",
    "requested_occupation_name": "The occupation name if they requested something outside recommendations, or null"
}}

Always return a valid JSON object matching this exact schema.
"""

    def _build_generic_prompt(self, user_input: str, state: RecommenderAdvisorAgentState) -> str:
        """
        Build generic intent classification prompt for other phases.
        """
        # Build occupation list if available
        occ_list = ""
        if state.recommendations:
            for i, occ in enumerate(state.recommendations.occupation_recommendations[:5], 1):
                occ_list += f"{i}. {occ.occupation}\n"

        occ_context = f"""
Available recommendations:
{occ_list}
""" if occ_list else ""

        return f"""
Classify what the user wants to do based on their message.

User said: "{user_input}"

{occ_context}
Possible intents:
- "express_concern": They're expressing worry, doubt, or hesitation
- "ask_question": They have a question
- "accept": They're agreeing or accepting something
  Examples: "okay", "I understand", "makes sense", "let's do that", "okay let's compare that"
  Key: ANY statement indicating agreement, understanding, or acceptance
- "reject": They're rejecting or declining something
- "request_outside_recommendations": They want to pursue an occupation/career that is NOT in the available recommendations
  Examples: "I want to be a DJ", "What about becoming a pilot?", "I'd rather be a doctor"
  Key: They mention a specific NEW occupation/career that is NOT in the list above
  CRITICAL: If the user is just agreeing to compare ("okay let's compare", "I understand"), this is NOT a new request - classify as "accept" instead
  IMPORTANT: Set requested_occupation_name to the occupation they mentioned (e.g., "DJ", "pilot", "doctor")
- "other": Unclear or off-topic

CRITICAL DISTINCTION:
- "I want to be a DJ" = REQUEST_OUTSIDE_RECOMMENDATIONS (new request)
- "okay let's compare that" = ACCEPT (agreeing to previous suggestion)
- "I understand" = ACCEPT (acknowledging explanation)
- "tell me more about why" = ASK_QUESTION (wants more details)

Your response must be a JSON object with the following schema:
{{
    "reasoning": "A step by step explanation of why you classified this intent",
    "intent": "One of: express_concern, ask_question, accept, reject, request_outside_recommendations, other",
    "target_recommendation_id": null,
    "target_occupation_index": null,
    "requested_occupation_name": "The occupation name if they requested something outside recommendations, or null"
}}

Always return a valid JSON object matching this exact schema.
"""
