"""
Conversation Manager for Preference Elicitation Agent.

Handles natural language conversation and response generation
for the preference elicitation process.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

from app.agent.llm_caller import LLMCaller
from app.agent.preference_elicitation_agent.types import PreferenceVector
from common_libs.llm.generative_models import GeminiGenerativeLLM
from common_libs.llm.models_utils import (
    LLMConfig,
    LOW_TEMPERATURE_GENERATION_CONFIG,
    JSON_GENERATION_CONFIG
)
from app.agent.prompt_template.agent_prompt_template import (
    STD_AGENT_CHARACTER,
    STD_LANGUAGE_STYLE
)
from app.agent.simple_llm_agent.prompt_response_template import get_json_response_instructions


class ConversationResponse(BaseModel):
    """
    Response model for the conversation LLM.

    Handles presenting vignettes and responding to user input
    in a natural, conversational way.
    """
    reasoning: str
    """Chain of thought reasoning about the response"""

    message: str
    """Message to present to the user"""

    finished: bool
    """Whether the preference elicitation is complete"""

    class Config:
        extra = "forbid"


class PreferenceSummaryGenerator(BaseModel):
    """
    LLM response model for generating preference summary.

    Generates natural, conversational bullet points summarizing
    the user's key job preferences from their preference vector.
    """
    reasoning: str = Field(
        description="Brief reasoning about what stands out in their preferences"
    )

    finished: bool = Field(
        description="Always set to True when summary is generated"
    )

    message: str = Field(
        description="Summary of user's preferences as formatted bullet points (use • for bullets)"
    )

    class Config:
        extra = "forbid"


class ConversationManager:
    """
    Manages conversational interactions for preference elicitation.

    Responsibilities:
    - Generate natural language responses for each conversation phase
    - Format vignettes for presentation
    - Generate preference summaries
    - Build conversation context and system instructions
    """

    def __init__(self):
        """Initialize the conversation manager with LLM."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize LLM
        llm_config = LLMConfig(
            generation_config=LOW_TEMPERATURE_GENERATION_CONFIG | JSON_GENERATION_CONFIG
        )

        conversation_system_instructions = self._build_conversation_system_instructions()
        self._conversation_llm = GeminiGenerativeLLM(
            system_instructions=conversation_system_instructions,
            config=llm_config
        )
        self._conversation_caller: LLMCaller[ConversationResponse] = LLMCaller[ConversationResponse](
            model_response_type=ConversationResponse
        )

    def _build_conversation_system_instructions(self) -> str:
        """
        Build system instructions for the conversation LLM.

        Returns:
            System instruction prompt
        """
        return f"""
{STD_AGENT_CHARACTER}

{STD_LANGUAGE_STYLE}

# Your Role
You are a career guidance assistant helping users explore their job preferences.

# Task
Your task is to guide the user through preference discovery using:
1. Experience-based questions about their past work
2. Vignettes (hypothetical scenarios) where they choose between options
3. Follow-up questions to understand WHY they made certain choices

# Guidelines
- Be conversational, warm, and encouraging
- ONE question or scenario at a time
- Keep messages SHORT (2-3 sentences max)
- Focus on understanding their preferences, NOT giving career advice yet
- Make them feel comfortable sharing honestly
- When presenting vignettes, format them clearly with options A and B
- Don't use complex psychological or technical terminology

{get_json_response_instructions(ConversationResponse)}
"""

    async def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[str] = None
    ) -> ConversationResponse:
        """
        Generate a conversational response.

        Args:
            prompt: The prompt/instruction for the LLM
            conversation_history: Optional conversation context

        Returns:
            ConversationResponse with message and metadata
        """
        message = prompt
        if conversation_history:
            message = f"{conversation_history}\n\n{prompt}"

        response = await self._conversation_caller.call(
            llm=self._conversation_llm,
            message=message
        )
        return response

    async def generate_preference_summary(
        self,
        preference_vector: PreferenceVector
    ) -> tuple[str, list]:
        """
        Generate a natural language summary of preferences.

        Args:
            preference_vector: The user's preference vector

        Returns:
            Tuple of (summary string, LLM stats list)
        """
        formatted_prefs = self._format_preference_vector_for_summary(preference_vector)

        prompt = f"""
The user has completed a preference elicitation conversation. Below is their preference vector with scores and values.

Your task: Generate 3-5 natural, conversational bullet points summarizing what matters most to them in a job.

**Guidelines:**
1. Focus on the STRONGEST signals (high scores >0.7 or low scores <0.3)
2. Combine related preferences naturally (e.g., "flexibility and autonomy" not separate bullets)
3. Include task preferences - they're critical for recommendations
4. Use conversational language, not technical jargon
5. Be specific - reference actual values when they tell a story

**Preference Vector:**
{formatted_prefs}

Generate a summary that captures what's UNIQUE about this user's preferences.
"""

        summary_llm = GeminiGenerativeLLM(
            system_instructions="""
You are a career guidance assistant summarizing user preferences.
Be brief, specific, and conversational.
Use bullet points (•) for clarity.
""",
            config=LLMConfig(
                generation_config=LOW_TEMPERATURE_GENERATION_CONFIG | JSON_GENERATION_CONFIG
            )
        )

        caller: LLMCaller[PreferenceSummaryGenerator] = LLMCaller[PreferenceSummaryGenerator](
            model_response_type=PreferenceSummaryGenerator
        )

        try:
            response, llm_stats = await caller.call_llm(llm=summary_llm, llm_input=prompt, logger=self.logger)
            if response and response.message:
                return response.message, llm_stats
            else:
                self.logger.warning("LLM returned empty summary, using fallback")
                return self._generate_basic_preference_summary(preference_vector), llm_stats
        except Exception as e:
            self.logger.warning(f"LLM summary generation failed: {e}, using basic summary")
            return self._generate_basic_preference_summary(preference_vector), []

    def _format_preference_vector_for_summary(self, pv: PreferenceVector) -> str:
        """Format preference vector in a readable way for LLM."""
        return f"""
Financial:
- Importance: {pv.financial.importance:.2f}
- Benefits importance: {pv.financial.benefits_importance:.2f}
- Bonus/commission tolerance: {pv.financial.bonus_commission_tolerance:.2f}

Work Environment:
- Remote preference: {pv.work_environment.remote_work_preference}
- Flexibility importance: {pv.work_environment.work_hours_flexibility_importance:.2f}
- Autonomy importance: {pv.work_environment.autonomy_importance:.2f}
- Supervision preference: {pv.work_environment.supervision_preference}

Job Security:
- Importance: {pv.job_security.importance:.2f}
- Stability required: {pv.job_security.income_stability_required}
- Risk tolerance: {pv.job_security.risk_tolerance}
- Contract preference: {pv.job_security.contract_type_preference}

Career Advancement:
- Importance: {pv.career_advancement.importance:.2f}
- Learning value: {pv.career_advancement.learning_opportunities_value}
- Skill development: {pv.career_advancement.skill_development_importance:.2f}

Work-Life Balance:
- Importance: {pv.work_life_balance.importance:.2f}
- Max hours/week: {pv.work_life_balance.max_acceptable_hours_per_week or 'Not set'}
- Weekend work: {pv.work_life_balance.weekend_work_tolerance}
- Evening work: {pv.work_life_balance.evening_work_tolerance}

Task Preferences:
- Social tasks: {pv.task_preferences.social_tasks_preference:.2f}
- Cognitive tasks: {pv.task_preferences.cognitive_tasks_preference:.2f}
- Routine tolerance: {pv.task_preferences.routine_tasks_tolerance:.2f}
- Creative tasks: {pv.task_preferences.creative_tasks_preference:.2f}
- Manual tasks: {pv.task_preferences.manual_tasks_preference:.2f}

Overall Confidence: {pv.confidence_score:.2f}
"""

    def _generate_basic_preference_summary(self, pv: PreferenceVector) -> str:
        """Generate a basic preference summary without LLM (fallback)."""
        summary_parts = []

        if pv.financial.importance > 0.7:
            summary_parts.append("• Financial compensation is important to you")

        if pv.job_security.importance > 0.7:
            summary_parts.append("• Job security and stability matter to you")

        if pv.career_advancement.importance > 0.7:
            summary_parts.append("• Career growth is important to you")

        if not summary_parts:
            summary_parts.append("• I've learned about your job preferences")

        return "\n".join(summary_parts)
