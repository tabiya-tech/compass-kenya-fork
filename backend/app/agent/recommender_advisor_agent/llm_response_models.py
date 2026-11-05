"""
LLM Response Models for the Recommender/Advisor Agent.

Pydantic models for structured LLM responses across different phases.
They are the response schemas of the structured output of the LLM: the descriptions of the fields guide the LLM.
The order of the fields is important: the reasoning comes first, as it sets the context for the other fields.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ConversationResponse(BaseModel):
    """
    Response model for the conversation LLM.
    
    Handles presenting recommendations, addressing concerns,
    and guiding users toward action.
    """
    reasoning: str = Field(
        description="A step by step explanation of how your message relates to your instructions, "
                    "why you chose the message and why you set the finished flag to the specific value."
    )
    """Chain of thought reasoning about the response"""

    message: str = Field(
        description="Your message to the user."
    )
    """Message to present to the user"""

    finished: bool = Field(
        description="Set to true only if the recommender session is complete (see your instructions about when to set finished), false otherwise."
    )
    """Whether the recommender session is complete"""

    metadata: Optional[dict] = Field(
        default=None,
        description="Structured metadata for UI rendering. It is filled by the application, leave it null."
    )
    """Optional structured metadata for UI rendering"""

    class Config:
        extra = "forbid"


class ResistanceClassification(BaseModel):
    """
    LLM response model for classifying user resistance.
    """
    reasoning: str = Field(
        description="A step by step explanation of what type of resistance (or acceptance) the user is expressing"
    )
    resistance_type: str = Field(
        description="The classification, exactly one of: 'belief', 'salience', 'effort', 'financial', 'circumstantial', 'acceptance' or 'none' "
                    "(see the 'Classification Types' in your instructions)"
    )
    concern_summary: str = Field(
        description="Brief summary of the user's concern or their acceptance signal"
    )

    class Config:
        extra = "forbid"


class ActionExtractionResult(BaseModel):
    """
    LLM response model for extracting user's action commitment.
    """
    reasoning: str = Field(
        description="Reasoning about what action the user is committing to"
    )
    has_commitment: bool = Field(
        description="True if the user made a clear commitment to take action, false otherwise"
    )
    action_type: Optional[str] = Field(
        default=None,
        description="The type of action the user commits to, exactly one of: 'apply_to_job', 'enroll_in_training', "
                    "'explore_occupation', 'research_employer', 'network'. Null if has_commitment is false"
    )
    commitment_level: Optional[str] = Field(
        default=None,
        description="How strong the user's commitment is, exactly one of: 'will_do_this_week', 'will_do_this_month', "
                    "'interested', 'maybe_later', 'not_interested'. Null if has_commitment is false"
    )
    barriers_mentioned: list[str] = Field(
        default_factory=list,
        description="The barriers or concerns the user mentioned, an empty list if none"
    )

    # Implementation-intention plan slots (Change 7 / Change 10). Capture only what the user
    # actually stated this turn - leave null if not mentioned. These drive both the
    # plan-completeness gate that keeps the conversation in action-planning until a concrete
    # plan exists, and the verbatim plan restatement at wrapup.
    plan_when: Optional[str] = Field(
        default=None,
        description="When they'll act, in their words (e.g., 'Thursday morning', 'this Saturday'). Null if not stated."
    )
    plan_where: Optional[str] = Field(
        default=None,
        description="Where they'll act / the specific target (e.g., 'the depot', 'the Brookside posting'). Null if not stated."
    )
    plan_how: Optional[str] = Field(
        default=None,
        description="How they'll do it - transport, what they'll bring or say (e.g., 'matatu, ask the supervisor'). Null if not stated."
    )
    plan_backup: Optional[str] = Field(
        default=None,
        description="The if-then backup for the most likely obstacle (e.g., 'Saturday if the fare is short'). Null if not stated."
    )

    class Config:
        extra = "forbid"


class UserIntentClassification(BaseModel):
    """
    LLM response model for classifying user intent from their message.
    """
    reasoning: str = Field(
        description="A step by step explanation of why you classified this intent"
    )
    intent: str = Field(
        description="The user intent, one of the intents listed under 'Possible intents' in your instructions, "
                    "e.g. 'explore_occupation', 'show_opportunities', 'show_more', 'express_concern', 'ask_question', 'reject', 'accept', "
                    "'discuss_next_steps', 'explore_alternatives', 'address_more_concerns', 'request_outside_recommendations', 'other'"
    )
    target_recommendation_id: Optional[str] = Field(
        default=None,
        description="The UUID of the recommendation (occupation or training) the user refers to, copied from the list in your instructions, "
                    "or null if none is identified"
    )
    target_occupation_index: Optional[int] = Field(
        default=None,
        description="The 1-based index number, from the list in your instructions, of the occupation the user refers to "
                    "(e.g. they said '1', 'first' or named it), or null if none is identified"
    )
    requested_occupation_name: Optional[str] = Field(
        default=None,
        description="The name of the occupation the user requested that is not in the recommendations "
                    "(intent 'request_outside_recommendations', e.g. 'DJ', 'pilot'), or null otherwise"
    )

    class Config:
        extra = "forbid"
