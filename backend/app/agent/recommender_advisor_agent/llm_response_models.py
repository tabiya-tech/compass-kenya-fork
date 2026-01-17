"""
LLM Response Models for the Recommender/Advisor Agent.

Pydantic models for structured LLM responses across different phases.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ConversationResponse(BaseModel):
    """
    Response model for the conversation LLM.
    
    Handles presenting recommendations, addressing concerns,
    and guiding users toward action.
    """
    reasoning: str
    """Chain of thought reasoning about the response"""
    
    message: str
    """Message to present to the user"""
    
    finished: bool
    """Whether the recommender session is complete"""
    
    metadata: Optional[dict] = None
    """Optional structured metadata for UI rendering"""
    
    class Config:
        extra = "forbid"


class ResistanceClassification(BaseModel):
    """
    LLM response model for classifying user resistance.
    """
    reasoning: str = Field(
        description="Reasoning about what type of resistance the user is expressing"
    )
    resistance_type: str = Field(
        description="Type of resistance: 'belief', 'salience', 'effort', 'financial', 'circumstantial', 'acceptance', or 'none'"
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
        description="Whether the user made a clear action commitment"
    )
    action_type: Optional[str] = Field(
        default=None,
        description="Type of action: 'apply_to_job', 'enroll_in_training', 'explore_occupation', etc."
    )
    commitment_level: Optional[str] = Field(
        default=None,
        description="Commitment level: 'will_do_this_week', 'will_do_this_month', 'interested', 'maybe_later'"
    )
    barriers_mentioned: list[str] = Field(
        default_factory=list,
        description="Any barriers or concerns the user mentioned"
    )
    
    class Config:
        extra = "forbid"


class UserIntentClassification(BaseModel):
    """
    LLM response model for classifying user intent from their message.
    """
    reasoning: str = Field(
        description="Reasoning about what the user wants to do"
    )
    intent: str = Field(
        description="User intent: 'explore_occupation', 'show_opportunities', 'express_concern', 'ask_question', 'reject', 'accept', 'discuss_next_steps', 'explore_alternatives', 'address_more_concerns', 'other'"
    )
    target_recommendation_id: Optional[str] = Field(
        default=None,
        description="ID of the recommendation they're referring to (if identifiable)"
    )
    target_occupation_index: Optional[int] = Field(
        default=None,
        description="Index (1-based) of occupation if they said a number like '1' or 'first'"
    )
    
    class Config:
        extra = "forbid"
