"""
Data models for the Recommender/Advisor Agent.

This module defines the core data structures used for:
- Input from Node2Vec algorithm (occupation, opportunity, training recommendations)
- User engagement tracking (interest signals, resistance types)
- Action commitments (what the user commits to doing)

Epic 3: Recommender Agent Implementation
"""

from typing import Any, Literal, Optional
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, field_validator, field_serializer


# ========== ENUMS ==========

class ConversationPhase(str, Enum):
    """Phases of the recommender/advisor conversation flow."""
    
    INTRO = "INTRO"
    """Explain what's coming, set expectations"""
    
    PRESENT_RECOMMENDATIONS = "PRESENT"
    """Show top 3-5 occupation recommendations"""
    
    CAREER_EXPLORATION = "EXPLORATION"
    """Deep-dive on a specific occupation the user is interested in"""
    
    ADDRESS_CONCERNS = "CONCERNS"
    """Handle user resistance/objections"""
    
    DISCUSS_TRADEOFFS = "TRADEOFFS"
    """Balance preference vs labor demand"""
    
    FOLLOW_UP = "FOLLOW_UP"
    """Clarify user responses, handle ambiguity"""
    
    SKILLS_UPGRADE_PIVOT = "SKILLS_UPGRADE"
    """User rejected all occupations → present training path"""
    
    ACTION_PLANNING = "ACTION"
    """Concrete next steps (apply, enroll, explore)"""
    
    WRAPUP = "WRAPUP"
    """Summarize, confirm plan, save to DB6"""
    
    COMPLETE = "COMPLETE"
    """Session finished"""


class ResistanceType(str, Enum):
    """Types of user resistance to recommendations."""
    
    BELIEF_BASED = "belief"
    """'I don't think I could succeed', 'There are no jobs'"""
    
    SALIENCE_BASED = "salience"
    """'It doesn't feel like real work', 'My family won't respect this'"""
    
    EFFORT_BASED = "effort"
    """'Applications are exhausting', 'I'll get rejected anyway'"""
    
    FINANCIAL = "financial"
    """'The pay is too low', 'I can't afford to take this'"""
    
    CIRCUMSTANTIAL = "circumstantial"
    """'I can't relocate', 'The hours don't work for me'"""


class UserInterestLevel(str, Enum):
    """User interest level in a recommendation."""
    
    INTERESTED = "interested"
    """Expressed interest, wants to know more"""
    
    EXPLORING = "exploring"
    """Currently in deep-dive discussion"""
    
    NEUTRAL = "neutral"
    """No strong signal either way"""
    
    REJECTED = "rejected"
    """Explicitly rejected this option"""
    
    COMMITTED = "committed"
    """Committed to taking action on this"""


class CommitmentLevel(str, Enum):
    """Level of commitment to an action."""
    
    WILL_DO_THIS_WEEK = "will_do_this_week"
    """Strong commitment with immediate timeline"""
    
    WILL_DO_THIS_MONTH = "will_do_this_month"
    """Commitment with near-term timeline"""
    
    INTERESTED = "interested"
    """Interested but no timeline commitment"""
    
    MAYBE_LATER = "maybe_later"
    """Tentative, no clear commitment"""
    
    NOT_INTERESTED = "not_interested"
    """Declined to commit"""


class ActionType(str, Enum):
    """Types of actions user can commit to."""

    APPLY_TO_JOB = "apply_to_job"
    """Submit job application"""

    ENROLL_IN_TRAINING = "enroll_in_training"
    """Enroll in training course"""

    EXPLORE_OCCUPATION = "explore_occupation"
    """Research occupation further"""

    RESEARCH_EMPLOYER = "research_employer"
    """Learn more about specific employer"""

    NETWORK = "network"
    """Reach out to contacts in the field"""


# ========== RECOMMENDATION MODELS (from Node2Vec) ==========

class OccupationRecommendation(BaseModel):
    """
    Career path recommendation from Node2Vec algorithm.

    Represents a recommended occupation/career based on user's
    skills, preferences, and labor market data.
    """

    # Taxonomy identifiers (updated to use uuid/originUuid per Jasmin's schema)
    uuid: str = Field(description="Taxonomy UUID")
    originUuid: str = Field(description="Taxonomy origin UUID")
    rank: int = Field(ge=1, description="Ranking (1 = best match)")

    # Occupation identification
    occupation_id: str = Field(description="ESCO/KeSCO occupation ID")
    occupation_code: str = Field(description="Occupation code (e.g., '2512')")
    occupation: str = Field(description="Occupation title (e.g., 'Data Analyst')")

    # Match quality
    confidence_score: float = Field(ge=0.0, le=1.0, description="Match confidence (0-1)")
    justification: Optional[str] = Field(
        default=None,
        description="Why this matches the user (optional - LLM generates if missing)"
    )

    # Score breakdown components (from Node2Vec algorithm)
    skills_match_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Component score for skills alignment"
    )
    preference_match_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Component score for preference alignment"
    )
    labor_demand_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Component score for labor market demand"
    )
    graph_proximity_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Component score for graph-based similarity"
    )
    
    # Skills alignment
    essential_skills: list[str] = Field(
        default_factory=list,
        description="Skills needed for this occupation"
    )
    user_skill_coverage: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Pct of required skills user already has"
    )
    skill_gaps: list[str] = Field(
        default_factory=list,
        description="Skills user would need to develop"
    )
    
    # Optional metadata (from DB1/DB2)
    description: Optional[str] = Field(
        default=None,
        description="Occupation description"
    )
    typical_tasks: list[str] = Field(
        default_factory=list,
        description="Typical daily tasks"
    )
    career_path_next_steps: list[str] = Field(
        default_factory=list,
        description="Next steps in career progression"
    )
    labor_demand_category: Optional[Literal["high", "medium", "low"]] = Field(
        default=None,
        description="Current labor market demand"
    )
    salary_range: Optional[str] = Field(
        default=None,
        description="Typical salary range (e.g., 'KES 60,000-120,000/month')"
    )
    
    class Config:
        extra = "forbid"


class OpportunityRecommendation(BaseModel):
    """
    Actual job posting / internship recommendation.

    Represents a specific job opportunity that matches the user's
    profile and recommended occupation paths.
    """

    # Taxonomy identifiers (updated to use uuid/originUuid per Jasmin's schema)
    uuid: str = Field(description="Taxonomy UUID")
    originUuid: str = Field(description="Taxonomy origin UUID")
    rank: int = Field(ge=1, description="Ranking (1 = best match)")
    
    # Opportunity details
    opportunity_title: str = Field(
        description="Job title (e.g., 'Internship at XYZ Foundation')"
    )
    location: str = Field(
        description="Location (e.g., 'Nairobi' or 'Remote')"
    )
    
    # Match quality
    justification: Optional[str] = Field(
        default=None,
        description="Why this matches the user (optional - LLM generates if missing)"
    )
    essential_skills: list[str] = Field(
        default_factory=list,
        description="Skills required for this opportunity"
    )
    
    # Optional metadata (from DB3 - jobs database)
    employer: Optional[str] = Field(
        default=None,
        description="Employer/company name"
    )
    salary_range: Optional[str] = Field(
        default=None,
        description="Salary range if available"
    )
    contract_type: Optional[Literal["full_time", "part_time", "internship", "contract", "freelance"]] = Field(
        default=None,
        description="Employment contract type"
    )
    posting_url: Optional[str] = Field(
        default=None,
        description="Link to job posting"
    )
    posted_date: Optional[str] = Field(
        default=None,
        description="When the job was posted"
    )
    application_deadline: Optional[str] = Field(
        default=None,
        description="Application deadline if specified"
    )
    
    # Linkage to occupation recommendations
    related_occupation_id: Optional[str] = Field(
        default=None,
        description="ID of related occupation recommendation"
    )
    
    class Config:
        extra = "forbid"


class SkillsTrainingRecommendation(BaseModel):
    """
    Training course / skills development recommendation.

    Represents a training opportunity that can help the user
    develop skills needed for their target occupations.

    NOTE: This may not be available in first iteration - agent must handle gracefully.
    """

    # Taxonomy identifiers (updated to use uuid/originUuid per Jasmin's schema)
    uuid: str = Field(description="Taxonomy UUID")
    originUuid: str = Field(description="Taxonomy origin UUID")
    rank: int = Field(ge=1, description="Ranking (1 = most relevant)")

    # Training details
    skill: str = Field(
        description="Skill name (e.g., 'Advanced Econometrics')"
    )
    training_title: Optional[str] = Field(
        default=None,
        description="Course/training name"
    )

    # Match quality
    justification: Optional[str] = Field(
        default=None,
        description="Why this training is relevant (optional - LLM generates if missing)"
    )

    # Optional metadata (from DB4 - training database)
    # NOTE: provider is now optional (may not have this data in v1)
    provider: Optional[str] = Field(
        default=None,
        description="Training provider (e.g., 'Coursera', 'ALX')"
    )
    estimated_hours: Optional[int] = Field(
        default=None, ge=0,
        description="Estimated hours to complete"
    )
    cost: Optional[str] = Field(
        default=None,
        description="Cost (e.g., 'Free', 'KES 5,000')"
    )
    location: Optional[str] = Field(
        default=None,
        description="Location (e.g., 'Online', 'Nairobi')"
    )
    delivery_mode: Optional[Literal["online", "in_person", "hybrid"]] = Field(
        default=None,
        description="How training is delivered"
    )
    target_occupations: list[str] = Field(
        default_factory=list,
        description="Occupations this training unlocks/supports"
    )
    enrollment_url: Optional[str] = Field(
        default=None,
        description="Link to enroll"
    )
    
    # Skill gap coverage
    fills_gap_for: list[str] = Field(
        default_factory=list,
        description="Occupation IDs this training fills skill gaps for"
    )
    
    class Config:
        extra = "forbid"


class Node2VecRecommendations(BaseModel):
    """
    Complete output from Jasmin's Node2Vec algorithm.
    
    Contains all three types of recommendations:
    - Occupation recommendations (career paths)
    - Opportunity recommendations (actual job postings)
    - Skills training recommendations (courses/training)
    """
    
    youth_id: str = Field(description="User/youth identifier")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When recommendations were generated"
    )
    recommended_by: list[str] = Field(
        default_factory=lambda: ["Algorithm"],
        description="Who/what generated this (e.g., ['Algorithm'] or ['Human', 'Algorithm'])"
    )
    
    # Three types of recommendations
    occupation_recommendations: list[OccupationRecommendation] = Field(
        default_factory=list,
        description="Career path recommendations"
    )
    opportunity_recommendations: list[OpportunityRecommendation] = Field(
        default_factory=list,
        description="Actual job posting recommendations"
    )
    skillstraining_recommendations: list[SkillsTrainingRecommendation] = Field(
        default_factory=list,
        description="Training/course recommendations"
    )
    
    # Algorithm metadata
    algorithm_version: str = Field(
        default="node2vec_v1",
        description="Version of recommendation algorithm"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Overall confidence in recommendations"
    )
    
    class Config:
        extra = "forbid"
    
    @field_serializer("generated_at")
    def serialize_generated_at(self, dt: datetime) -> str:
        return dt.isoformat()
    
    @field_validator("generated_at", mode='before')
    @classmethod
    def deserialize_generated_at(cls, value: str | datetime) -> datetime:
        if isinstance(value, str):
            dt = datetime.fromisoformat(value)
        else:
            dt = value
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ========== USER ENGAGEMENT TRACKING ==========

class ConcernRecord(BaseModel):
    """Record of a concern raised by the user about a recommendation."""
    
    item_id: str = Field(description="ID of recommendation concern is about")
    item_type: Literal["occupation", "opportunity", "training"] = Field(
        description="Type of recommendation"
    )
    concern: str = Field(description="User's stated concern")
    resistance_type: ResistanceType = Field(description="Category of resistance")
    addressed: bool = Field(default=False, description="Whether concern was addressed")
    response_given: Optional[str] = Field(
        default=None,
        description="Agent's response to the concern"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    class Config:
        extra = "forbid"


class ActionCommitment(BaseModel):
    """
    User's commitment to take action on a recommendation.
    
    Tracks what action they'll take, when, and any barriers mentioned.
    """
    
    recommendation_id: str = Field(
        description="ID of recommendation (occupation, opportunity, or training)"
    )
    recommendation_type: Literal["occupation", "opportunity", "training"] = Field(
        description="Type of recommendation"
    )
    recommendation_title: str = Field(
        description="Human-readable title of what they're committing to"
    )
    
    action_type: ActionType = Field(description="Type of action committed to")
    commitment_level: CommitmentLevel = Field(description="Strength of commitment")
    
    # Details
    barriers_mentioned: list[str] = Field(
        default_factory=list,
        description="Barriers user mentioned that might prevent action"
    )
    specific_opportunity: Optional[str] = Field(
        default=None,
        description="Specific job posting ID if they picked one"
    )
    specific_training: Optional[str] = Field(
        default=None,
        description="Specific training ID if they picked one"
    )
    
    # Timestamps
    committed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    class Config:
        extra = "forbid"
    
    @field_serializer("committed_at")
    def serialize_committed_at(self, dt: datetime) -> str:
        return dt.isoformat()
    
    @field_validator("committed_at", mode='before')
    @classmethod
    def deserialize_committed_at(cls, value: str | datetime) -> datetime:
        if isinstance(value, str):
            dt = datetime.fromisoformat(value)
        else:
            dt = value
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    
    @field_serializer("action_type")
    def serialize_action_type(self, action_type: ActionType) -> str:
        return action_type.value
    
    @field_serializer("commitment_level")
    def serialize_commitment_level(self, level: CommitmentLevel) -> str:
        return level.value


# ========== DB6 SESSION LOG SCHEMA ==========

class RecommenderSessionLog(BaseModel):
    """
    Complete session log saved to DB6 (Youth Database).
    
    Tracks all recommendations shown, user reactions,
    concerns addressed, and final action commitment.
    """
    
    session_id: str = Field(description="Unique session identifier")
    youth_id: str = Field(description="User/youth identifier")
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    # Recommendations presented
    recommendations_presented: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "occupations": [],
            "opportunities": [],
            "trainings": []
        },
        description="IDs of recommendations presented by type"
    )
    
    # User engagement
    user_engagement: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "occupations_explored": [],
            "occupations_rejected": [],
            "opportunities_explored": [],
            "opportunities_rejected": [],
            "trainings_explored": [],
            "trainings_rejected": []
        },
        description="User engagement by type and action"
    )
    
    # Concerns
    concerns_raised: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All concerns raised during session"
    )
    concerns_addressed: int = Field(
        default=0, ge=0,
        description="Number of concerns addressed"
    )
    
    # Final outcome
    action_commitment: Optional[ActionCommitment] = Field(
        default=None,
        description="User's final action commitment"
    )
    
    # Session metrics
    turns_count: int = Field(default=0, ge=0)
    pivoted_to_training: bool = Field(default=False)
    recommendation_flow: list[str] = Field(
        default_factory=list,
        description="Path taken (e.g., ['occupation', 'opportunity', 'action'])"
    )
    
    class Config:
        extra = "forbid"
    
    @field_serializer("completed_at")
    def serialize_completed_at(self, dt: datetime) -> str:
        return dt.isoformat()
