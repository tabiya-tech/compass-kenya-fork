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

# Nested models for Jasmin's rich Node2Vec output

class SkillComponent(BaseModel):
    """Skill scoring components from Node2Vec algorithm."""
    loc: float = Field(ge=0.0, le=1.0, description="Location match component")
    ess: float = Field(ge=0.0, le=1.0, description="Essential skills component")
    opt: float = Field(ge=0.0, le=1.0, description="Optional skills component")
    grp: float = Field(ge=0.0, le=1.0, description="Skill group component")

    class Config:
        extra = "allow"  # Allow future additions from algorithm


class ScoreBreakdown(BaseModel):
    """Detailed score breakdown from Node2Vec algorithm."""
    total_skill_utility: float = Field(ge=0.0, le=1.0, description="Combined skill utility score")
    skill_components: SkillComponent = Field(description="Breakdown of skill scoring")
    skill_penalty_applied: float = Field(ge=0.0, description="Penalty if skills below threshold")
    preference_score: float = Field(ge=0.0, le=1.0, description="Preference alignment score")
    demand_score: float = Field(ge=0.0, le=1.0, description="Labor demand score")
    demand_label: str = Field(description="Human-readable demand (e.g., 'High Expected Demand')")

    class Config:
        extra = "allow"


class EssentialSkillMatch(BaseModel):
    """Match between a job's required skill and user's skill."""
    job_skill_id: str = Field(description="Required skill ID from job/occupation")
    job_skill_label: str = Field(description="Required skill name")
    best_user_skill_id: str = Field(description="User's matching skill ID")
    best_user_skill_label: str = Field(description="User's matching skill name")
    similarity: float = Field(ge=0.0, le=1.0, description="Cosine similarity score")
    meets_threshold: bool = Field(description="Whether similarity meets minimum threshold")

    class Config:
        extra = "allow"


class SkillGroupMatch(BaseModel):
    """Skill group/category that matches."""
    skill_group_id: str = Field(description="Skill group ID")
    skill_group_label: str = Field(description="Skill group name")

    class Config:
        extra = "allow"


class MatchedSkills(BaseModel):
    """Detailed skill matching data from Node2Vec."""
    essential_skill_matches: list[EssentialSkillMatch] = Field(
        default_factory=list,
        description="Matches for essential/required skills"
    )
    optional_exact_matches: list[Any] = Field(
        default_factory=list,
        description="Optional skill exact matches (structure TBD)"
    )
    skill_group_matches: list[SkillGroupMatch] = Field(
        default_factory=list,
        description="Skill group/category matches"
    )

    class Config:
        extra = "allow"


class PreferenceMatch(BaseModel):
    """Single preference attribute match from Bayesian preference model."""
    attribute: str = Field(description="Preference dimension (e.g., 'earnings_per_month')")
    job_value: str = Field(description="Job's value for this attribute (e.g., 'earn_70k')")
    job_value_label: str = Field(description="Human-readable label (e.g., '70K')")
    user_weight: float = Field(ge=0.0, le=1.0, description="User's importance weight (0-1)")
    beta: float = Field(description="Bayesian coefficient for this attribute")
    encoded_value: float = Field(ge=0.0, le=1.0, description="Encoded match value")
    contribution: float = Field(description="Contribution to final preference score")
    matched: bool = Field(description="Whether this preference dimension matched")

    class Config:
        extra = "allow"


class OccupationRecommendation(BaseModel):
    """
    Career path recommendation from Node2Vec algorithm.

    Represents a recommended occupation/career based on user's
    skills, preferences, and labor market data.

    Updated to match Jasmin's Node2Vec output format with backward-compatible
    property accessors for agent code.
    """

    # Taxonomy identifiers (from Jasmin's schema)
    uuid: str = Field(description="Taxonomy UUID")
    originUuid: str = Field(description="Taxonomy origin UUID")
    rank: int = Field(ge=1, description="Ranking (1 = best match)")

    # Occupation identification
    occupation_id: str = Field(description="ESCO/KeSCO occupation ID")
    occupation_code: str = Field(description="Occupation code (e.g., '2512')")
    occupation: str = Field(description="Occupation title (e.g., 'Data Analyst')")

    # Eligibility and scoring (Jasmin's format)
    is_eligible: bool = Field(
        default=True,
        description="Whether user meets skill threshold for this occupation"
    )
    final_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Combined final score from Node2Vec algorithm (if available)"
    )
    score_breakdown: Optional[ScoreBreakdown] = Field(
        default=None,
        description="Detailed component scores from Node2Vec (if available)"
    )

    # Rich matching data (Jasmin's format)
    matched_skills: Optional[MatchedSkills] = Field(
        default=None,
        description="Detailed skill match information (if available)"
    )
    matched_preferences: list[PreferenceMatch] = Field(
        default_factory=list,
        description="Detailed preference match breakdown"
    )

    # Justification (Jasmin provides this)
    justification: Optional[str] = Field(
        default=None,
        description="Why this matches the user (from Node2Vec algorithm)"
    )

    # Optional metadata (from DB1/DB2 - may not be available initially)
    description: Optional[str] = Field(
        default=None,
        description="Occupation description (LLM generates if missing)"
    )
    typical_tasks: list[str] = Field(
        default_factory=list,
        description="Typical daily tasks (LLM generates if missing)"
    )
    career_path_next_steps: list[str] = Field(
        default_factory=list,
        description="Next steps in career progression (LLM generates if missing)"
    )
    salary_range: Optional[str] = Field(
        default=None,
        description="Typical salary range (e.g., 'KES 60,000-120,000/month')"
    )

    class Config:
        extra = "allow"  # Allow Node2Vec to add new fields

    # ========== BACKWARD COMPATIBILITY PROPERTIES ==========
    # These provide agent code with familiar field access while using new schema

    @property
    def confidence_score(self) -> float:
        """Alias for final_score for backward compatibility."""
        return self.final_score if self.final_score is not None else 0.5

    @property
    def skills_match_score(self) -> Optional[float]:
        """Extract skills match from score_breakdown for agent use."""
        return self.score_breakdown.total_skill_utility if self.score_breakdown else None

    @property
    def preference_match_score(self) -> Optional[float]:
        """Extract preference match from score_breakdown for agent use."""
        return self.score_breakdown.preference_score if self.score_breakdown else None

    @property
    def labor_demand_score(self) -> Optional[float]:
        """Extract labor demand from score_breakdown for agent use."""
        return self.score_breakdown.demand_score if self.score_breakdown else None

    @property
    def labor_demand_category(self) -> Optional[Literal["high", "medium", "low"]]:
        """Extract and normalize demand category from score_breakdown."""
        if not self.score_breakdown:
            return None
        label = self.score_breakdown.demand_label.lower()
        if "high" in label:
            return "high"
        elif "moderate" in label or "medium" in label:
            return "medium"
        else:
            return "low"

    @property
    def essential_skills(self) -> list[str]:
        """Extract skill labels from matched_skills for agent use."""
        if not self.matched_skills:
            return []
        return [match.job_skill_label for match in self.matched_skills.essential_skill_matches]

    @property
    def skill_gaps(self) -> list[str]:
        """Extract unmet skills from matched_skills for agent use."""
        if not self.matched_skills:
            return []
        return [
            match.job_skill_label
            for match in self.matched_skills.essential_skill_matches
            if not match.meets_threshold
        ]

    @property
    def user_skill_coverage(self) -> float:
        """Calculate skill coverage from matched_skills for agent use."""
        if not self.matched_skills or not self.matched_skills.essential_skill_matches:
            return 0.0
        met = sum(1 for m in self.matched_skills.essential_skill_matches if m.meets_threshold)
        return met / len(self.matched_skills.essential_skill_matches)


class OpportunityRecommendation(BaseModel):
    """
    Actual job posting / internship recommendation.

    Represents a specific job opportunity that matches the user's
    profile and recommended occupation paths.

    Updated to match Jasmin's Node2Vec output format with rich scoring
    and matching data.
    """

    # Taxonomy identifiers (from Jasmin's schema)
    uuid: str = Field(description="Taxonomy UUID")
    originUuid: str = Field(description="Taxonomy origin UUID")
    rank: int = Field(ge=1, description="Ranking (1 = best match)")

    # Opportunity details
    opportunity_title: str = Field(
        description="Job title (e.g., 'Senior Python Developer')"
    )
    location: str = Field(
        description="Location (e.g., 'Nairobi' or 'Remote')"
    )

    # Eligibility and scoring (Jasmin's format)
    is_eligible: bool = Field(
        default=True,
        description="Whether user meets skill threshold for this opportunity"
    )
    final_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Combined final score from Node2Vec algorithm (if available)"
    )
    score_breakdown: Optional[ScoreBreakdown] = Field(
        default=None,
        description="Detailed component scores from Node2Vec (if available)"
    )

    # Rich matching data (Jasmin's format)
    matched_skills: Optional[MatchedSkills] = Field(
        default=None,
        description="Detailed skill match information (if available)"
    )
    matched_preferences: list[PreferenceMatch] = Field(
        default_factory=list,
        description="Detailed preference match breakdown"
    )

    # Justification (Jasmin provides this)
    justification: Optional[str] = Field(
        default=None,
        description="Why this matches the user (from Node2Vec algorithm)"
    )

    # Contract type (from Jasmin's output)
    contract_type: Optional[Literal["full_time", "part_time", "internship", "contract", "freelance", "commission"]] = Field(
        default=None,
        description="Employment contract type"
    )

    # Optional metadata (from DB3 - jobs database, may not be available initially)
    employer: Optional[str] = Field(
        default=None,
        description="Employer/company name"
    )
    salary_range: Optional[str] = Field(
        default=None,
        description="Salary range if available"
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
        extra = "allow"  # Allow Node2Vec to add new fields

    # ========== BACKWARD COMPATIBILITY PROPERTIES ==========

    @property
    def essential_skills(self) -> list[str]:
        """Extract skill labels from matched_skills for agent use."""
        if not self.matched_skills:
            return []
        return [match.job_skill_label for match in self.matched_skills.essential_skill_matches]


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

    Updated to match Jasmin's actual output format with converter.
    """

    youth_id: str = Field(description="User/youth identifier", alias="user_id")
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
        description="Career path recommendations (from Jasmin - coming next week)"
    )
    opportunity_recommendations: list[OpportunityRecommendation] = Field(
        default_factory=list,
        description="Actual job posting recommendations (from Jasmin)"
    )
    skillstraining_recommendations: list[SkillsTrainingRecommendation] = Field(
        default_factory=list,
        description="Training/course recommendations (mapped from skill gaps)"
    )

    # Raw skill gap data from Jasmin (before training mapping)
    skill_gap_recommendations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw skill gap data from Node2Vec (skill_id, skill_label, scores, reasoning)"
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
        extra = "allow"  # Allow Node2Vec to add new fields
        populate_by_name = True  # Allow alias (user_id → youth_id)

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

    @classmethod
    def from_jasmin_output(cls, jasmin_data: dict | list) -> "Node2VecRecommendations":
        """
        Convert Jasmin's Node2Vec output format to agent format.

        Jasmin's format: [{user_id, opportunity_recommendations, skill_gap_recommendations}]
        Agent format: {youth_id, occupation_recommendations, opportunity_recommendations, skillstraining_recommendations}

        Args:
            jasmin_data: Raw output from Jasmin's Node2Vec algorithm
                Can be array of user objects or single user object

        Returns:
            Node2VecRecommendations object ready for agent use

        Note:
            - Jasmin outputs an array of user objects, we extract the first one
            - skill_gap_recommendations are converted to skillstraining_recommendations
            - occupation_recommendations will be empty until Jasmin implements them next week
        """
        # Import here to avoid circular dependency
        from app.agent.recommender_advisor_agent.recommendation_interface import convert_skill_gaps_to_trainings

        # Handle array wrapper
        if isinstance(jasmin_data, list) and len(jasmin_data) > 0:
            jasmin_data = jasmin_data[0]
        elif isinstance(jasmin_data, list) and len(jasmin_data) == 0:
            raise ValueError("Empty recommendations array from Node2Vec")

        # Extract raw data
        youth_id = jasmin_data.get("user_id", "unknown")
        opportunities = jasmin_data.get("opportunity_recommendations", [])
        skill_gaps = jasmin_data.get("skill_gap_recommendations", [])
        occupations = jasmin_data.get("occupation_recommendations", [])  # Empty for now

        # Convert skill gaps to training recommendations
        trainings = convert_skill_gaps_to_trainings(skill_gaps)

        return cls(
            youth_id=youth_id,
            occupation_recommendations=occupations,
            opportunity_recommendations=opportunities,
            skillstraining_recommendations=trainings,
            skill_gap_recommendations=skill_gaps,  # Keep raw data for reference
            algorithm_version=jasmin_data.get("algorithm_version", "node2vec_v1"),
            confidence=jasmin_data.get("confidence", 0.8)
        )


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
