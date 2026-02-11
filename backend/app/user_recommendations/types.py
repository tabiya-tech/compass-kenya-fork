from typing import Any, Optional
from pydantic import BaseModel, Field


class SkillComponentDB(BaseModel):
    """Skill score components (loc=location, ess=essential, opt=optional, grp=group)."""
    loc: float = 0.0
    """Location-related skill score component."""
    ess: float = 0.0
    """Essential skill score component."""
    opt: float = 0.0
    """Optional skill score component."""
    grp: float = 0.0
    """Skill group score component."""

    class Config:
        extra = "allow"


class ScoreBreakdownDB(BaseModel):
    """Score breakdown for a recommendation (skill utility, preference, demand)."""
    total_skill_utility: float = 0.0
    """Total skill utility contribution to the score."""
    skill_components: SkillComponentDB = Field(default_factory=SkillComponentDB)
    """Breakdown of skill score by component (loc, ess, opt, grp)."""
    skill_penalty_applied: float = 0.0
    """Penalty applied to the skill score."""
    preference_score: float = 0.0
    """Preference match contribution to the score."""
    demand_score: float = 0.0
    """Labour market demand contribution to the score."""
    demand_label: str = ""
    """Label for the demand level (e.g. high, medium, low)."""

    class Config:
        extra = "allow"


class EssentialSkillMatchDB(BaseModel):
    """Match between a job skill and the user's best matching skill."""
    job_skill_id: str = ""
    """ID of the skill required by the job."""
    job_skill_label: str = ""
    """Human-readable label of the job skill."""
    best_user_skill_id: str = ""
    """ID of the user's best matching skill."""
    best_user_skill_label: str = ""
    """Human-readable label of the user's skill."""
    similarity: float = 0.0
    """Similarity score between job skill and user skill."""
    meets_threshold: bool = False
    """Whether the match meets the eligibility threshold."""

    class Config:
        extra = "allow"


class SkillGroupMatchDB(BaseModel):
    """Match to a skill group when no exact skill match exists."""
    skill_group_id: str = ""
    """ID of the matched skill group."""
    skill_group_label: str = ""
    """Human-readable label of the skill group."""

    class Config:
        extra = "allow"


class MatchedSkillsDB(BaseModel):
    """All skill matches for a recommendation (essential, optional, groups)."""
    essential_skill_matches: list[EssentialSkillMatchDB] = Field(default_factory=list)
    """Matches for required/essential skills."""
    optional_exact_matches: list[Any] = Field(default_factory=list)
    """Exact matches for optional skills."""
    skill_group_matches: list[SkillGroupMatchDB] = Field(default_factory=list)
    """Matches to skill groups when no exact skill match exists."""

    class Config:
        extra = "allow"


class PreferenceMatchDB(BaseModel):
    """Preference match between a job attribute and user preferences."""
    attribute: str = ""
    """Name of the preference attribute."""
    job_value: str = ""
    """Raw value of the attribute in the job."""
    job_value_label: str = ""
    """Human-readable label of the job value."""
    user_weight: float = 0.0
    """Weight the user assigned to this preference."""
    beta: float = 0.0
    """Preference model parameter."""
    encoded_value: float = 0.0
    """Encoded value for scoring."""
    contribution: float = 0.0
    """This match's contribution to the total preference score."""
    matched: bool = False
    """Whether the job value matches the user's preference."""

    class Config:
        extra = "allow"


class OccupationRecommendationDB(BaseModel):
    """A recommended occupation for the user."""
    uuid: str = ""
    """Unique identifier for this recommendation."""
    originUuid: str = ""
    """UUID of the source occupation in the taxonomy."""
    rank: int = 1
    """Ranking order (1 = top recommendation)."""
    occupation_label: str = ""
    """Human-readable occupation name."""
    province: Optional[str] = None
    """Province or region for the occupation."""
    is_eligible: bool = True
    """Whether the user is eligible for this occupation."""
    justification: str = ""
    """Explanation of why this occupation was recommended."""
    occupation_description: str = ""
    """Full description of the occupation."""
    final_score: float = 0.0
    """Combined recommendation score."""
    score_breakdown: Optional[ScoreBreakdownDB] = None
    """Detailed breakdown of how the score was computed."""
    matched_skills: Optional[MatchedSkillsDB] = None
    """Skills matched between user and occupation."""
    matched_preferences: list[PreferenceMatchDB] = Field(default_factory=list)
    """Preference matches for this occupation."""

    class Config:
        extra = "allow"


class OpportunityRecommendationDB(BaseModel):
    """A recommended job opportunity for the user."""
    uuid: str = ""
    """Unique identifier for this recommendation."""
    URL: Optional[str] = None
    """Link to the job posting."""
    rank: int = 1
    """Ranking order (1 = top recommendation)."""
    opportunity_title: str = ""
    """Job title."""
    location: str = ""
    """Job location."""
    is_eligible: bool = True
    """Whether the user is eligible for this opportunity."""
    justification: str = ""
    """Explanation of why this opportunity was recommended."""
    opportunity_description: Optional[str] = None
    """Full description of the job."""
    contract_type: Optional[str] = None
    """Type of contract (e.g. full-time, part-time)."""
    final_score: float = 0.0
    """Combined recommendation score."""
    score_breakdown: Optional[ScoreBreakdownDB] = None
    """Detailed breakdown of how the score was computed."""
    matched_skills: Optional[MatchedSkillsDB] = None
    """Skills matched between user and opportunity."""
    matched_preferences: list[PreferenceMatchDB] = Field(default_factory=list)
    """Preference matches for this opportunity."""

    class Config:
        extra = "allow"


class SkillGapRecommendationDB(BaseModel):
    """A skill the user could acquire to unlock more job options."""
    skill_id: str = ""
    """ID of the skill in the taxonomy."""
    skill_label: str = ""
    """Human-readable skill name."""
    proximity_score: float = 0.0
    """How closely related this skill is to the user's current skills."""
    job_unlock_count: int = 0
    """Number of additional jobs this skill would unlock."""
    combined_score: float = 0.0
    """Combined score for recommending this skill gap."""
    reasoning: str = ""
    """Explanation of why this skill was recommended."""

    class Config:
        extra = "allow"


class UserRecommendations(BaseModel):
    """
    Recommendations for a user, stored in the recommendation DB.

    Injected by cron (e.g. from Node2Vec, Kobotoolbox). Used to determine
    if a user can skip to the recommendation phase.
    """

    user_id: str = ""
    """
    The user ID these recommendations belong to.
    """

    occupation_recommendations: list[OccupationRecommendationDB] = Field(default_factory=list)
    """
    Recommended occupations based on the user's profile.
    """

    opportunity_recommendations: list[OpportunityRecommendationDB] = Field(default_factory=list)
    """
    Recommended job opportunities.
    """

    skill_gap_recommendations: list[SkillGapRecommendationDB] = Field(default_factory=list)
    """
    Skills the user could acquire to unlock more options.
    """

    class Config:
        extra = "allow"
