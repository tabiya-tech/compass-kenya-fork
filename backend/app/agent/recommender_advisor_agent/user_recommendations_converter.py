"""
Convert from cron/Kobotoolbox to Node2Vec (agent format) to be used as input
when the user skips to the recommendation phase.
"""

from app.user_recommendations.types import (
    UserRecommendations,
    OccupationRecommendationDB,
    OpportunityRecommendationDB,
    SkillGapRecommendationDB,
)
from app.agent.recommender_advisor_agent.types import (
    Node2VecRecommendations,
    OccupationRecommendation,
    OpportunityRecommendation,
)
from app.agent.recommender_advisor_agent.recommendation_interface import convert_skill_gaps_to_trainings


def _occupation_db_to_agent(db: OccupationRecommendationDB) -> OccupationRecommendation:
    return OccupationRecommendation(
        uuid=db.uuid or f"occ_{db.rank}",
        originUuid=db.originUuid or "",
        rank=db.rank,
        occupation_id=db.uuid or "",
        occupation_code="",
        occupation=db.occupation_label or "Unknown",
        is_eligible=db.is_eligible,
        final_score=db.final_score if db.final_score else None,
        justification=db.justification or None,
        description=db.occupation_description or None,
    )


def _opportunity_db_to_agent(db: OpportunityRecommendationDB) -> OpportunityRecommendation:
    origin_uuid = getattr(db, "originUuid", None) or db.uuid or ""
    return OpportunityRecommendation(
        uuid=db.uuid or f"job_{db.rank}",
        originUuid=origin_uuid,
        rank=db.rank,
        opportunity_title=db.opportunity_title or "Job opportunity",
        location=db.location or "",
        is_eligible=db.is_eligible,
        final_score=db.final_score if db.final_score else None,
        justification=db.justification or None,
        contract_type=db.contract_type,
        posting_url=getattr(db, "URL", None),
    )


def _skill_gap_db_to_dict(db: SkillGapRecommendationDB) -> dict:
    return {
        "skill_id": db.skill_id,
        "skill_label": db.skill_label,
        "proximity_score": db.proximity_score,
        "job_unlock_count": db.job_unlock_count,
        "combined_score": db.combined_score,
        "reasoning": db.reasoning,
    }


def user_recommendations_to_node2vec(
    user_id: str, db_recs: UserRecommendations
) -> Node2VecRecommendations:
    """
    Convert UserRecommendations from the DB to Node2VecRecommendations for the agent.
    """
    occupations = [_occupation_db_to_agent(o) for o in db_recs.occupation_recommendations]
    opportunities = [_opportunity_db_to_agent(o) for o in db_recs.opportunity_recommendations]
    skill_gaps_raw = [_skill_gap_db_to_dict(s) for s in db_recs.skill_gap_recommendations]
    skillstraining = convert_skill_gaps_to_trainings(skill_gaps_raw)

    return Node2VecRecommendations(
        user_id=user_id,
        occupation_recommendations=occupations,
        opportunity_recommendations=opportunities,
        skillstraining_recommendations=skillstraining,
        skill_gap_recommendations=skill_gaps_raw,
        algorithm_version="user_recommendations_db",
        confidence=0.8,
    )
