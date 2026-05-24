import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, RootModel

from app.matching.client import MatchingServiceClient
from app.matching.service import MatchingService
from app.matching.matching_types import SkillsVector, PreferenceVector, CompassMatchingResult, MatchingAlgorithmVersion, \
    MatchingRequest, CompassOpportunity


class MatchConcatGeminiCeJobRecommendation(BaseModel):
    rank: int
    rank_cosine: Optional[int] = None
    job_uuid: str
    opportunity_title: str = ""
    employer: Optional[str] = None
    location: Optional[str] = None
    URL: Optional[str] = None
    concat_cosine_similarity: Optional[float] = None
    cross_encoder_logit: Optional[float] = None
    cross_encoder_score: Optional[float] = None


class _Response(BaseModel):
    user_id: str
    n_jobs_scored: int
    n_jobs_active_loaded: int
    concat_gemini_ce_recommendations: List[MatchConcatGeminiCeJobRecommendation]
    config_summary: Dict[str, Any] = Field(default_factory=dict)


class _ResponseList(RootModel[List[_Response]]):
    """The `/match_v3` endpoint returns a list with one entry per user in the request.

    We always send a single user, so the list has at most one element.
    """


def _to_compass_opportunity(rec: MatchConcatGeminiCeJobRecommendation) -> CompassOpportunity:
    return CompassOpportunity(
        uuid=rec.job_uuid,
        rank=rec.rank,
        opportunity_title=rec.opportunity_title,
        url=rec.URL,
        employer=rec.employer,
        location=rec.location,
        final_score=rec.cross_encoder_score,
        raw=rec.model_dump(),
    )


class MatchingServiceV3(MatchingService):
    def __init__(self, client: MatchingServiceClient):
        self._client = client
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def algorithm_version(self) -> MatchingAlgorithmVersion:
        return "v3"

    async def generate_recommendations(self,
                                       youth_id: str,
                                       city: Optional[str],
                                       province: Optional[str],
                                       skills_vector: SkillsVector,
                                       preference_vector: PreferenceVector) -> CompassMatchingResult:
        request = MatchingRequest(
            user_id=youth_id,
            city=city or "",
            province=province or "",
            skills_vector=skills_vector,
            preference_vector=preference_vector,
        )

        response = await self._client.process_request(_ResponseList, "/match_v3", request)
        if not response.root:
            return CompassMatchingResult(user_id=youth_id, algorithm_version="v3")

        first = response.root[0]
        metadata: Dict[str, Any] = {
            "n_jobs_scored": first.n_jobs_scored,
            "n_jobs_active_loaded": first.n_jobs_active_loaded,
        }
        if first.config_summary:
            metadata["config_summary"] = first.config_summary

        opportunities = [_to_compass_opportunity(r) for r in first.concat_gemini_ce_recommendations]
        self._logger.info(f"Found {len(opportunities)} opportunities")
        return CompassMatchingResult(
            user_id=first.user_id or youth_id,
            algorithm_version="v3",
            opportunities=opportunities,
            metadata=metadata,
        )
