"""
Service layer for user recommendations.

Used by the agent director and conversation flow to check if a user can skip
to the recommendation phase. Injected via get_user_recommendations_service.
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional

from app.user_recommendations.repository.repository import IUserRecommendationsRepository
from app.user_recommendations.types import UserRecommendations


class IUserRecommendationsService(ABC):
    """Interface for the user recommendations service."""

    @abstractmethod
    async def upsert(self, user_id: str, data: UserRecommendations) -> None:
        """
        Create or update recommendations for a user.

        :param user_id: The user ID (must match data.user_id)
        :param data: The recommendations to store
        :raises ValueError: If data.user_id does not match user_id
        """
        raise NotImplementedError()

    @abstractmethod
    async def get_by_user_id(self, user_id: str) -> Optional[UserRecommendations]:
        """
        Get recommendations for a user.

        :param user_id: The user ID to look up
        :return: The recommendations if found, None otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    async def has_recommendations(self, user_id: str) -> bool:
        """
        Check whether a user has any recommendations.

        :param user_id: The user ID to check
        :return: True if the user has recommendations, False otherwise
        """
        raise NotImplementedError()


class UserRecommendationsService(IUserRecommendationsService):
    """Implementation of the user recommendations service."""

    def __init__(self, repository: IUserRecommendationsRepository):
        self._repository = repository
        self._logger = logging.getLogger(self.__class__.__name__)

    async def upsert(self, user_id: str, data: UserRecommendations) -> None:
        if data.user_id != user_id:
            raise ValueError(f"user_id mismatch: {data.user_id} != {user_id}")
        await self._repository.upsert(user_id, data)

    async def get_by_user_id(self, user_id: str) -> Optional[UserRecommendations]:
        return await self._repository.get_by_user_id(user_id)

    async def has_recommendations(self, user_id: str) -> bool:
        rec = await self._repository.get_by_user_id(user_id)
        return rec is not None and (
            len(rec.occupation_recommendations) > 0
            or len(rec.opportunity_recommendations) > 0
            or len(rec.skill_gap_recommendations) > 0
        )
