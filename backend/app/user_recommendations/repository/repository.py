"""
Repository layer for user recommendations.

Populated by cron jobs (e.g. from Node2Vec, Kobotoolbox). Used to check if a user
can skip skills elicitation and preference elicitation and go straight to recommendations.
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.server_dependencies.database_collections import Collections
from app.user_recommendations.types import UserRecommendations


class IUserRecommendationsRepository(ABC):
    """Interface for storing and retrieving user recommendations."""

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


class UserRecommendationsRepository(IUserRecommendationsRepository):
    """MongoDB implementation for the user recommendations collection."""

    def __init__(self, *, db: AsyncIOMotorDatabase):
        self.db = db
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collection = db.get_collection(Collections.USER_RECOMMENDATIONS)

    async def upsert(self, user_id: str, data: UserRecommendations) -> None:
        if data.user_id != user_id:
            raise ValueError(f"user_id mismatch: {data.user_id} != {user_id}")
        doc = data.model_dump()
        await self.collection.update_one(
            {"user_id": user_id},
            {"$set": doc},
            upsert=True,
        )
        self.logger.info("Upserted user_recommendations for user_id=%s", user_id)

    async def get_by_user_id(self, user_id: str) -> Optional[UserRecommendations]:
        doc = await self.collection.find_one({"user_id": user_id})
        if doc is None:
            return None
        if "_id" in doc:
            del doc["_id"]
        return UserRecommendations(**doc)
