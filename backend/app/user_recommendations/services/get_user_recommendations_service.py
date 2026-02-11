import asyncio

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.server_dependencies.db_dependencies import CompassDBProvider
from app.user_recommendations.repository.repository import UserRecommendationsRepository
from app.user_recommendations.services.service import IUserRecommendationsService, UserRecommendationsService

_user_recommendations_service_lock = asyncio.Lock()
_user_recommendations_service_singleton: IUserRecommendationsService | None = None


async def get_user_recommendations_service(
    application_db: AsyncIOMotorDatabase = Depends(CompassDBProvider.get_application_db),
) -> IUserRecommendationsService:
    global _user_recommendations_service_singleton
    if _user_recommendations_service_singleton is None:
        async with _user_recommendations_service_lock:
            if _user_recommendations_service_singleton is None:
                _user_recommendations_service_singleton = UserRecommendationsService(
                    repository=UserRecommendationsRepository(db=application_db),
                )
    return _user_recommendations_service_singleton
