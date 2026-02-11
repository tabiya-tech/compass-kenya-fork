from unittest.mock import AsyncMock

import pytest

from app.user_recommendations.repository.repository import IUserRecommendationsRepository
from app.user_recommendations.services.service import UserRecommendationsService
from app.user_recommendations.types import OccupationRecommendationDB, UserRecommendations


@pytest.fixture(scope="function")
def _mock_repository() -> IUserRecommendationsRepository:
    class MockUserRecommendationsRepository(IUserRecommendationsRepository):
        async def upsert(self, user_id: str, data: UserRecommendations) -> None:
            raise NotImplementedError()

        async def get_by_user_id(self, user_id: str):
            raise NotImplementedError()

    return MockUserRecommendationsRepository()


def _user_recommendations(user_id: str, recommendation_count: int = 1) -> UserRecommendations:
    occupations = [
        OccupationRecommendationDB(uuid="occ-1", occupation_label="Label 1"),
        OccupationRecommendationDB(uuid="occ-2", occupation_label="Label 2"),
    ][:recommendation_count]
    return UserRecommendations(
        user_id=user_id,
        occupation_recommendations=occupations,
        opportunity_recommendations=[],
        skill_gap_recommendations=[],
    )


class TestUserRecommendationsService:
    @pytest.mark.asyncio
    async def test_upsert_delegates_to_repository(self, _mock_repository):
        # GIVEN a service with a mock repository and valid payload
        _mock_repository.upsert = AsyncMock(return_value=None)
        service = UserRecommendationsService(repository=_mock_repository)
        user_id = "user-1"
        data = _user_recommendations(user_id)

        # WHEN we upsert
        await service.upsert(user_id, data)

        # THEN the repository upsert is called with the same arguments
        _mock_repository.upsert.assert_called_once_with(user_id, data)

    @pytest.mark.asyncio
    async def test_upsert_with_user_id_mismatch_raises(self, _mock_repository):
        # GIVEN a service and payload whose user_id does not match the path user_id
        service = UserRecommendationsService(repository=_mock_repository)
        path_user_id = "user-path"
        data = _user_recommendations("user-other")

        # WHEN we upsert with mismatched user_id
        # THEN ValueError is raised
        with pytest.raises(ValueError) as exc_info:
            await service.upsert(path_user_id, data)

        # AND the message mentions user_id mismatch
        assert "user_id mismatch" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_by_user_id_delegates_to_repository(self, _mock_repository):
        # GIVEN a service and repository that returns a value
        user_id = "user-1"
        expected = _user_recommendations(user_id)
        _mock_repository.get_by_user_id = AsyncMock(return_value=expected)
        service = UserRecommendationsService(repository=_mock_repository)

        # WHEN we get by user_id
        result = await service.get_by_user_id(user_id)

        # THEN the repository result is returned
        assert result == expected
        # AND the repository was called with the user_id
        _mock_repository.get_by_user_id.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_has_recommendations_true_when_non_empty_list(self, _mock_repository):
        # GIVEN a service and repository that returns recommendations with at least one item
        user_id = "user-1"
        _mock_repository.get_by_user_id = AsyncMock(return_value=_user_recommendations(user_id, recommendation_count=1))
        service = UserRecommendationsService(repository=_mock_repository)

        # WHEN we call has_recommendations
        result = await service.has_recommendations(user_id)

        # THEN the result is True
        assert result is True

    @pytest.mark.asyncio
    async def test_has_recommendations_false_when_none(self, _mock_repository):
        # GIVEN a service and repository that returns None
        _mock_repository.get_by_user_id = AsyncMock(return_value=None)
        service = UserRecommendationsService(repository=_mock_repository)
        user_id = "user-missing"

        # WHEN we call has_recommendations
        result = await service.has_recommendations(user_id)

        # THEN the result is False
        assert result is False

    @pytest.mark.asyncio
    async def test_has_recommendations_false_when_empty_list(self, _mock_repository):
        # GIVEN a service and repository that returns a record with empty recommendations
        user_id = "user-1"
        _mock_repository.get_by_user_id = AsyncMock(
            return_value=UserRecommendations(
                user_id=user_id,
                occupation_recommendations=[],
                opportunity_recommendations=[],
                skill_gap_recommendations=[],
            )
        )
        service = UserRecommendationsService(repository=_mock_repository)

        # WHEN we call has_recommendations
        result = await service.has_recommendations(user_id)

        # THEN the result is False
        assert result is False
