from typing import Awaitable

import pytest
import pytest_mock

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.user_recommendations.repository.repository import UserRecommendationsRepository
from app.user_recommendations.types import (
    UserRecommendations,
    OccupationRecommendationDB,
    OpportunityRecommendationDB,
    SkillGapRecommendationDB,
)

from common_libs.test_utilities import get_custom_error, get_random_user_id


@pytest.fixture(scope="function")
async def get_user_recommendations_repository(
        in_memory_application_database: Awaitable[AsyncIOMotorDatabase]) -> UserRecommendationsRepository:
    application_db = await in_memory_application_database
    repository = UserRecommendationsRepository(db=application_db)
    return repository


def _get_user_recommendations(
        *,
        user_id: str,
        occupation_count: int = 0,
        opportunity_count: int = 0,
        skill_gap_count: int = 0,
) -> UserRecommendations:
    return UserRecommendations(
        user_id=user_id,
        occupation_recommendations=[
            OccupationRecommendationDB(
                uuid=f"occ-{idx}",
                rank=idx + 1,
                occupation_label=f"Occupation {idx}",
                final_score=0.8 + idx * 0.1,
            )
            for idx in range(occupation_count)
        ],
        opportunity_recommendations=[
            OpportunityRecommendationDB(
                uuid=f"opp-{idx}",
                rank=idx + 1,
                opportunity_title=f"Opportunity {idx}",
                final_score=0.7 + idx * 0.1,
            )
            for idx in range(opportunity_count)
        ],
        skill_gap_recommendations=[
            SkillGapRecommendationDB(
                skill_id=f"skill-{idx}",
                skill_label=f"Skill {idx}",
                proximity_score=0.5,
                job_unlock_count=2,
            )
            for idx in range(skill_gap_count)
        ],
    )


def _assert_doc_matches(actual_doc: dict, given: UserRecommendations) -> None:
    assert actual_doc["user_id"] == given.user_id
    assert len(actual_doc["occupation_recommendations"]) == len(given.occupation_recommendations)
    assert len(actual_doc["opportunity_recommendations"]) == len(given.opportunity_recommendations)
    assert len(actual_doc["skill_gap_recommendations"]) == len(given.skill_gap_recommendations)
    for index, occupation in enumerate(given.occupation_recommendations):
        assert actual_doc["occupation_recommendations"][index]["uuid"] == occupation.uuid
        assert actual_doc["occupation_recommendations"][index]["rank"] == occupation.rank


class TestUpsert:
    @pytest.mark.asyncio
    async def test_insert_new(self, get_user_recommendations_repository: Awaitable[UserRecommendationsRepository]):
        repository = await get_user_recommendations_repository

        given_user_id = get_random_user_id()
        given_recommendations = _get_user_recommendations(
            user_id=given_user_id,
            occupation_count=2,
            opportunity_count=1,
            skill_gap_count=1,
        )

        await repository.upsert(given_user_id, given_recommendations)

        actual_doc = await repository.collection.find_one({"user_id": given_user_id})
        assert actual_doc is not None
        _assert_doc_matches(actual_doc, given_recommendations)

    @pytest.mark.asyncio
    async def test_update_existing(self, get_user_recommendations_repository: Awaitable[UserRecommendationsRepository]):
        repository = await get_user_recommendations_repository

        given_user_id = get_random_user_id()
        initial_recommendations = _get_user_recommendations(
            user_id=given_user_id,
            occupation_count=1,
            opportunity_count=0,
            skill_gap_count=0,
        )
        await repository.upsert(given_user_id, initial_recommendations)

        updated_recommendations = _get_user_recommendations(
            user_id=given_user_id,
            occupation_count=3,
            opportunity_count=2,
            skill_gap_count=2,
        )
        await repository.upsert(given_user_id, updated_recommendations)

        actual_doc = await repository.collection.find_one({"user_id": given_user_id})
        assert actual_doc is not None
        _assert_doc_matches(actual_doc, updated_recommendations)

    @pytest.mark.asyncio
    async def test_upsert_raises_on_user_id_mismatch(
            self, get_user_recommendations_repository: Awaitable[UserRecommendationsRepository]):
        repository = await get_user_recommendations_repository

        given_recommendations = _get_user_recommendations(user_id="user-a", occupation_count=1)

        with pytest.raises(ValueError) as exc_info:
            await repository.upsert("user-b", given_recommendations)

        assert "user_id mismatch" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_one_throws(
            self,
            get_user_recommendations_repository: Awaitable[UserRecommendationsRepository],
            mocker: pytest_mock.MockerFixture,
    ):
        repository = await get_user_recommendations_repository

        given_user_id = get_random_user_id()
        given_recommendations = _get_user_recommendations(user_id=given_user_id, occupation_count=1)

        error_class, given_error = get_custom_error()
        mocker.patch.object(repository.collection, "update_one", side_effect=given_error)

        with pytest.raises(error_class) as actual_error_info:
            await repository.upsert(given_user_id, given_recommendations)

        assert actual_error_info.value == given_error


class TestGetByUserId:
    @pytest.mark.asyncio
    async def test_find_existing(self, get_user_recommendations_repository: Awaitable[UserRecommendationsRepository]):
        repository = await get_user_recommendations_repository

        given_user_id = get_random_user_id()
        given_recommendations = _get_user_recommendations(
            user_id=given_user_id,
            occupation_count=2,
            opportunity_count=1,
            skill_gap_count=1,
        )
        await repository.upsert(given_user_id, given_recommendations)

        result = await repository.get_by_user_id(given_user_id)

        assert result is not None
        assert result.user_id == given_recommendations.user_id
        assert len(result.occupation_recommendations) == 2
        assert len(result.opportunity_recommendations) == 1
        assert len(result.skill_gap_recommendations) == 1

    @pytest.mark.asyncio
    async def test_not_found_returns_none(
            self, get_user_recommendations_repository: Awaitable[UserRecommendationsRepository]):
        repository = await get_user_recommendations_repository

        given_user_id = get_random_user_id()

        result = await repository.get_by_user_id(given_user_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_find_one_throws(
            self,
            get_user_recommendations_repository: Awaitable[UserRecommendationsRepository],
            mocker: pytest_mock.MockerFixture,
    ):
        repository = await get_user_recommendations_repository

        error_class, given_error = get_custom_error()
        mocker.patch.object(repository.collection, "find_one", side_effect=given_error)

        with pytest.raises(error_class) as actual_error_info:
            await repository.get_by_user_id(get_random_user_id())

        assert actual_error_info.value == given_error
