import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.users.cv.repository import UserCVRepository
from app.users.cv.types import UploadProcessState


@pytest.fixture
def mock_collection():
    collection = AsyncMock()
    return collection


@pytest.fixture
def repo(mock_collection):
    db = MagicMock()
    db.get_collection.return_value = mock_collection
    return UserCVRepository(db)


@pytest.mark.asyncio
async def test_store_structured_extraction_updates_document(repo, mock_collection):
    mock_collection.update_one.return_value = MagicMock(modified_count=1)
    extraction = {
        "experiences": [{"experience_title": "Engineer", "responsibilities": ["Built APIs"]}],
        "qualifications": [{"name": "BSc Computer Science"}],
    }
    result = await repo.store_structured_extraction("user1", "upload1", extraction=extraction)
    assert result is True
    mock_collection.update_one.assert_called_once()
    call_args = mock_collection.update_one.call_args
    assert call_args[0][0] == {"user_id": "user1", "upload_id": "upload1"}
    assert call_args[0][1]["$set"]["structured_extraction"] == extraction


@pytest.mark.asyncio
async def test_store_structured_extraction_returns_false_when_not_found(repo, mock_collection):
    mock_collection.update_one.return_value = MagicMock(modified_count=0)
    result = await repo.store_structured_extraction("user1", "upload1", extraction={})
    assert result is False


@pytest.mark.asyncio
async def test_get_latest_structured_extraction_returns_data(repo, mock_collection):
    cursor = AsyncMock()
    cursor.to_list.return_value = [{
        "structured_extraction": {
            "experiences": [{"experience_title": "Dev", "responsibilities": ["Coded"]}],
            "qualifications": [],
        }
    }]
    mock_collection.find.return_value = cursor
    result = await repo.get_latest_structured_extraction("user1")
    assert result is not None
    assert len(result["experiences"]) == 1
    mock_collection.find.assert_called_once()
    query = mock_collection.find.call_args[0][0]
    assert query["user_id"] == "user1"
    assert query["upload_process_state"] == UploadProcessState.COMPLETED
    assert query["structured_extraction"] == {"$ne": None}


@pytest.mark.asyncio
async def test_get_latest_structured_extraction_returns_none_when_no_uploads(repo, mock_collection):
    cursor = AsyncMock()
    cursor.to_list.return_value = []
    mock_collection.find.return_value = cursor
    result = await repo.get_latest_structured_extraction("user1")
    assert result is None
