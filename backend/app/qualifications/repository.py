import logging
from abc import ABC, abstractmethod

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.qualifications.types import QualificationEntity
from app.server_dependencies.database_collections import Collections


class IQualificationRepository(ABC):
    @abstractmethod
    async def save_qualifications(self, session_id: int, qualifications: list[QualificationEntity]) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def get_qualifications(self, session_id: int) -> list[QualificationEntity]:
        raise NotImplementedError()

    @abstractmethod
    async def update_qualification(self, session_id: int, qualification_uuid: str, updates: dict) -> bool:
        raise NotImplementedError()

    @abstractmethod
    async def delete_qualification(self, session_id: int, qualification_uuid: str) -> bool:
        raise NotImplementedError()


class QualificationRepository(IQualificationRepository):
    def __init__(self, db: AsyncIOMotorDatabase):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._collection = db.get_collection(Collections.QUALIFICATIONS)

    async def save_qualifications(self, session_id: int, qualifications: list[QualificationEntity]) -> None:
        if not qualifications:
            return
        docs = [{"session_id": session_id, **q.model_dump()} for q in qualifications]
        await self._collection.insert_many(docs)
        self._logger.info("Saved %d qualifications {session_id=%s}", len(docs), session_id)

    async def get_qualifications(self, session_id: int) -> list[QualificationEntity]:
        cursor = self._collection.find({"session_id": session_id})
        docs = await cursor.to_list(length=None)
        return [QualificationEntity(**{k: v for k, v in doc.items() if k != "_id" and k != "session_id"}) for doc in docs]

    async def update_qualification(self, session_id: int, qualification_uuid: str, updates: dict) -> bool:
        # Prevent overwriting identity fields
        updates.pop("uuid", None)
        updates.pop("session_id", None)
        res = await self._collection.update_one(
            {"session_id": session_id, "uuid": qualification_uuid},
            {"$set": updates},
        )
        return res.modified_count > 0

    async def delete_qualification(self, session_id: int, qualification_uuid: str) -> bool:
        res = await self._collection.delete_one({"session_id": session_id, "uuid": qualification_uuid})
        return res.deleted_count > 0
