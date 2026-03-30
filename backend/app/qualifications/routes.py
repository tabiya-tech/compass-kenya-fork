import logging
from http import HTTPStatus

from fastapi import APIRouter, Depends, HTTPException, Path
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.constants.errors import HTTPErrorResponse
from app.qualifications.repository import QualificationRepository, IQualificationRepository
from app.qualifications.types import QualificationEntity
from app.server_dependencies.db_dependencies import CompassDBProvider
from app.users.auth import Authentication, UserInfo

logger = logging.getLogger(__name__)


def _get_qualification_repository(
    db: AsyncIOMotorDatabase = Depends(CompassDBProvider.get_application_db),
) -> IQualificationRepository:
    return QualificationRepository(db)


def add_qualification_routes(conversations_router: APIRouter, auth: Authentication) -> None:
    router = APIRouter(prefix="/qualifications", tags=["qualifications"])

    @router.get(
        path="",
        status_code=HTTPStatus.OK,
        response_model=list[QualificationEntity],
        responses={HTTPStatus.FORBIDDEN: {"model": HTTPErrorResponse}},
        description="Get all qualifications for a conversation session",
    )
    async def get_qualifications(
        session_id: int = Path(description="the conversation session ID"),
        user_info: UserInfo = Depends(auth.get_user_info()),
        repository: IQualificationRepository = Depends(_get_qualification_repository),
    ) -> list[QualificationEntity]:
        try:
            return await repository.get_qualifications(session_id)
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to retrieve qualifications")

    @router.post(
        path="",
        status_code=HTTPStatus.CREATED,
        response_model=QualificationEntity,
        responses={HTTPStatus.FORBIDDEN: {"model": HTTPErrorResponse}},
        description="Add a qualification to a conversation session",
    )
    async def add_qualification(
        qualification: QualificationEntity,
        session_id: int = Path(description="the conversation session ID"),
        user_info: UserInfo = Depends(auth.get_user_info()),
        repository: IQualificationRepository = Depends(_get_qualification_repository),
    ) -> QualificationEntity:
        try:
            await repository.save_qualifications(session_id, [qualification])
            return qualification
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to save qualification")

    @router.patch(
        path="/{qualification_uuid}",
        status_code=HTTPStatus.OK,
        response_model=dict,
        responses={
            HTTPStatus.FORBIDDEN: {"model": HTTPErrorResponse},
            HTTPStatus.NOT_FOUND: {"model": HTTPErrorResponse},
        },
        description="Update a qualification",
    )
    async def update_qualification(
        updates: dict,
        session_id: int = Path(description="the conversation session ID"),
        qualification_uuid: str = Path(description="the qualification UUID"),
        user_info: UserInfo = Depends(auth.get_user_info()),
        repository: IQualificationRepository = Depends(_get_qualification_repository),
    ) -> dict:
        try:
            updated = await repository.update_qualification(session_id, qualification_uuid, updates)
            if not updated:
                raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Qualification not found")
            return {"updated": True}
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to update qualification")

    @router.delete(
        path="/{qualification_uuid}",
        status_code=HTTPStatus.OK,
        response_model=dict,
        responses={
            HTTPStatus.FORBIDDEN: {"model": HTTPErrorResponse},
            HTTPStatus.NOT_FOUND: {"model": HTTPErrorResponse},
        },
        description="Delete a qualification",
    )
    async def delete_qualification(
        session_id: int = Path(description="the conversation session ID"),
        qualification_uuid: str = Path(description="the qualification UUID"),
        user_info: UserInfo = Depends(auth.get_user_info()),
        repository: IQualificationRepository = Depends(_get_qualification_repository),
    ) -> dict:
        try:
            deleted = await repository.delete_qualification(session_id, qualification_uuid)
            if not deleted:
                raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Qualification not found")
            return {"deleted": True}
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to delete qualification")

    conversations_router.include_router(router)
