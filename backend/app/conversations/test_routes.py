from datetime import datetime
from http import HTTPStatus
from typing import Generator
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock

from app.conversations.reactions.routes import get_user_preferences_repository
from app.conversations.types import ConversationResponse, ConversationMessage, ConversationInput, \
    ConversationMessageSender, ConversationPhaseResponse, CurrentConversationPhaseResponse
from app.conversations.constants import MAX_MESSAGE_LENGTH
from app.conversations.streaming import format_sse_event
from app.conversations.routes import get_conversation_service, add_conversation_routes
from app.conversations.service import IConversationService, ConversationAlreadyConcludedError
from app.users.auth import UserInfo

from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.users.repositories import IUserPreferenceRepository
from app.users.sensitive_personal_data.types import SensitivePersonalDataRequirement
from app.users.types import UserPreferencesRepositoryUpdateRequest, UserPreferences
from common_libs.test_utilities.mock_auth import MockAuth, UnauthenticatedMockAuth

TestClientWithMocks = tuple[TestClient, IConversationService, IUserPreferenceRepository, UserInfo | None]


async def _stream_events(*events: str):
    for event in events:
        yield event


def get_mock_user_preferences(session_id: int):
    return UserPreferences(
        language="en",
        invitation_code="foo",
        accepted_tc=datetime.now(),
        sessions=[session_id],  # mock a user that owns the given session
        sensitive_personal_data_requirement=SensitivePersonalDataRequirement.NOT_REQUIRED
    )


def _create_test_client_with_mocks(auth) -> TestClientWithMocks:
    """
    Factory function to create a test client with mocked dependencies
    """

    # mock the conversation service
    class MockConversationService(IConversationService):
        async def send(self, user_id: str, session_id: int, user_input: str, clear_memory: bool, filter_pii: bool,
                       city: str | None = None, province: str | None = None):
            raise NotImplementedError

        async def stream_send(self, user_id: str, session_id: int, user_input: str, clear_memory: bool, filter_pii: bool):
            raise NotImplementedError

        async def get_history_by_session_id(self, user_id: str, session_id: int):
            raise NotImplementedError

    _instance_conversation_service = MockConversationService()

    def _mocked_conversation_service() -> IConversationService:
        return _instance_conversation_service

    class MockUserPreferencesRepository(IUserPreferenceRepository):
        async def get_user_preference_by_user_id(self, user_id):
            raise NotImplementedError

        async def insert_user_preference(self, user_id: str, user_preference: UserPreferences):
            raise NotImplementedError

        async def update_user_preference(self, user_id: str, update: UserPreferencesRepositoryUpdateRequest):
            raise NotImplementedError

        async def get_experiments_by_user_id(self, user_id: str) -> dict[str, str]:
            raise NotImplementedError

        async def get_experiments_by_user_ids(self, user_ids: list[str]) -> dict[str, dict[str, str]]:
            raise NotImplementedError()

        async def set_experiment_by_user_id(self, user_id: str, experiment_id: str, experiment_class: str) -> None:
            raise NotImplementedError()

    _instance_user_preferences_repository = MockUserPreferencesRepository()

    def get_mock_user_preferences_repository():
        return _instance_user_preferences_repository

    # Set up the FastAPI app with the mocked dependencies
    app = FastAPI()

    # Set up the app dependency overrides
    app.dependency_overrides[get_conversation_service] = _mocked_conversation_service
    app.dependency_overrides[get_user_preferences_repository] = get_mock_user_preferences_repository

    # Add the conversation routes to the app
    add_conversation_routes(app, authentication=auth)

    return TestClient(app), _instance_conversation_service, _instance_user_preferences_repository, auth.mocked_user


@pytest.fixture(scope='function')
def authenticated_client_with_mocks() -> Generator[TestClientWithMocks, None, None]:
    """
    Returns a test client with authenticated mock auth
    """
    app = FastAPI()
    _instance_auth = MockAuth()

    client, service, preferences, user = _create_test_client_with_mocks(_instance_auth)
    yield client, service, preferences, user
    app.dependency_overrides = {}


@pytest.fixture(scope='function')
def unauthenticated_client_with_mocks() -> Generator[TestClientWithMocks, None, None]:
    """
    Returns a test client with unauthenticated mock auth
    """
    app = FastAPI()
    _instance_auth = UnauthenticatedMockAuth()

    client, service, preferences, user = _create_test_client_with_mocks(_instance_auth)
    yield client, service, preferences, user
    app.dependency_overrides = {}


class TestConversationsRoutes:
    # ----- send message tests -----

    @pytest.mark.asyncio
    async def test_send_successful(self, authenticated_client_with_mocks: TestClientWithMocks,
                                   setup_application_config,
                                   mocker: pytest_mock.MockerFixture):
        client, mocked_service, mocked_preferences_repository, mocked_user = authenticated_client_with_mocks
        # GIVEN a payload to send a message
        given_user_message = ConversationInput(
            user_input="foo"
        )

        # AND the user has a valid session
        given_session_id = 123

        # AND a ConversationService that will return a valid SSE stream
        expected_turn_completed = {
            "conversation_completed": False,
            "conversation_conducted_at": datetime.now().isoformat(),
            "experiences_explored": 0,
            "current_phase": ConversationPhaseResponse(
                percentage=0,
                phase=CurrentConversationPhaseResponse.INTRO
            ).model_dump(mode="json"),
        }
        expected_message = ConversationMessage(
            message_id="bar_id",
            message="Hello, I'm compass",
            sender=ConversationMessageSender.COMPASS,
            sent_at=datetime.now().isoformat()
        )
        expected_stream = "".join([
            format_sse_event("message_started", {
                "message_id": expected_message.message_id,
                "sender": "COMPASS",
                "message_type": "TEXT",
                "metadata": None,
            }),
            format_sse_event("message_completed", expected_message.model_dump(mode="json")),
            format_sse_event("turn_completed", expected_turn_completed),
        ])

        # AND mock the repository and service responses
        mocked_preferences_repository.get_user_preference_by_user_id = AsyncMock(
            return_value=get_mock_user_preferences(given_session_id))
        preferences_spy = mocker.spy(mocked_preferences_repository, "get_user_preference_by_user_id")

        mocked_service.stream_send = Mock(return_value=_stream_events(expected_stream))
        service_spy = mocker.spy(mocked_service, "stream_send")

        # WHEN a POST request where the session_id is in the Path
        response = client.post(
            f"/conversations/{given_session_id}/messages",
            json=given_user_message.model_dump(),
        )

        # THEN the response is CREATED
        assert response.status_code == HTTPStatus.CREATED

        # AND the response is an SSE stream with the expected payload
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.text == expected_stream

        # AND the user preferences repository was called with the correct user_id
        preferences_spy.assert_called_once_with(mocked_user.user_id)

        # AND the conversation service was called with the correct arguments
        service_spy.assert_called_once_with(
            mocked_user.user_id,
            given_session_id,
            given_user_message.user_input,
            False,  # clear_memory
            False,  # filter_pii
            city=None,
            province=None
        )

    @pytest.mark.asyncio
    async def test_send_message_unauthorized(self, unauthenticated_client_with_mocks: TestClientWithMocks):
        client, _, _, _ = unauthenticated_client_with_mocks
        # GIVEN a payload to send a message
        given_user_message = ConversationInput(
            user_input="foo"
        )

        # AND a session id
        given_session_id = 123

        # WHEN a POST request is made without authentication
        response = client.post(
            f"/conversations/{given_session_id}/messages",
            json=given_user_message.model_dump(),
        )

        # THEN the response is UNAUTHORIZED
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_send_forbidden(self, authenticated_client_with_mocks: TestClientWithMocks, setup_application_config):
        client, mocked_conversation_service, mocked_preferences_repository, mocked_user = authenticated_client_with_mocks
        # GIVEN a payload to send a message
        given_user_message = ConversationInput(
            user_input="foo"
        )

        # AND the user has a valid session
        given_session_id = 123

        # AND the conversation service will successfully send a message
        mocked_conversation_service.send = AsyncMock()

        # AND the user doesnt have the given session_id in their sessions array
        mock_user_preferences = get_mock_user_preferences(given_session_id)
        mock_user_preferences.sessions = [999]  # sessions doesnt include the given session id
        mocked_preferences_repository.get_user_preference_by_user_id = AsyncMock(
            return_value=mock_user_preferences)

        # WHEN a POST request where the session_id is in the Path
        response = client.post(
            f"/conversations/{given_session_id}/messages",
            json=given_user_message.model_dump(),
        )

        # THEN the response is FORBIDDEN
        assert response.status_code == HTTPStatus.FORBIDDEN

        # AND the user preferences repository was called with the correct user_id
        mocked_preferences_repository.get_user_preference_by_user_id.assert_called_once_with(mocked_user.user_id)

    @pytest.mark.asyncio
    async def test_send_message_too_large(self, authenticated_client_with_mocks: TestClientWithMocks,
                                          setup_application_config):
        client, _, mocked_preferences_repository, _ = authenticated_client_with_mocks
        # GIVEN a payload with a message that is too long
        given_user_message = ConversationInput(
            user_input="a" * (MAX_MESSAGE_LENGTH + 1)  # Create a string longer than MAX_MESSAGE_LENGTH characters
        )

        # AND the user has a valid session
        given_session_id = 123

        # WHEN a POST request where the session_id is in the Path
        mocked_preferences_repository.get_user_preference_by_user_id = AsyncMock(
            return_value=get_mock_user_preferences(given_session_id))
        response = client.post(
            f"/conversations/{given_session_id}/messages",
            json=given_user_message.model_dump(),
        )

        # THEN the response is REQUEST_ENTITY_TOO_LARGE
        assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE

    @pytest.mark.asyncio
    async def test_send_conversation_already_concluded(self, authenticated_client_with_mocks: TestClientWithMocks,
                                                       setup_application_config,
                                                       mocker: pytest_mock.MockerFixture):
        client, mocked_service, mocked_preferences_repository, mocked_user = authenticated_client_with_mocks
        # GIVEN a payload to send a message
        given_user_message = ConversationInput(
            user_input="foo"
        )

        # AND the user has a valid session
        given_session_id = 123

        # AND a ConversationService that will emit an SSE error event
        mocked_preferences_repository.get_user_preference_by_user_id = AsyncMock(
            return_value=get_mock_user_preferences(given_session_id))
        expected_stream = format_sse_event("error", {
            "code": "conversation_already_concluded",
            "message": str(ConversationAlreadyConcludedError(given_session_id)),
            "recoverable": False,
        })
        mocked_service.stream_send = Mock(return_value=_stream_events(expected_stream))
        stream_spy = mocker.spy(mocked_service, "stream_send")

        # WHEN a POST request where the session_id is in the Path
        response = client.post(
            f"/conversations/{given_session_id}/messages",
            json=given_user_message.model_dump(),
        )

        # THEN the response is CREATED and contains an SSE error payload
        assert response.status_code == HTTPStatus.CREATED
        assert response.text == expected_stream

        # AND the conversation service was called with the correct arguments
        stream_spy.assert_called_once_with(
            mocked_user.user_id,
            given_session_id,
            given_user_message.user_input,
            False,  # clear_memory
            False,  # filter_pii
            city=None,
            province=None
        )

    @pytest.mark.asyncio
    async def test_send_service_internal_server_error(self, authenticated_client_with_mocks: TestClientWithMocks,
                                                      setup_application_config,
                                                      mocker: pytest_mock.MockerFixture):
        client, mocked_service, mocked_preferences_repository, mocked_user = authenticated_client_with_mocks
        # GIVEN a payload to send a message
        given_user_message = ConversationInput(
            user_input="foo"
        )

        # AND the user has a valid session
        given_session_id = 123

        # AND a ConversationService that will emit an unexpected SSE error event
        mocked_preferences_repository.get_user_preference_by_user_id = AsyncMock(
            return_value=get_mock_user_preferences(given_session_id))
        expected_stream = format_sse_event("error", {
            "code": "unexpected_failure",
            "message": "Oops! something went wrong",
            "recoverable": False,
        })
        mocked_service.stream_send = Mock(return_value=_stream_events(expected_stream))
        stream_spy = mocker.spy(mocked_service, "stream_send")

        # WHEN a POST request where the session_id is in the Path
        response = client.post(
            f"/conversations/{given_session_id}/messages",
            json=given_user_message.model_dump(),
        )

        # THEN the response is CREATED and contains an SSE error payload
        assert response.status_code == HTTPStatus.CREATED
        assert response.text == expected_stream

        # AND the conversation service was called with the correct arguments
        stream_spy.assert_called_once_with(
            mocked_user.user_id,
            given_session_id,
            given_user_message.user_input,
            False,  # clear_memory
            False,  # filter_pii
            city=None,
            province=None
        )

    @pytest.mark.asyncio
    async def test_send_user_preferences_internal_server_error(self,
                                                               authenticated_client_with_mocks: TestClientWithMocks,
                                                               setup_application_config,
                                                               mocker: pytest_mock.MockerFixture):
        client, _, mocked_preferences_repository, mocked_user = authenticated_client_with_mocks
        # GIVEN a payload to send a message
        given_user_message = ConversationInput(
            user_input="foo"
        )

        # AND the user has a valid session
        given_session_id = 123

        # AND a UserPreferencesRepository that will raise an unexpected error
        get_user_preferences_spy = mocker.spy(mocked_preferences_repository, "get_user_preference_by_user_id")
        get_user_preferences_spy.side_effect = Exception("Unexpected error")

        # WHEN a POST request where the session_id is in the Path
        response = client.post(
            f"/conversations/{given_session_id}/messages",
            json=given_user_message.model_dump(),
        )

        # THEN the response is INTERNAL_SERVER_ERROR
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR

        # AND the user preferences repository was called with the correct user_id
        get_user_preferences_spy.assert_called_once_with(mocked_user.user_id)

    @pytest.mark.asyncio
    async def test_send_invalid_payload(self, authenticated_client_with_mocks: TestClientWithMocks):
        client, _, _, _ = authenticated_client_with_mocks
        # GIVEN an invalid payload as a dictionary
        given_invalid_payload = {"foo": "bar"}  # expected object with user_input

        # AND the request has a valid session id
        given_session_id = 123

        # WHEN a GET request where `user_id` in the path matches the authenticated user's `user_id`
        response = client.post(
            f"/conversations/{given_session_id}/messages",
            json=given_invalid_payload,
        )

        # THEN the response is UNPROCESSABLE_ENTITY
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    # ----- get conversation history tests -----

    @pytest.mark.asyncio
    async def test_get_history_successful(self, authenticated_client_with_mocks: TestClientWithMocks,
                                          mocker: pytest_mock.MockerFixture, setup_application_config):
        client, mocked_service, mocked_preferences_repository, mocked_user = authenticated_client_with_mocks
        # GIVEN a valid session id
        given_session_id = 123

        # AND mock the repository and service responses
        mocked_preferences_repository.get_user_preference_by_user_id = AsyncMock(
            return_value=get_mock_user_preferences(given_session_id))
        preferences_spy = mocker.spy(mocked_preferences_repository, "get_user_preference_by_user_id")

        # AND a ConversationService that will return a valid conversation response
        expected_response = ConversationResponse(
            messages=[
                ConversationMessage(
                    message_id="test_id",
                    message="test message",
                    sender=ConversationMessageSender.USER,
                    sent_at=datetime.now().isoformat()
                )
            ],
            conversation_completed=False,
            conversation_conducted_at=datetime.now().isoformat(),
            experiences_explored=0,
            current_phase=ConversationPhaseResponse(
                percentage=0,
                phase=CurrentConversationPhaseResponse.INTRO
            )
        )
        mocked_service.get_history_by_session_id = AsyncMock(return_value=expected_response)
        service_spy = mocker.spy(mocked_service, "get_history_by_session_id")

        # WHEN a GET request where the session_id is in the Path
        response = client.get(f"/conversations/{given_session_id}/messages")

        # THEN the response is OK
        assert response.status_code == HTTPStatus.OK

        # AND the response matches the expected response
        assert response.json() == expected_response.model_dump()

        # AND the user preferences repository was called with the correct user_id
        preferences_spy.assert_called_once_with(mocked_user.user_id)

        # AND the conversation service was called with the correct arguments
        service_spy.assert_called_once_with(
            mocked_user.user_id,
            given_session_id
        )

    @pytest.mark.asyncio
    async def test_get_history_unauthorized(self, unauthenticated_client_with_mocks: TestClientWithMocks):
        client, _, _, _ = unauthenticated_client_with_mocks
        # GIVEN a session id
        given_session_id = 123

        # WHEN a GET request is made without authentication
        response = client.get(f"/conversations/{given_session_id}/messages")

        # THEN the response is UNAUTHORIZED
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_get_history_forbidden(self, authenticated_client_with_mocks: TestClientWithMocks,
                                         setup_application_config,
                                         mocker: pytest_mock.MockerFixture):
        client, mocked_conversation_service, mocked_preferences_repository, mocked_user = authenticated_client_with_mocks
        # GIVEN a valid session id
        given_session_id = 123

        # AND the conversation get history will successfully get the history
        mocked_conversation_service.get_history_by_session_id = AsyncMock()

        # AND the user doesnt have the given session_id in their sessions array
        mock_user_preferences = get_mock_user_preferences(given_session_id)
        mock_user_preferences.sessions = [999]  # sessions doesnt include the given session id
        mocked_preferences_repository.get_user_preference_by_user_id = AsyncMock(
            return_value=mock_user_preferences)

        # WHEN a GET request where the session_id is in the Path
        response = client.get(f"/conversations/{given_session_id}/messages")

        # THEN the response is FORBIDDEN
        assert response.status_code == HTTPStatus.FORBIDDEN

        # AND the user preferences repository was called with the correct user_id
        mocked_preferences_repository.get_user_preference_by_user_id.assert_called_once_with(mocked_user.user_id)

    @pytest.mark.asyncio
    async def test_get_history_service_internal_server_error(self, authenticated_client_with_mocks: TestClientWithMocks,
                                                             setup_application_config,
                                                             mocker: pytest_mock.MockerFixture):
        client, mocked_service, mocked_preferences_repository, mocked_user = authenticated_client_with_mocks
        # GIVEN a valid session id
        given_session_id = 123

        # AND a ConversationService that will raise an unexpected error
        mocked_preferences_repository.get_user_preference_by_user_id = AsyncMock(
            return_value=get_mock_user_preferences(given_session_id))
        get_history_spy = mocker.spy(mocked_service, "get_history_by_session_id")
        get_history_spy.side_effect = Exception("Unexpected error")

        # WHEN a GET request where the session_id is in the Path
        response = client.get(f"/conversations/{given_session_id}/messages")

        # THEN the response is INTERNAL_SERVER_ERROR
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR

        # AND the conversation service was called with the correct arguments
        get_history_spy.assert_called_once_with(
            mocked_user.user_id,
            given_session_id
        )

    @pytest.mark.asyncio
    async def test_get_history_user_preferences_internal_server_error(self,
                                                                      authenticated_client_with_mocks: TestClientWithMocks,
                                                                      setup_application_config,
                                                                      mocker: pytest_mock.MockerFixture):
        client, _, mocked_preferences_repository, mocked_user = authenticated_client_with_mocks
        # GIVEN a valid session id
        given_session_id = 123

        # AND a UserPreferencesRepository that will raise an unexpected error
        get_user_preferences_spy = mocker.spy(mocked_preferences_repository, "get_user_preference_by_user_id")
        get_user_preferences_spy.side_effect = Exception("Unexpected error")

        # WHEN a GET request where the session_id is in the Path
        response = client.get(f"/conversations/{given_session_id}/messages")

        # THEN the response is INTERNAL_SERVER_ERROR
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR

        # AND the user preferences repository was called with the correct user_id
        get_user_preferences_spy.assert_called_once_with(mocked_user.user_id)
