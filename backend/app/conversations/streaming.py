import asyncio
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel

from app.agent.agent_types import AgentOutput
from app.app_config import get_application_config
from app.conversations.types import ConversationMessage, ConversationMessageSender, ConversationResponse


class ConversationStreamEventType(str, Enum):
    TURN_STARTED = "turn_started"
    STATUS_UPDATED = "status_updated"
    PHASE_UPDATED = "phase_updated"
    MESSAGE_STARTED = "message_started"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_COMPLETED = "message_completed"
    TURN_COMPLETED = "turn_completed"
    ERROR = "error"


class TurnStartedEvent(BaseModel):
    session_id: int
    user_message_id: str
    current_phase: dict[str, Any] | None = None


class MessageStartedEvent(BaseModel):
    message_id: str
    sender: str
    message_type: str = "TEXT"
    metadata: dict[str, Any] | None = None


class MessageDeltaEvent(BaseModel):
    message_id: str
    delta: str


class StatusUpdatedEvent(BaseModel):
    label: str
    status: str = "running"
    agent_type: str | None = None
    detail: str | None = None
    current_phase: dict[str, Any] | None = None


class PhaseUpdatedEvent(BaseModel):
    current_phase: dict[str, Any]
    agent_type: str | None = None
    detail: str | None = None


class TurnCompletedEvent(BaseModel):
    conversation_completed: bool
    conversation_conducted_at: str | None = None
    experiences_explored: int
    current_phase: dict[str, Any]


class ErrorEvent(BaseModel):
    code: str
    message: str
    recoverable: bool = False


def format_sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n"


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _get_message_type_and_metadata(agent_output: AgentOutput) -> tuple[str, dict[str, Any] | None]:
    output_metadata = agent_output.metadata if hasattr(agent_output, "metadata") else None
    message_type = "TEXT"
    if output_metadata and output_metadata.get("task_id") is not None and "alternatives" in output_metadata:
        message_type = "BWS_TASK"
    return message_type, output_metadata


def conversation_message_from_agent_output(agent_output: AgentOutput) -> ConversationMessage:
    message_type, metadata = _get_message_type_and_metadata(agent_output)
    return ConversationMessage(
        message_id=agent_output.message_id,
        message=agent_output.message_for_user,
        sent_at=agent_output.sent_at,
        sender=ConversationMessageSender.COMPASS,
        reaction=None,
        message_type=message_type,
        metadata=metadata,
    )


class ConversationStreamingSink:
    def __init__(self):
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._started_message_ids: set[str] = set()

    async def _emit_model(self, event_type: ConversationStreamEventType, model: BaseModel | ConversationMessage) -> None:
        await self._queue.put(format_sse_event(event_type.value, model.model_dump(mode="json")))

    async def emit_turn_started(self, *, session_id: int, user_message_id: str, current_phase: dict[str, Any] | None) -> None:
        await self._emit_model(
            ConversationStreamEventType.TURN_STARTED,
            TurnStartedEvent(
                session_id=session_id,
                user_message_id=user_message_id,
                current_phase=current_phase,
            ),
        )

    async def emit_status_update(
        self,
        *,
        label: str,
        status: str = "running",
        agent_type: str | None = None,
        detail: Any | None = None,
        current_phase: dict[str, Any] | None = None,
    ) -> None:
        await self._emit_model(
            ConversationStreamEventType.STATUS_UPDATED,
            StatusUpdatedEvent(
                label=label,
                status=status,
                agent_type=agent_type,
                detail=_normalize_optional_text(detail),
                current_phase=current_phase,
            ),
        )

    async def emit_phase_update(
        self,
        *,
        current_phase: dict[str, Any],
        agent_type: str | None = None,
        detail: Any | None = None,
    ) -> None:
        await self._emit_model(
            ConversationStreamEventType.PHASE_UPDATED,
            PhaseUpdatedEvent(
                current_phase=current_phase,
                agent_type=agent_type,
                detail=_normalize_optional_text(detail),
            ),
        )

    async def start_message(
        self,
        *,
        message_id: str,
        sender: str = "COMPASS",
        message_type: str = "TEXT",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if message_id in self._started_message_ids:
            return
        self._started_message_ids.add(message_id)
        await self._emit_model(
            ConversationStreamEventType.MESSAGE_STARTED,
            MessageStartedEvent(
                message_id=message_id,
                sender=sender,
                message_type=message_type,
                metadata=metadata,
            ),
        )

    async def append_text(self, *, message_id: str, delta: str) -> None:
        if not delta:
            return
        await self.start_message(message_id=message_id)
        await self._emit_model(
            ConversationStreamEventType.MESSAGE_DELTA,
            MessageDeltaEvent(message_id=message_id, delta=delta),
        )

    async def complete_message(self, message: ConversationMessage) -> None:
        await self.start_message(
            message_id=message.message_id,
            sender=message.sender.name,
            message_type=message.message_type,
            metadata=message.metadata,
        )
        await self._emit_model(ConversationStreamEventType.MESSAGE_COMPLETED, message)

    async def emit_agent_output(self, agent_output: AgentOutput) -> None:
        message = conversation_message_from_agent_output(agent_output)
        already_streamed = message.message_id in self._started_message_ids
        if already_streamed:
            return
        if message.message_type != "TEXT" or not message.message:
            await self.complete_message(message)
            return
        await self._stream_text_then_complete(message)

    async def _stream_text_then_complete(self, message: ConversationMessage) -> None:
        app_config = get_application_config()
        chunk_size = app_config.stream_chunk_size
        chunk_mode = app_config.stream_chunk_mode
        delay_sec = app_config.stream_delta_delay_ms / 1000.0
        text = message.message
        await self.start_message(
            message_id=message.message_id,
            sender=message.sender.name,
            message_type=message.message_type,
            metadata=message.metadata,
        )
        if chunk_mode == "words":
            words = text.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                await self._emit_model(
                    ConversationStreamEventType.MESSAGE_DELTA,
                    MessageDeltaEvent(message_id=message.message_id, delta=chunk),
                )
                await asyncio.sleep(delay_sec)
        else:
            offset = 0
            while offset < len(text):
                end = min(offset + chunk_size, len(text))
                if end < len(text):
                    space = text.rfind(" ", offset, end + 1)
                    if space > offset:
                        end = space + 1
                chunk = text[offset:end]
                await self._emit_model(
                    ConversationStreamEventType.MESSAGE_DELTA,
                    MessageDeltaEvent(message_id=message.message_id, delta=chunk),
                )
                offset = end
                await asyncio.sleep(delay_sec)
        await self._emit_model(ConversationStreamEventType.MESSAGE_COMPLETED, message)

    async def emit_turn_completed(self, response: ConversationResponse) -> None:
        serialized_response = response.model_dump(mode="json")
        await self._emit_model(
            ConversationStreamEventType.TURN_COMPLETED,
            TurnCompletedEvent(
                conversation_completed=serialized_response["conversation_completed"],
                conversation_conducted_at=serialized_response["conversation_conducted_at"],
                experiences_explored=serialized_response["experiences_explored"],
                current_phase=serialized_response["current_phase"],
            ),
        )

    async def emit_error(self, *, code: str, message: str, recoverable: bool = False) -> None:
        await self._emit_model(
            ConversationStreamEventType.ERROR,
            ErrorEvent(code=code, message=message, recoverable=recoverable),
        )

    async def close(self) -> None:
        await self._queue.put(None)

    async def iter_sse(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item
