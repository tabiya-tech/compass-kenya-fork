import json
from datetime import datetime, timezone

import pytest

from app.agent.agent_types import AgentOutput
from app.conversations.streaming import ConversationStreamingSink, format_sse_event
from app.conversations.types import ConversationPhaseResponse, ConversationResponse, CurrentConversationPhaseResponse


def test_format_sse_event():
    payload = {"message": "hello"}
    assert format_sse_event("message_completed", payload) == (
        'event: message_completed\n'
        'data: {"message": "hello"}\n\n'
    )


@pytest.mark.asyncio
async def test_streaming_sink_emits_message_and_turn_events():
    sink = ConversationStreamingSink()
    agent_output = AgentOutput(
        message_id="msg-1",
        message_for_user="Hello from Compass",
        finished=False,
        agent_response_time_in_sec=0.42,
        llm_stats=[],
    )
    response = ConversationResponse(
        messages=[],
        conversation_completed=False,
        conversation_conducted_at=datetime.now(timezone.utc),
        experiences_explored=1,
        current_phase=ConversationPhaseResponse(
            percentage=25,
            phase=CurrentConversationPhaseResponse.COLLECT_EXPERIENCES,
        ),
    )

    await sink.emit_turn_started(
        session_id=123,
        user_message_id="user-1",
        current_phase=response.current_phase.model_dump(mode="json"),
    )
    await sink.emit_agent_output(agent_output)
    await sink.emit_turn_completed(response)
    await sink.close()

    events = []
    async for chunk in sink.iter_sse():
        lines = [line for line in chunk.split("\n") if line]
        event_name_line, data_line = lines[:2]
        events.append((event_name_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))

    event_names = [event_name for event_name, _ in events]
    assert event_names[0] == "turn_started"
    assert event_names[-1] == "turn_completed"
    assert "message_started" in event_names
    assert "message_completed" in event_names
    # message_delta events appear when streaming text chunks
    assert event_names.count("message_delta") >= 1

    message_completed = next(data for name, data in events if name == "message_completed")
    assert message_completed["message"] == "Hello from Compass"

    turn_completed = next(data for name, data in events if name == "turn_completed")
    assert turn_completed["experiences_explored"] == 1


@pytest.mark.asyncio
async def test_streaming_sink_emits_status_and_phase_updates():
    sink = ConversationStreamingSink()

    await sink.emit_status_update(
        label="routing",
        status="running",
        agent_type="welcome_agent",
        detail="INTRO",
        current_phase={"phase": "INTRO", "percentage": 0, "current": None, "total": None},
    )
    await sink.emit_phase_update(
        current_phase={"phase": "PREFERENCE_ELICITATION", "percentage": 72, "current": 1, "total": 6},
        agent_type="preference_elicitation_agent",
        detail="phase_progressed",
    )
    await sink.close()

    events = []
    async for chunk in sink.iter_sse():
        lines = [line for line in chunk.split("\n") if line]
        event_name_line, data_line = lines[:2]
        events.append((event_name_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))

    assert [event_name for event_name, _ in events] == [
        "status_updated",
        "phase_updated",
    ]
    assert events[0][1]["label"] == "routing"
    assert events[1][1]["current_phase"]["phase"] == "PREFERENCE_ELICITATION"


@pytest.mark.asyncio
async def test_streaming_sink_coerces_non_string_detail_values():
    sink = ConversationStreamingSink()

    await sink.emit_status_update(
        label="running_agent",
        status="started",
        agent_type="welcome_agent",
        detail=0,
    )
    await sink.close()

    events = []
    async for chunk in sink.iter_sse():
        lines = [line for line in chunk.split("\n") if line]
        event_name_line, data_line = lines[:2]
        events.append((event_name_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))

    assert events == [
        (
            "status_updated",
            {
                "label": "running_agent",
                "status": "started",
                "agent_type": "welcome_agent",
                "detail": "0",
                "current_phase": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_streaming_sink_appends_multiple_deltas_once_per_message_start():
    sink = ConversationStreamingSink()

    await sink.append_text(message_id="msg-1", delta="Hello")
    await sink.append_text(message_id="msg-1", delta=", world")
    await sink.append_text(message_id="msg-1", delta="")
    await sink.close()

    events = []
    async for chunk in sink.iter_sse():
        lines = [line for line in chunk.split("\n") if line]
        event_name_line, data_line = lines[:2]
        events.append((event_name_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))

    assert [event_name for event_name, _ in events] == [
        "message_started",
        "message_delta",
        "message_delta",
    ]
    assert events[1][1]["delta"] == "Hello"
    assert events[2][1]["delta"] == ", world"
