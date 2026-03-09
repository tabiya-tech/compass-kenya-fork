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

    assert [event_name for event_name, _ in events] == [
        "turn_started",
        "message_started",
        "message_completed",
        "turn_completed",
    ]
    assert events[2][1]["message"] == "Hello from Compass"
    assert events[3][1]["experiences_explored"] == 1
