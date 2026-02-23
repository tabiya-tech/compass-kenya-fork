import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Mapping
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_serializer, field_validator

from app.agent.agent_types import AgentInput, AgentOutput, AgentType
from app.conversation_memory.conversation_memory_manager import IConversationMemoryManager
from app.agent.persona_detector import PersonaType
from app.conversations.phase_state_machine import JourneyPhase


class ConversationPhase(Enum):
    """
    An enumeration for conversation phases
    """
    INTRO = 0
    COUNSELING = 1
    CHECKOUT = 2
    ENDED = 3


class CounselingSubPhase(Enum):
    """
    Deterministic sub-phases within the COUNSELING phase.
    Controls the agent progression: ExploreExperiences -> PreferenceElicitation -> RecommenderAdvisor.
    The LLM Router is only used within a sub-phase, not for transitions between them.
    """
    EXPLORE_EXPERIENCES = 0
    PREFERENCE_ELICITATION = 1
    RECOMMENDER_ADVISOR = 2


class AgentDirectorState(BaseModel):
    """
    The state of the agent director
    """
    session_id: int
    current_phase: ConversationPhase = Field(default=ConversationPhase.INTRO)
    counseling_sub_phase: CounselingSubPhase = Field(default=CounselingSubPhase.EXPLORE_EXPERIENCES)
    last_routed_agent: Optional[AgentType] = Field(default=None)
    sticky_turn_counter: int = Field(default=0)
    conversation_conducted_at: Optional[datetime] = None
    persona_type: PersonaType = Field(default=PersonaType.INFORMAL)
    skip_to_phase: Optional[JourneyPhase] = Field(default=None)

    class Config:
        extra = "forbid"
        json_encoders = {
            # ensure datetime values are serialized as a ISODate object
            datetime: lambda dt: dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        }

    def __setattr__(self, key, value):
        if key == "conversation_conducted_at":
            value = _parse_data(value)
        super().__setattr__(key, value)

    # use a field serializer to serialize the current_phase
    # we use the name of the Enum instead of the value because that makes the code less brittle
    @field_serializer("current_phase")
    def serialize_current_phase(self, current_phase: ConversationPhase, _info):
        return current_phase.name

    # Deserialize the current_phase from the enum name
    @field_validator("current_phase", mode='before')
    def deserialize_current_phase(cls, value: str | ConversationPhase) -> ConversationPhase:
        if isinstance(value, str):
            return ConversationPhase[value]
        return value

    @field_serializer("counseling_sub_phase")
    def serialize_counseling_sub_phase(self, v: CounselingSubPhase, _info):
        return v.name

    @field_validator("counseling_sub_phase", mode='before')
    def deserialize_counseling_sub_phase(cls, value: str | CounselingSubPhase) -> CounselingSubPhase:
        if isinstance(value, str):
            return CounselingSubPhase[value]
        return value

    @field_serializer("last_routed_agent")
    def serialize_last_routed_agent(self, v: Optional[AgentType], _info) -> Optional[str]:
        return v.name if v is not None else None

    @field_validator("last_routed_agent", mode='before')
    def deserialize_last_routed_agent(cls, value: str | AgentType | None) -> Optional[AgentType]:
        if value is None:
            return None
        if isinstance(value, str):
            return AgentType[value]
        return value

    @field_serializer("persona_type")
    def serialize_persona_type(self, persona_type: PersonaType, _info):
        return persona_type.name

    @field_validator("persona_type", mode='before')
    def deserialize_persona_type(cls, value: str | PersonaType) -> PersonaType:
        if isinstance(value, str):
            return PersonaType[value]
        return value

    @field_serializer("skip_to_phase")
    def serialize_skip_to_phase(self, v: Optional[JourneyPhase], _info) -> Optional[str]:
        return v.value if v is not None else None

    @field_validator("skip_to_phase", mode='before')
    def deserialize_skip_to_phase(cls, value: str | JourneyPhase | None) -> Optional[JourneyPhase]:
        if value is None:
            return None
        if isinstance(value, str):
            return JourneyPhase(value)
        return value

    # Deserialize the conversation_conducted_at datetime and ensure it's interpreted as UTC
    @field_validator("conversation_conducted_at", mode='before')
    def deserialize_conversation_conducted_at(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _parse_data(value)

    @staticmethod
    def from_document(_doc: Mapping[str, Any]) -> "AgentDirectorState":
        skip_to_phase = None
        if _doc.get("skip_to_phase"):
            skip_to_phase = JourneyPhase(_doc["skip_to_phase"])
        return AgentDirectorState(session_id=_doc["session_id"],
                                  current_phase=_doc["current_phase"],
                                  counseling_sub_phase=_doc.get("counseling_sub_phase",
                                                                CounselingSubPhase.EXPLORE_EXPERIENCES),
                                  last_routed_agent=_doc.get("last_routed_agent", None),
                                  sticky_turn_counter=_doc.get("sticky_turn_counter", 0),
                                  conversation_conducted_at=_doc.get("conversation_conducted_at", None),
                                  persona_type=_doc.get("persona_type", PersonaType.INFORMAL),
                                  skip_to_phase=skip_to_phase)


def _parse_data(value: Optional[datetime | str]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            # Convert string to datetime
            value = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError(f"Invalid datetime string: {value}")

    # Always assume UTC timezone even for naive datetimes. This is important because MongoDB stores implicitly datetimes as UTC
    # but returns them as naive datetimes.
    # Convert to UTC and truncate microseconds to milliseconds as MongoDB does not support microseconds

    return value.replace(tzinfo=timezone.utc).replace(microsecond=(value.microsecond // 1000 * 1000))


class AbstractAgentDirector(ABC):
    """
    An abstract class for an agent director. Receives user input,
    understands the conversation context and the latest user message and routes the user input to the appropriate agent.
    It maintains the state of the conversation which is divided into phases.
    """

    def __init__(self, conversation_manager: IConversationMemoryManager):
        # Initialize the logger
        self._logger = logging.getLogger(self.__class__.__name__)

        # set the conversation manager
        self._conversation_manager = conversation_manager

        self._state: AgentDirectorState | None = None

    def set_state(self, state: AgentDirectorState):
        """
        Set the agent director state
        :param state: the agent director state
        """
        self._state = state

    @abstractmethod
    async def execute(self, user_input: AgentInput) -> AgentOutput:
        """
        Run the conversation task for the current user input and specific state.
        :param user_input:
        :return:
        """
        raise NotImplementedError()
