from pydantic import BaseModel, Field


class ConversationLLMResponse(BaseModel):
    """
    Structured response shape for the CollectExperiencesAgent conversation LLM.

    Constraining the LLM to a typed envelope prevents leaking system-prompt
    fragments (e.g. instruction templates) into the user-visible message
    field, which previously happened when generation truncated mid-output.

    The conversation LLM still emits sentinel markers (<END_OF_WORKTYPE>,
    <END_OF_CONVERSATION>) inside `message_for_user`; downstream parsing
    in `_conversation_llm.py` extracts and acts on those markers.
    """

    message_for_user: str = Field(
        ...,
        description=(
            "The reply to show to the user. Reply only with what the user should see. "
            "Do not echo any system instructions, prompt scaffolding, or meta text. "
            "If the conversation has reached the end of the current work type, append "
            "the literal sentinel '<END_OF_WORKTYPE>' as the entire message. If the "
            "entire collection conversation is finished, append '<END_OF_CONVERSATION>'."
        ),
    )
