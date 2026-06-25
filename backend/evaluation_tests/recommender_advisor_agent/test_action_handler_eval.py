import asyncio
import logging
from pydantic import BaseModel
from pathlib import Path

import pytest

from app.agent.llm_caller import LLMCaller
from app.agent.recommender_advisor_agent.llm_response_models import (
    ActionExtractionResult,
    ConversationResponse,
)
from app.agent.recommender_advisor_agent.phase_handlers.action_handler import ActionPhaseHandler
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
)
from common_libs.llm.generative_models import GeminiGenerativeLLM
from common_libs.llm.models_utils import (
    JSON_GENERATION_CONFIG,
    LOW_TEMPERATURE_GENERATION_CONFIG,
    ZERO_TEMPERATURE_GENERATION_CONFIG,
    LLMConfig,
)
from common_libs.llm.schema_builder import with_response_schema
from evaluation_tests.compass_test_case import CompassTestCase

logger = logging.getLogger(__name__)
LOG_DIR = Path(__file__).parent / "logs"



class ActionExtractionTest(CompassTestCase):
    user_input: str
    expected_action_result: ActionExtractionResult


# Non-None value in a plan slot means "must be present in actual output".
# The exact text is not compared — only presence (not None) is asserted.
_PRESENT = "present"

TESTS: list[ActionExtractionTest] = [
    ActionExtractionTest(
        name="concrete commitment with when and how",
        user_input=(
            "Yes, I'll go to the Nyali construction site Thursday morning by matatu "
            "and ask the foreman directly about the apprenticeship opening."
        ),
        expected_action_result=ActionExtractionResult(
            reasoning="",
            has_commitment=True,
            action_type="apply_to_job",
            commitment_level="will_do_this_week",
            barriers_mentioned=[],
            plan_when=_PRESENT,
            plan_how=_PRESENT,
        ),
    ),
    ActionExtractionTest(
        name="vague strong commitment, no plan details stated",
        user_input="Yes I want to apply for the electrical apprenticeship, I'll do it this week.",
        expected_action_result=ActionExtractionResult(
            reasoning="",
            has_commitment=True,
            action_type="apply_to_job",
            commitment_level="will_do_this_week",
            barriers_mentioned=[],
        ),
    ),
    ActionExtractionTest(
        name="training enrollment commitment next month",
        user_input=(
            "I want to register for the electrician grade test course. "
            "I'll sign up at Mombasa Technical next month."
        ),
        expected_action_result=ActionExtractionResult(
            reasoning="",
            has_commitment=True,
            action_type="enroll_in_training",
            commitment_level="will_do_this_month",
            barriers_mentioned=[],
        ),
    ),
    ActionExtractionTest(
        name="barriers only, no commitment made",
        user_input=(
            "I'm not sure I can afford the bus fare to get there, "
            "and I don't know if they're still hiring for that position."
        ),
        expected_action_result=ActionExtractionResult(
            reasoning="",
            has_commitment=False,
            action_type=None,
            commitment_level=None,
            barriers_mentioned=["placeholder"],  # non-empty → assert actual has barriers
        ),
    ),
    ActionExtractionTest(
        name="full concrete plan including backup",
        user_input=(
            "I'll go Monday morning, take the 7am Coast Bus from Mwembe Tayari, "
            "bring my ID and ask for the site manager. "
            "If he's not there I'll come back Wednesday."
        ),
        expected_action_result=ActionExtractionResult(
            reasoning="",
            has_commitment=True,
            action_type="apply_to_job",
            commitment_level="will_do_this_week",
            barriers_mentioned=[],
            plan_when=_PRESENT,
            plan_how=_PRESENT,
            plan_backup=_PRESENT,
        ),
    ),
    ActionExtractionTest(
        name="networking commitment this week",
        user_input=(
            "I'll ask my uncle to introduce me to his contacts at the construction "
            "company this weekend — he knows the site supervisor."
        ),
        expected_action_result=ActionExtractionResult(
            reasoning="",
            has_commitment=True,
            action_type="network",
            commitment_level="will_do_this_week",
            barriers_mentioned=[],
        ),
    ),
    ActionExtractionTest(
        name="explicit refusal, no commitment",
        user_input="No, I don't think any of these are right for me right now.",
        expected_action_result=ActionExtractionResult(
            reasoning="",
            has_commitment=False,
            action_type=None,
            commitment_level=None,
            barriers_mentioned=[],
        ),
    ),
    ActionExtractionTest(
        name="barrier plus weak interest, no firm commitment",
        user_input=(
            "The pay sounds okay but I'm worried about the cost of the training. "
            "Maybe I could look into it at some point."
        ),
        expected_action_result=ActionExtractionResult(
            reasoning="",
            has_commitment=False,
            action_type=None,
            commitment_level=None,
            barriers_mentioned=["placeholder"],  # expects barriers (cost concern)
        ),
    ),
]

def _assert_extraction(
    actual: ActionExtractionResult,
    expected: ActionExtractionResult,
    description: str,
) -> None:
    assert actual.has_commitment == expected.has_commitment, (
        f"[{description}] has_commitment: got {actual.has_commitment!r}, "
        f"expected {expected.has_commitment!r}. reasoning='{actual.reasoning}'"
    )

    if expected.action_type is not None:
        assert actual.action_type == expected.action_type, (
            f"[{description}] action_type: got {actual.action_type!r}, "
            f"expected {expected.action_type!r}. reasoning='{actual.reasoning}'"
        )

    if expected.commitment_level is not None:
        assert actual.commitment_level == expected.commitment_level, (
            f"[{description}] commitment_level: got {actual.commitment_level!r}, "
            f"expected {expected.commitment_level!r}. reasoning='{actual.reasoning}'"
        )

    if expected.barriers_mentioned:
        assert actual.barriers_mentioned, (
            f"[{description}] expected barriers to be mentioned but got none. "
            f"reasoning='{actual.reasoning}'"
        )
    else:
        assert not actual.barriers_mentioned, (
            f"[{description}] expected no barriers but got {actual.barriers_mentioned!r}. "
            f"reasoning='{actual.reasoning}'"
        )

    for slot in ("plan_when", "plan_how", "plan_backup"):
        if getattr(expected, slot) is not None:
            assert getattr(actual, slot) is not None, (
                f"[{description}] {slot} should be present but was None. "
                f"reasoning='{actual.reasoning}'"
            )

async def _build_extraction_handler() -> ActionPhaseHandler:
    llm = GeminiGenerativeLLM(
        system_instructions="",
        config=LLMConfig(generation_config=LOW_TEMPERATURE_GENERATION_CONFIG | JSON_GENERATION_CONFIG),
    )
    action_llm = GeminiGenerativeLLM(
        config=LLMConfig(
            generation_config=ZERO_TEMPERATURE_GENERATION_CONFIG | with_response_schema(ActionExtractionResult)
        )
    )
    return ActionPhaseHandler(
        conversation_llm=llm,
        conversation_caller=LLMCaller[ConversationResponse](model_response_type=ConversationResponse),
        action_caller=LLMCaller[ActionExtractionResult](model_response_type=ActionExtractionResult),
        action_llm=action_llm,
    )


def _empty_context() -> ConversationContext:
    h = ConversationHistory()
    return ConversationContext(all_history=h, history=h, summary="")


@pytest.mark.asyncio
@pytest.mark.evaluation_test
@pytest.mark.parametrize(
    "test_case",
    TESTS,
    ids=[t.name for t in TESTS],
)
async def test_action_extraction(test_case: ActionExtractionTest):
    """Call _extract_action and assert the LLM produces the expected result."""
    handler = await _build_extraction_handler()
    context = _empty_context()

    result, _stats = await handler._extract_action(test_case.user_input, context)

    LOG_DIR.mkdir(exist_ok=True)
    safe_name = test_case.name.replace(" ", "_").replace(",", "")
    with open(LOG_DIR / f"action_extraction_{safe_name}.log", "w") as f:
        f.write(f"user_input : {test_case.user_input}\n")
        f.write(f"result     : {result}\n")

    assert result is not None, (
        f"[{test_case.name}] _extract_action returned None (LLM call failed)"
    )

    _assert_extraction(result, test_case.expected_action_result, test_case.name)
