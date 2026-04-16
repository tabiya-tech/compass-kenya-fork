"""
Evaluation tests for the Preference Elicitation Agent.

Runs simulated users through the full preference elicitation flow and asserts:
1. Behavioral: extracted PreferenceVector reflects the persona's preferences
2. Conversation quality: coherence, relevance, no crashes or loops

Run with:
    poetry run pytest evaluation_tests/preference_elicitation_agent/ -v -m evaluation_test
"""

import json
import logging
import os

import pytest
from _pytest.logging import LogCaptureFixture

from app.conversation_memory.conversation_memory_manager import ConversationMemoryManager
from app.conversation_memory.conversation_memory_types import ConversationMemoryManagerState
from app.server_config import UNSUMMARIZED_WINDOW_SIZE, TO_BE_SUMMARIZED_WINDOW_SIZE
from common_libs.test_utilities import get_random_session_id
from common_libs.test_utilities.guard_caplog import guard_caplog, assert_log_error_warnings
from evaluation_tests.conversation_libs.conversation_test_function import (
    LLMSimulatedUser,
    ConversationTestConfig,
    conversation_test_function,
    assert_expected_evaluation_results,
)
from evaluation_tests.conversation_libs.evaluators.evaluation_result import ConversationEvaluationRecord
from evaluation_tests.get_test_cases_to_run_func import get_test_cases_to_run
from evaluation_tests.preference_elicitation_agent.preference_elicitation_executor import (
    PreferenceElicitationAgentExecutor,
    PreferenceElicitationAgentGetConversationContextExecutor,
    PreferenceElicitationAgentIsFinished,
)
from evaluation_tests.preference_elicitation_agent.preference_elicitation_test_cases import (
    test_cases,
    PreferenceElicitationTestCase,
)


@pytest.mark.asyncio
@pytest.mark.evaluation_test("gemini-2.5-flash-lite/")
@pytest.mark.repeat(1)
@pytest.mark.parametrize(
    "test_case",
    get_test_cases_to_run(test_cases),
    ids=[case.name for case in get_test_cases_to_run(test_cases)],
)
async def test_preference_elicitation_simulated_user(
    test_case: PreferenceElicitationTestCase,
    caplog: LogCaptureFixture,
    setup_application_config,
):
    """
    Runs each persona through the full preference elicitation flow.

    Asserts:
    - Conversation completes (agent signals finished)
    - PreferenceVector reflects persona's stated preferences (behavioral)
    - Conversation quality scores meet thresholds (coherence, relevance)
    - No errors or unexpected warnings logged
    """
    print(f"\nRunning preference elicitation eval: {test_case.name}")

    session_id = get_random_session_id()
    output_folder = os.path.join(
        os.getcwd(),
        "test_output/preference_elicitation/simulated_user/",
        test_case.name,
    )

    conversation_manager = ConversationMemoryManager(
        UNSUMMARIZED_WINDOW_SIZE, TO_BE_SUMMARIZED_WINDOW_SIZE
    )
    conversation_manager.set_state(
        state=ConversationMemoryManagerState(session_id=session_id)
    )

    execute_evaluated_agent = PreferenceElicitationAgentExecutor(
        conversation_manager=conversation_manager,
        session_id=session_id,
        use_adaptive_selection=True,
    )

    config = ConversationTestConfig(
        max_iterations=20,  # generous ceiling; Bayesian stopping fires at 9-12
        test_case=test_case,
        output_folder=output_folder,
        execute_evaluated_agent=execute_evaluated_agent,
        execute_simulated_user=LLMSimulatedUser(
            system_instructions=test_case.simulated_user_prompt
        ),
        is_finished=PreferenceElicitationAgentIsFinished(),
        get_conversation_context=PreferenceElicitationAgentGetConversationContextExecutor(
            conversation_manager=conversation_manager
        ),
        deferred_evaluation_assertions=True,
    )

    with caplog.at_level(logging.DEBUG):
        guard_caplog(logger=execute_evaluated_agent._agent._logger, caplog=caplog)

        evaluation_result: ConversationEvaluationRecord = await conversation_test_function(
            config=config
        )

    # --- 1. Behavioral assertions: PreferenceVector correctness ---
    pv = execute_evaluated_agent.get_preference_vector()
    state = execute_evaluated_agent.get_state()

    # Save preference vector alongside conversation output for comparison
    _save_preference_vector_snapshot(output_folder, test_case.name, pv, state)

    failures = test_case.check_preference_vector(pv)
    if failures:
        pytest.fail(
            f"Preference vector behavioral assertions failed for '{test_case.name}':\n"
            + "\n".join(f"  - {f}" for f in failures)
        )

    # --- 2. Conversation quality assertions ---
    assert_expected_evaluation_results(
        evaluation_result=evaluation_result,
        test_case=test_case,
    )

    # --- 3. Error log check (errors indicate Bayesian runtime failures) ---
    assert_log_error_warnings(
        caplog=caplog,
        expect_errors_in_logs=test_case.expect_errors_in_logs,
        expect_warnings_in_logs=True,  # warnings always expected (e.g. missing experiences)
    )


def _save_preference_vector_snapshot(
    output_folder: str,
    test_name: str,
    pv,
    state,
) -> None:
    """Save a JSON snapshot of the preference vector for offline comparison."""
    os.makedirs(output_folder, exist_ok=True)
    snapshot = {
        "test_case": test_name,
        "preference_vector": pv.model_dump(),
        "n_vignettes_completed": pv.n_vignettes_completed,
        "confidence_score": pv.confidence_score,
        "posterior_mean": pv.posterior_mean,
        "posterior_covariance_diagonal": pv.posterior_covariance_diagonal,
        "per_dimension_uncertainty": pv.per_dimension_uncertainty,
        "fim_determinant": pv.fim_determinant,
        "stopped_early": state.stopped_early,
        "stopping_reason": state.stopping_reason,
    }
    path = os.path.join(output_folder, f"{test_name}_preference_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"  Preference snapshot saved: file:///{path}")
