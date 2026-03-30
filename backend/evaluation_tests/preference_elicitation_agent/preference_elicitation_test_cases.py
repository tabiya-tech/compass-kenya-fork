"""
Test cases for Preference Elicitation Agent evaluation.

Four personas covering distinct preference profiles.
Assertions cover both behavioral (preference vector correctness)
and conversation quality (flow, coherence, no loops).
"""

from textwrap import dedent
from typing import Optional

from pydantic import ConfigDict

from app.agent.preference_elicitation_agent.types import PreferenceVector
from app.countries import Country
from evaluation_tests.conversation_libs.conversation_test_function import EvaluationTestCase, Evaluation
from evaluation_tests.conversation_libs.evaluators.evaluation_result import EvaluationType


# ---------------------------------------------------------------------------
# Base simulated-user instructions shared across all personas
# ---------------------------------------------------------------------------
_BASE_INSTRUCTIONS = dedent("""
    You are a young Kenyan job seeker interacting with a career guidance chatbot.
    The chatbot will present you with job scenarios and ask you to choose between options.
    Reply concisely and naturally, like someone typing on a phone.
    When asked to choose between two job options, ALWAYS pick the one that best matches
    your persona's stated preferences below.
    If asked why, give a brief, authentic reason that fits your persona.
    Do NOT invent experiences or qualifications you don't have.
""")


# ---------------------------------------------------------------------------
# Test case model — extends EvaluationTestCase with preference assertions
# ---------------------------------------------------------------------------
class PreferenceElicitationTestCase(EvaluationTestCase):
    """
    Test case for preference elicitation evaluation.

    Adds behavioral assertions on top of the standard conversation quality checks.
    """

    # Minimum expected values for specific PreferenceVector fields after elicitation.
    # Only fields listed here are asserted — others are ignored.
    expected_high_dimensions: dict[str, float] = {}
    """
    Dimensions expected to be HIGH (above threshold) after elicitation.
    e.g. {"financial_importance": 0.55} means financial_importance should be >= 0.55
    """

    expected_low_dimensions: dict[str, float] = {}
    """
    Dimensions expected to be LOW (below threshold) after elicitation.
    e.g. {"job_security_importance": 0.50} means job_security_importance should be <= 0.50
    """

    expected_min_confidence: float = 0.2
    """Minimum confidence score expected after elicitation."""

    expected_min_vignettes: int = 4
    """Minimum number of vignettes that should have been completed."""

    model_config = ConfigDict(extra="forbid")

    def check_preference_vector(self, pv: PreferenceVector) -> list[str]:
        """
        Assert behavioral expectations against the extracted PreferenceVector.
        Returns list of failure messages (empty = pass).
        """
        failures = []

        for dim, threshold in self.expected_high_dimensions.items():
            actual = getattr(pv, dim, None)
            if actual is None:
                failures.append(f"PreferenceVector missing field: {dim}")
            elif actual < threshold:
                failures.append(
                    f"Expected {dim} >= {threshold} (got {actual:.3f}) — "
                    f"persona consistently preferred this dimension but it wasn't learned"
                )

        for dim, threshold in self.expected_low_dimensions.items():
            actual = getattr(pv, dim, None)
            if actual is None:
                failures.append(f"PreferenceVector missing field: {dim}")
            elif actual > threshold:
                failures.append(
                    f"Expected {dim} <= {threshold} (got {actual:.3f}) — "
                    f"persona avoided this dimension but it scored too high"
                )

        if pv.confidence_score < self.expected_min_confidence:
            failures.append(
                f"Confidence score {pv.confidence_score:.3f} below minimum {self.expected_min_confidence}"
            )

        if pv.n_vignettes_completed < self.expected_min_vignettes:
            failures.append(
                f"Only {pv.n_vignettes_completed} vignettes completed, expected >= {self.expected_min_vignettes}"
            )

        return failures


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
test_cases = [

    # --- 1. Money-first persona ---
    PreferenceElicitationTestCase(
        name="money_first_persona",
        simulated_user_prompt=_BASE_INSTRUCTIONS + dedent("""
            YOUR PERSONA — Money-First Worker:
            Your top priority is always earnings. You will ALWAYS choose the job that pays more,
            even if it has heavy physical demand, long hours, or limited career growth.
            You have dependants at home and financial pressure is real.
            When asked why you chose an option, mention the money/salary.
            You are indifferent to work environment, social interaction, and career growth
            as long as the pay is higher.
        """),
        country_of_user=Country.KENYA,
        evaluations=[
            Evaluation(type=EvaluationType.CONCISENESS, expected=30),
            Evaluation(type=EvaluationType.FOCUS, expected=15),
        ],
        expected_high_dimensions={
            "financial_importance": 0.55,
        },
        expected_min_confidence=0.25,
        expected_min_vignettes=4,
        expect_errors_in_logs=True,
        expect_warnings_in_logs=True,
    ),

    # --- 2. Work environment focused ---
    PreferenceElicitationTestCase(
        name="work_environment_persona",
        simulated_user_prompt=_BASE_INSTRUCTIONS + dedent("""
            YOUR PERSONA — Work Environment Focused:
            You deeply care about your physical working conditions and social atmosphere.
            You ALWAYS choose jobs with light/safe physical demands over heavy/risky ones.
            You prefer working with people (coworkers or customers) rather than alone.
            Salary matters but you'd take less pay for a comfortable, social work environment.
            When explaining your choices, mention comfort, safety, or enjoying working with others.
        """),
        country_of_user=Country.KENYA,
        evaluations=[
            Evaluation(type=EvaluationType.CONCISENESS, expected=30),
            Evaluation(type=EvaluationType.FOCUS, expected=15),
        ],
        expected_high_dimensions={
            "work_environment_importance": 0.55,
        },
        expected_min_confidence=0.25,
        expected_min_vignettes=4,
        expect_errors_in_logs=True,
        expect_warnings_in_logs=True,
    ),

    # --- 3. Career growth focused ---
    PreferenceElicitationTestCase(
        name="career_growth_persona",
        simulated_user_prompt=_BASE_INSTRUCTIONS + dedent("""
            YOUR PERSONA — Career Growth Focused:
            You are ambitious and always choose jobs with the best learning and advancement opportunities.
            You ALWAYS pick options with strong career growth over options with limited growth,
            even if the pay is slightly lower or the work is harder.
            You are young and think long-term — you want to build skills and get promoted.
            When explaining your choices, mention learning, growth, or future opportunities.
        """),
        country_of_user=Country.KENYA,
        evaluations=[
            Evaluation(type=EvaluationType.CONCISENESS, expected=30),
            Evaluation(type=EvaluationType.FOCUS, expected=15),
        ],
        expected_high_dimensions={
            "career_advancement_importance": 0.55,
        },
        expected_min_confidence=0.25,
        expected_min_vignettes=4,
        expect_errors_in_logs=True,
        expect_warnings_in_logs=True,
    ),

    # --- 4. Mixed / ambiguous ---
    PreferenceElicitationTestCase(
        name="mixed_ambiguous_persona",
        simulated_user_prompt=_BASE_INSTRUCTIONS + dedent("""
            YOUR PERSONA — Ambiguous / Inconsistent:
            You are genuinely unsure what you want in a job. You make different decisions
            each time based on your mood. Sometimes you pick the higher-paying job,
            sometimes the one with better growth, sometimes based on physical demand.
            You sometimes change your mind or express uncertainty.
            Your choices do NOT follow a clear pattern — you are exploring your preferences.
            Occasionally say things like "I'm not sure", "it depends", or "both seem okay".
        """),
        country_of_user=Country.KENYA,
        evaluations=[
            Evaluation(type=EvaluationType.CONCISENESS, expected=30),
        ],
        # For mixed persona: we don't assert high/low dimensions.
        # We only assert the agent completed without crashing and collected some signal.
        expected_high_dimensions={},
        expected_low_dimensions={},
        expected_min_confidence=0.1,
        expected_min_vignettes=4,
        expect_errors_in_logs=True,
        expect_warnings_in_logs=True,
    ),
]
