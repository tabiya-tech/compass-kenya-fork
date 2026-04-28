"""
Tests for three targeted fixes in the preference elicitation agent:
1. Location-agnostic vignettes (no Nairobi hardcoding)
2. Multi-experience context formatting (no forced single-background pick)
3. BWS findings included in final summary
"""

import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.preference_elicitation_agent.types import UserContext, PreferenceVector
from app.agent.preference_elicitation_agent.vignette_personalizer import VignettePersonalizer
from app.agent.preference_elicitation_agent.state import PreferenceElicitationAgentState
from app.agent.preference_elicitation_agent.agent import (
    PreferenceElicitationAgent,
    PreferenceSummaryGenerator,
)
from app.agent.preference_elicitation_agent import bws_utils
from app.agent.llm_caller import LLMCaller


class TestLocationAgnosticVignettes:
    """Issue 1: system prompt must not hardcode Nairobi or any specific city."""

    def test_no_nairobi_in_source(self):
        import inspect
        source = inspect.getsource(VignettePersonalizer)
        assert "Nairobi" not in source, "VignettePersonalizer source still contains 'Nairobi'"


class TestMultiExperienceContextFormatting:
    """Issue 2: _format_user_context should expose all backgrounds, not force-pick one."""

    @pytest.fixture
    def personalizer(self):
        # Bypass __init__ — _format_user_context is pure string logic, uses no self state
        return object.__new__(VignettePersonalizer)

    def test_all_backgrounds_listed(self, personalizer):
        context = UserContext(
            current_role="Software Developer",
            industry="Technology",
            experience_level="junior",
            all_backgrounds=["Software Developer | Technology", "Sales Associate | Retail"],
        )
        result = personalizer._format_user_context(context)
        assert "Software Developer | Technology" in result
        assert "Sales Associate | Retail" in result

    def test_open_instruction_present(self, personalizer):
        context = UserContext(
            current_role="Teacher",
            industry="Education",
            experience_level="mid",
            all_backgrounds=["Teacher | Education", "Tutor | Education", "Cashier | Retail"],
        )
        result = personalizer._format_user_context(context, previous_vignettes=["Previous scenario"])
        assert "Use any of these backgrounds" in result

    def test_no_mandatory_pick_one_constraint(self, personalizer):
        context = UserContext(
            current_role="Nurse",
            industry="Healthcare",
            experience_level="mid",
            all_backgrounds=["Nurse | Healthcare", "Shop Assistant | Retail"],
        )
        for prev in [None, ["Some earlier vignette scenario"]]:
            result = personalizer._format_user_context(context, previous_vignettes=prev)
            assert "MANDATORY" not in result
            assert "Pick ONE" not in result
            assert "Pick a background" not in result

    def test_single_background_no_extra_section(self, personalizer):
        # len(all_backgrounds) == 1 → no extra section shown
        context = UserContext(
            current_role="Driver",
            industry="Transportation",
            experience_level="junior",
            all_backgrounds=["Driver | Transportation"],
        )
        result = personalizer._format_user_context(context)
        assert "All Past Experience Contexts" not in result
        assert "Use any of these backgrounds" not in result


class TestBWSFindingsInSummary:
    """Issue 3: _generate_preference_summary must include BWS top activities when available."""

    def _make_agent(self, bws_complete: bool, top_bws: list) -> PreferenceElicitationAgent:
        agent = object.__new__(PreferenceElicitationAgent)
        agent._logger = logging.getLogger("test")  # base Agent exposes this via read-only property
        agent._state = PreferenceElicitationAgentState(session_id=99)
        agent._state.preference_vector = PreferenceVector()
        agent._state.bws_phase_complete = bws_complete
        agent._state.top_10_bws = top_bws
        agent._conversation_llm = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_summary_prompt_includes_bws_section(self):
        top_ids = ["4.A.4.b.4", "4.A.2.b.1", "4.A.3.b.1", "4.A.1.b.2", "4.A.3.a.1"]
        agent = self._make_agent(bws_complete=True, top_bws=top_ids)

        mock_response = PreferenceSummaryGenerator(
            reasoning="ok", finished=True, message="• Prefers analytical work"
        )

        with patch.object(
            LLMCaller, "call_llm", new_callable=AsyncMock, return_value=(mock_response, [])
        ) as mock_call:
            await agent._generate_preference_summary()

        prompt = mock_call.call_args.kwargs["llm_input"]
        assert "Top Work Activities" in prompt
        wa_labels = bws_utils.load_wa_labels()
        assert wa_labels["4.A.4.b.4"] in prompt

    @pytest.mark.asyncio
    async def test_summary_prompt_omits_bws_when_not_complete(self):
        agent = self._make_agent(bws_complete=False, top_bws=[])

        mock_response = PreferenceSummaryGenerator(
            reasoning="ok", finished=True, message="• Some preferences"
        )

        with patch.object(
            LLMCaller, "call_llm", new_callable=AsyncMock, return_value=(mock_response, [])
        ) as mock_call:
            await agent._generate_preference_summary()

        prompt = mock_call.call_args.kwargs["llm_input"]
        assert "Top Work Activities" not in prompt
