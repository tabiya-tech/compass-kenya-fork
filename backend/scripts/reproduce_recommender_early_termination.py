#!/usr/bin/env python3
"""
Reproduces the bug where the recommender agent abruptly terminates the conversation
after the user rejects all recommendations in the ADDRESS_CONCERNS phase.

Transcript being reproduced:
  Turn 0: (initial) → agent presents 5 occupations
  Turn 1: "yeah"
  Turn 2: "I don't think I have the skills to become a Web Developer"  ← concern
  Turn 3: "No I don't like any of them"  ← blanket rejection in ADDRESS_CONCERNS ← BUG HERE
  Turn 4: "No I want to keep talking"
  Turn 5: "More careers"
  Turn 6: "No don't end. What about jobs, do you have any recommendations here?"  ← session ends

Usage:
    cd backend/
    poetry run python scripts/reproduce_recommender_early_termination.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.markdown import Markdown

from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    Node2VecRecommendations,
    OccupationRecommendation,
    ScoreBreakdown,
    SkillComponent,
)
from app.agent.preference_elicitation_agent.types import PreferenceVector
from app.agent.recommender_advisor_agent.phase_handlers.intro_handler import IntroPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.present_handler import PresentPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.exploration_handler import ExplorationPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.concerns_handler import ConcernsPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.tradeoffs_handler import TradeoffsPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.followup_handler import FollowupPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.skills_pivot_handler import SkillsPivotPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.action_handler import ActionPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.wrapup_handler import WrapupPhaseHandler
from app.agent.recommender_advisor_agent.intent_classifier import IntentClassifier
from app.agent.recommender_advisor_agent.recommendation_interface import RecommendationInterface
from app.agent.recommender_advisor_agent.llm_response_models import (
    ConversationResponse,
    ResistanceClassification,
    UserIntentClassification,
    ActionExtractionResult,
)
from app.agent.llm_caller import LLMCaller
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
    ConversationTurn,
)
from app.agent.agent_types import AgentInput, AgentOutput
from common_libs.llm.generative_models import GeminiGenerativeLLM
from common_libs.llm.models_utils import LLMConfig, MEDIUM_TEMPERATURE_GENERATION_CONFIG, JSON_GENERATION_CONFIG
from app.countries import Country

console = Console()

# ---------------------------------------------------------------------------
# Recommendations matching the bug transcript
# ---------------------------------------------------------------------------

def create_transcript_recommendations() -> Node2VecRecommendations:
    """
    Build the 5 occupations that appear in the bug transcript.
    No training recommendations — so should_pivot_to_training() can never fire,
    which is also part of the failure path.
    """
    return Node2VecRecommendations(
        youth_id="bug_repro_user",
        generated_at="2026-04-24T10:00:00Z",
        recommended_by=["Algorithm"],
        occupation_recommendations=[
            OccupationRecommendation(
                uuid="occ_borehole_uuid",
                originUuid="kesco_borehole_origin",
                rank=1,
                occupation_id="KESCO_BOREHOLE",
                occupation_code="8121",
                occupation="Borehole Driller",
                final_score=0.80,
                score_breakdown=ScoreBreakdown(
                    total_skill_utility=0.75,
                    skill_components=SkillComponent(loc=0.80, ess=0.72, opt=0.74, grp=0.76),
                    skill_penalty_applied=0.0,
                    preference_score=0.78,
                    demand_score=0.72,
                    demand_label="Moderate Demand",  # → labor_demand_category == "medium"
                ),
                salary_range="KES 30,000/month",
                justification="Good earning potential; high career growth; involves working with people.",
            ),
            OccupationRecommendation(
                uuid="occ_webdev_uuid",
                originUuid="kesco_webdev_origin",
                rank=2,
                occupation_id="KESCO_WEBDEV",
                occupation_code="2512",
                occupation="Web Developer",
                final_score=0.78,
                score_breakdown=ScoreBreakdown(
                    total_skill_utility=0.70,
                    skill_components=SkillComponent(loc=0.75, ess=0.68, opt=0.70, grp=0.72),
                    skill_penalty_applied=0.0,
                    preference_score=0.80,
                    demand_score=0.74,
                    demand_label="Moderate Demand",
                ),
                salary_range="KES 30,000/month",
                justification="Strong earning potential; high career growth; involves working with people.",
            ),
            OccupationRecommendation(
                uuid="occ_truck_uuid",
                originUuid="kesco_truck_origin",
                rank=3,
                occupation_id="KESCO_TRUCK",
                occupation_code="8322",
                occupation="TRUCK DRIVER",
                final_score=0.75,
                score_breakdown=ScoreBreakdown(
                    total_skill_utility=0.70,
                    skill_components=SkillComponent(loc=0.78, ess=0.65, opt=0.68, grp=0.70),
                    skill_penalty_applied=0.0,
                    preference_score=0.72,
                    demand_score=0.70,
                    demand_label="Moderate Demand",
                ),
                salary_range="KES 30,000/month",
                justification="Good earnings; high growth; opportunities to interact with people. Zambia only.",
            ),
            OccupationRecommendation(
                uuid="occ_electrician_uuid",
                originUuid="kesco_elec_origin",
                rank=4,
                occupation_id="KESCO_ELEC",
                occupation_code="7411",
                occupation="Electrician",
                final_score=0.82,
                score_breakdown=ScoreBreakdown(
                    total_skill_utility=0.80,
                    skill_components=SkillComponent(loc=0.85, ess=0.78, opt=0.80, grp=0.82),
                    skill_penalty_applied=0.0,
                    preference_score=0.80,
                    demand_score=0.90,
                    demand_label="High Demand",     # → labor_demand_category == "high"
                ),
                salary_range="KES 25,000-40,000/month",
                justification="High demand; leverages practical skills; good advancement opportunities.",
            ),
            OccupationRecommendation(
                uuid="occ_port_uuid",
                originUuid="kesco_port_origin",
                rank=5,
                occupation_id="KESCO_PORT",
                occupation_code="9333",
                occupation="Port Cargo Handler",
                final_score=0.72,
                score_breakdown=ScoreBreakdown(
                    total_skill_utility=0.65,
                    skill_components=SkillComponent(loc=0.88, ess=0.60, opt=0.62, grp=0.65),
                    skill_penalty_applied=0.0,
                    preference_score=0.70,
                    demand_score=0.88,
                    demand_label="High Demand",
                ),
                salary_range="KES 20,000-35,000/month",
                justification="High demand; solid salary range; physical work and equipment operation.",
            ),
        ],
        opportunity_recommendations=[],
        skillstraining_recommendations=[],   # <-- empty: pivot to training can never fire
    )


def create_preference_vector() -> PreferenceVector:
    return PreferenceVector(
        financial_importance=0.85,
        work_environment_importance=0.70,
        career_advancement_importance=0.55,
        work_life_balance_importance=0.55,
        job_security_importance=0.50,
        task_preference_importance=0.50,
        social_impact_importance=0.45,
    )


# ---------------------------------------------------------------------------
# Handler initialisation (mirrors test_recommender_agent_interactive.py)
# ---------------------------------------------------------------------------

async def build_handlers(llm: GeminiGenerativeLLM):
    conversation_caller = LLMCaller[ConversationResponse](model_response_type=ConversationResponse)
    resistance_caller = LLMCaller[ResistanceClassification](model_response_type=ResistanceClassification)
    intent_caller = LLMCaller[UserIntentClassification](model_response_type=UserIntentClassification)
    action_caller = LLMCaller[ActionExtractionResult](model_response_type=ActionExtractionResult)

    intent_classifier = IntentClassifier(intent_caller=intent_caller)
    recommendation_interface = RecommendationInterface(node2vec_client=None)

    action_handler = ActionPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        action_caller=action_caller,
        intent_classifier=intent_classifier,
    )
    concerns_handler = ConcernsPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        resistance_caller=resistance_caller,
        intent_classifier=intent_classifier,
        action_handler=action_handler,
    )
    tradeoffs_handler = TradeoffsPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
    )
    wrapup_handler = WrapupPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        db6_client=None,
    )
    followup_handler = FollowupPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
    )
    exploration_handler = ExplorationPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
        concerns_handler=concerns_handler,
        tradeoffs_handler=tradeoffs_handler,
        occupation_search_service=None,
    )
    skills_pivot_handler = SkillsPivotPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
    )
    present_handler = PresentPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
        exploration_handler=exploration_handler,
        concerns_handler=concerns_handler,
        tradeoffs_handler=tradeoffs_handler,
        occupation_search_service=None,
    )
    intro_handler = IntroPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        recommendation_interface=recommendation_interface,
        occupation_search_service=None,
    )

    # Wire delegation chains
    exploration_handler._action_handler = action_handler
    exploration_handler._skills_pivot_handler = skills_pivot_handler
    present_handler._skills_pivot_handler = skills_pivot_handler
    action_handler._present_handler = present_handler
    action_handler._concerns_handler = concerns_handler
    action_handler._wrapup_handler = wrapup_handler
    skills_pivot_handler._exploration_handler = exploration_handler
    skills_pivot_handler._concerns_handler = concerns_handler
    skills_pivot_handler._action_planning_handler = action_handler
    skills_pivot_handler._present_handler = present_handler

    return {
        ConversationPhase.INTRO: intro_handler,
        ConversationPhase.PRESENT_RECOMMENDATIONS: present_handler,
        ConversationPhase.CAREER_EXPLORATION: exploration_handler,
        ConversationPhase.ADDRESS_CONCERNS: concerns_handler,
        ConversationPhase.DISCUSS_TRADEOFFS: tradeoffs_handler,
        ConversationPhase.FOLLOW_UP: followup_handler,
        ConversationPhase.SKILLS_UPGRADE_PIVOT: skills_pivot_handler,
        ConversationPhase.ACTION_PLANNING: action_handler,
        ConversationPhase.WRAPUP: wrapup_handler,
        ConversationPhase.COMPLETE: wrapup_handler,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _phase_color(phase: ConversationPhase) -> str:
    colors = {
        ConversationPhase.PRESENT_RECOMMENDATIONS: "cyan",
        ConversationPhase.ADDRESS_CONCERNS: "yellow",
        ConversationPhase.ACTION_PLANNING: "magenta",
        ConversationPhase.WRAPUP: "red",
        ConversationPhase.COMPLETE: "bold red",
    }
    return colors.get(phase, "white")


def print_turn_header(turn_idx: int, user_msg: str):
    console.rule(f"[bold]Turn {turn_idx}[/]")
    console.print(Panel(Text(user_msg or "(initial presentation)"), title="[bold cyan]User[/]", border_style="cyan", box=box.ROUNDED))


def print_agent_response(response: ConversationResponse, phase_before: ConversationPhase, phase_after: ConversationPhase, state: RecommenderAdvisorAgentState):
    color = _phase_color(phase_after)

    # Agent message
    console.print(Panel(
        Markdown(response.message),
        title="[bold green]Agent[/]",
        border_style="green" if not response.finished else "bold red",
        box=box.ROUNDED,
    ))

    # Diagnostics row
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("key", style="dim")
    table.add_column("value")

    phase_changed = phase_before != phase_after
    phase_str = f"[{color}]{phase_after.value}[/]"
    if phase_changed:
        phase_str = f"[dim]{phase_before.value}[/] → {phase_str}"

    table.add_row("phase", phase_str)
    table.add_row("finished", f"[bold red]TRUE ← BUG[/]" if response.finished else "[green]False[/]")
    table.add_row("rejected_occupations", str(state.rejected_occupations))
    table.add_row("concerns_raised", str(len(state.concerns_raised)))
    if state.concerns_raised:
        latest = state.concerns_raised[-1]
        table.add_row("latest_concern_type", str(latest.resistance_type))
    table.add_row("reasoning", f"[dim]{response.reasoning[:120]}[/]")

    console.print(table)


# ---------------------------------------------------------------------------
# Main reproduction
# ---------------------------------------------------------------------------

TRANSCRIPT_TURNS = [
    "",                                                               # Turn 0 – initial presentation
    "yeah",                                                           # Turn 1
    "I don't think I have the skills to become a Web Developer",      # Turn 2
    "No I don't like any of them",                                    # Turn 3 ← bug trigger
    "No I want to keep talking",                                      # Turn 4
    "More careers",                                                   # Turn 5
    "No don't end. What about jobs, do you have any recommendations here?",  # Turn 6
]

BUG_TURN = 3  # turn where the premature termination path begins


async def run_reproduction():
    console.print(Panel(
        Text("Reproducing: recommender agent abruptly ends conversation\n"
             "Bug trigger: blanket rejection in ADDRESS_CONCERNS phase",
             justify="center"),
        style="bold magenta",
        box=box.DOUBLE,
    ))

    # Initialise LLM
    llm_config = LLMConfig(generation_config=MEDIUM_TEMPERATURE_GENERATION_CONFIG | JSON_GENERATION_CONFIG)
    llm = GeminiGenerativeLLM(
        system_instructions=(
            "You are a career advisor. Always respond with valid JSON matching the ConversationResponse schema. "
            "Set finished=false unless the user has made a strong commitment and you are providing a final summary."
        ),
        config=llm_config,
    )

    handlers = await build_handlers(llm)

    # Initial state – start at PRESENT_RECOMMENDATIONS (preferences already elicited)
    recommendations = create_transcript_recommendations()
    state = RecommenderAdvisorAgentState(
        session_id=99001,
        youth_id="bug_repro_user",
        country_of_user=Country.KENYA,
        conversation_phase=ConversationPhase.PRESENT_RECOMMENDATIONS,
        recommendations=recommendations,
        preference_vector=create_preference_vector(),
        discuss_recommendations=True,
    )

    conversation_history = ConversationHistory()

    for turn_idx, user_msg in enumerate(TRANSCRIPT_TURNS):
        print_turn_header(turn_idx, user_msg)

        if turn_idx == BUG_TURN:
            console.print(
                Panel(
                    "⚠  This is the bug-triggering turn.\n"
                    "Expected: agent should acknowledge the rejection and ask what specifically doesn't work.\n"
                    "Actual:   agent treats 'no resistance' → ACTION_PLANNING → LLM generates farewell.",
                    style="bold yellow",
                    box=box.ROUNDED,
                )
            )

        context = ConversationContext(
            all_history=conversation_history,
            history=conversation_history,
            summary="",
        )

        phase_before = state.conversation_phase
        handler = handlers.get(state.conversation_phase)
        if handler is None:
            console.print(f"[bold red]No handler for phase {state.conversation_phase}. Stopping.[/]")
            break

        t0 = time.time()
        if state.conversation_phase == ConversationPhase.COMPLETE:
            response, llm_stats = await handler.handle_complete(user_msg, state, context)
        else:
            response, llm_stats = await handler.handle(user_msg, state, context)
        elapsed = time.time() - t0
        console.print(f"[dim]LLM call: {elapsed:.2f}s[/]")

        phase_after = state.conversation_phase
        print_agent_response(response, phase_before, phase_after, state)

        # Append to history
        conversation_history.turns.append(
            ConversationTurn(
                index=turn_idx,
                input=AgentInput(message=user_msg, is_artificial=(turn_idx == 0)),
                output=AgentOutput(
                    message_for_user=response.message,
                    finished=response.finished,
                    llm_stats=llm_stats,
                    agent_response_time_in_sec=elapsed if "elapsed" in dir() else 0.0,
                ),
            )
        )

        if response.finished:
            console.print(
                Panel(
                    f"[bold red]Session terminated at turn {turn_idx}[/]\n"
                    f"Phase: {state.conversation_phase.value}\n"
                    f"User message was: \"{user_msg}\"\n\n"
                    "Bug confirmed: conversation ended while user still wanted to continue.",
                    style="bold red",
                    box=box.DOUBLE,
                )
            )
            break

    console.rule("[bold]Reproduction complete[/]")


def main():
    logging.basicConfig(level=logging.WARNING)
    # Suppress noisy loggers from LLM libraries
    for noisy in ("httpx", "httpcore", "google", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    asyncio.run(run_reproduction())


if __name__ == "__main__":
    main()
