#!/usr/bin/env python3
"""
Interactive test script for the Preference Elicitation Agent.

This script allows you to test all components of the preference agent
in an interactive mode without needing full backend integration.

Conversation flow (current):
    INTRO → EXPERIENCE_QUESTIONS → VIGNETTES → [FOLLOW_UP] → GATE (3 clarifying questions) → BWS (12 tasks) → WRAPUP → COMPLETE

Usage:
    poetry run python scripts/test_preference_agent_interactive.py
"""

import asyncio
import sys
import logging
import time
from pathlib import Path
from datetime import timedelta
from typing import List, Optional
import pytest

# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.logging import RichHandler
from rich.traceback import install
from rich import box
from rich.text import Text
from rich.rule import Rule

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.preference_elicitation_agent.agent import PreferenceElicitationAgent
from app.agent.preference_elicitation_agent.state import PreferenceElicitationAgentState
from app.agent.preference_elicitation_agent.vignette_engine import VignetteEngine
from app.agent.preference_elicitation_agent.types import PreferenceVector
from app.agent.agent_types import AgentInput, AgentOutput
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
    ConversationTurn
)
from app.agent.experience.experience_entity import ExperienceEntity
from app.agent.experience import WorkType, Timeline

# Install rich traceback handler
install(show_locals=True)

# Initialize console
console = Console()

# Phase display metadata
PHASE_META = {
    "INTRO":               ("🌱", "blue",    "Introduction"),
    "EXPERIENCE_QUESTIONS":("💼", "cyan",    "Experience Questions"),
    "VIGNETTES":           ("🎭", "green",   "Vignette Scenarios"),
    "FOLLOW_UP":           ("🔍", "yellow",  "Follow-Up Probe"),
    "GATE":                ("🔑", "magenta", "GATE Clarification (max 3)"),
    "BWS":                 ("⚖️",  "orange",  "Best-Worst Scaling (12 tasks)"),
    "WRAPUP":              ("📋", "green",   "Wrap-Up Summary"),
    "COMPLETE":            ("✅", "bright_green", "Complete"),
}


class SessionStats:
    """Track session statistics."""
    def __init__(self):
        self.start_time = time.time()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency = 0.0
        self.turns = 0
        self.phase_history: list[str] = []

        # Estimated costs (~$0.10/1M input, $0.40/1M output for flash-tier)
        self.input_cost_per_1m = 0.10
        self.output_cost_per_1m = 0.40

    def add_turn(self, input_tokens: int, output_tokens: int, latency: float, phase: str):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency += latency
        self.turns += 1
        if not self.phase_history or self.phase_history[-1] != phase:
            self.phase_history.append(phase)

    @property
    def duration(self) -> float:
        return time.time() - self.start_time

    @property
    def estimated_cost(self) -> float:
        input_cost = (self.total_input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_cost_per_1m
        return input_cost + output_cost

    def get_summary_table(self) -> Table:
        table = Table(title="Session Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Total Duration", f"{timedelta(seconds=int(self.duration))}")
        table.add_row("Total Turns", str(self.turns))
        table.add_row("Total Latency (LLM)", f"{self.total_latency:.2f}s")
        table.add_row("Avg Latency/Turn", f"{self.total_latency/self.turns:.2f}s" if self.turns > 0 else "0s")
        table.add_row("Total Input Tokens", f"{self.total_input_tokens:,}")
        table.add_row("Total Output Tokens", f"{self.total_output_tokens:,}")
        table.add_row("Total Tokens", f"{self.total_input_tokens + self.total_output_tokens:,}")
        table.add_row("Estimated Cost", f"${self.estimated_cost:.6f}")
        table.add_row("Phases Visited", " → ".join(self.phase_history))

        return table


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level=logging.INFO):
    """Configure logging with RichHandler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
    )
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_header(text: str):
    console.print(Panel(Text(text, justify="center", style="bold magenta"), box=box.DOUBLE))


def print_section(text: str):
    console.print(f"\n[bold blue]{text}[/]")
    console.print(f"[blue]{'-'*len(text)}[/]")


def print_agent(text: str):
    console.print(Panel(Markdown(text), title="[bold green]Agent[/]", border_style="green", box=box.ROUNDED))


def print_user(text: str):
    console.print(Panel(Text(text), title="[bold cyan]You[/]", border_style="cyan", box=box.ROUNDED))


def print_info(text: str):
    console.print(f"[bold yellow]ℹ {text}[/]")


def print_error(text: str):
    console.print(f"[bold red]✗ {text}[/]")


def print_success(text: str):
    console.print(f"[bold green]✓ {text}[/]")


def get_user_input(prompt: str = "") -> str:
    return console.input(f"[bold cyan]{prompt}[/]")


def display_menu(options: List[str]) -> int:
    table = Table(show_header=False, box=box.SIMPLE)
    for i, option in enumerate(options, 1):
        table.add_row(f"[bold blue]{i}.[/]", option)
    console.print(table)
    while True:
        try:
            choice = int(console.input("\n[bold]Select option (number): [/]"))
            if 1 <= choice <= len(options):
                return choice
            print_error(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print_error("Please enter a valid number")


def print_phase_banner(phase: str):
    """Print a visual banner when transitioning to a new phase."""
    emoji, color, label = PHASE_META.get(phase, ("📌", "white", phase))
    console.print(Rule(f"[bold {color}]{emoji}  {label}  {emoji}[/]", style=color))


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

def create_sample_experiences() -> List[ExperienceEntity]:
    """Create sample experiences for testing."""
    return [
        ExperienceEntity(
            uuid="exp-1",
            experience_title="High School Teacher (Mathematics & Physics)",
            company="Alliance High School",
            location="Kikuyu",
            timeline=Timeline(start="2018", end="2023"),
            work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT
        ),
        ExperienceEntity(
            uuid="exp-2",
            experience_title="Private Tutor",
            company="Self-employed",
            location="Nairobi",
            timeline=Timeline(start="2023", end="Present"),
            work_type=WorkType.SELF_EMPLOYMENT
        ),
        ExperienceEntity(
            uuid="exp-3",
            experience_title="Student Teacher",
            company="Mang'u High School",
            location="Thika",
            timeline=Timeline(start="2017", end="2017"),
            work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT
        )
    ]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_preference_vector(pv: PreferenceVector):
    """Display preference vector in a rich table (7-dimensional)."""
    table = Table(title="Preference Vector (7 Dimensions)", box=box.ROUNDED, show_lines=True)
    table.add_column("Dimension", style="cyan", no_wrap=True)
    table.add_column("Score", style="yellow", justify="center")
    table.add_column("Level", style="white")

    def interpret(value: float) -> str:
        if value >= 0.7:  return "[bold green]HIGH[/]"
        elif value >= 0.5: return "[yellow]MODERATE[/]"
        elif value >= 0.3: return "[dim]LOW[/]"
        else:              return "[dim red]VERY LOW[/]"

    rows = [
        ("1. Financial Compensation",  pv.financial_importance,         "Salary, benefits, compensation"),
        ("2. Work Environment",        pv.work_environment_importance,   "Remote, commute, autonomy, pace"),
        ("3. Career Advancement",      pv.career_advancement_importance, "Growth, learning, promotion"),
        ("4. Work-Life Balance",       pv.work_life_balance_importance,  "Hours, flexibility, family time"),
        ("5. Job Security",            pv.job_security_importance,       "Stability, contract type, risk"),
        ("6. Task Preferences",        pv.task_preference_importance,    "Routine, cognitive, manual, social"),
        ("7. Social Impact",           pv.social_impact_importance,      "Purpose, helping others, community"),
    ]
    for label, val, desc in rows:
        table.add_row(label, f"{val:.2f}", f"{interpret(val)} — {desc}")

    table.add_row("", "", "")
    table.add_row(
        "Overall Confidence",
        f"[bold]{pv.confidence_score:.2f}[/]",
        f"{pv.n_vignettes_completed} vignettes completed"
    )
    console.print(table)


def display_state_info(state: PreferenceElicitationAgentState):
    """Display current agent state information."""
    emoji, color, label = PHASE_META.get(state.conversation_phase, ("📌", "white", state.conversation_phase))
    table = Table(title="Agent State", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Session ID", str(state.session_id))
    table.add_row("Phase", f"[bold {color}]{emoji} {label}[/]")
    table.add_row("Turn Count", str(state.conversation_turn_count))

    # Vignette progress
    completed = len(state.completed_vignettes)
    total = completed + len(state.categories_to_explore)
    table.add_row("Vignettes Progress", f"{completed}/{total}")
    table.add_row("Categories Covered",   ", ".join(state.categories_covered) if state.categories_covered else "None")
    table.add_row("Categories Remaining", ", ".join(state.categories_to_explore) if state.categories_to_explore else "All done")

    # GATE progress
    gate_done = state.gate_interventions_completed
    gate_status = "[green]✓ Complete[/]" if state.gate_complete else f"{gate_done}/3 asked"
    table.add_row("GATE Progress", gate_status)

    # BWS progress
    bws_status = "[green]✓ Complete[/]" if state.bws_phase_complete else f"{state.bws_tasks_completed}/12"
    table.add_row("BWS Progress", bws_status)
    if state.bws_phase_complete and state.top_10_occupations:
        table.add_row("Top Occupations", f"{len(state.top_10_occupations)} ranked")

    console.print(table)

    # Vignette response history
    if state.vignette_responses:
        history_table = Table(title="Vignette Responses", box=box.SIMPLE)
        history_table.add_column("#",          style="dim")
        history_table.add_column("ID",         style="cyan")
        history_table.add_column("Option",     style="yellow")
        history_table.add_column("Confidence", style="green")

        for i, resp in enumerate(state.vignette_responses, 1):
            history_table.add_row(
                str(i),
                resp.vignette_id,
                resp.chosen_option_id or "—",
                f"{resp.confidence:.2f}"
            )
        console.print(history_table)


def display_bws_task(metadata: dict) -> str:
    """
    Display a BWS task with rich UI and collect user selections.

    Returns JSON response string: {"type": "bws_response", "best": "...", "worst": "..."}
    """
    import json

    task_num   = metadata.get("task_number", 0)
    total      = metadata.get("total_tasks", 12)
    occupations = metadata.get("occupations", [])

    table = Table(
        title=f"[bold]BWS Task {task_num} of {total}[/]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Option", style="bold yellow", width=8)
    table.add_column("Job Type", style="bold white", width=40)
    table.add_column("Examples", style="dim", width=50)

    occupation_map = {}
    for i, occ in enumerate(occupations):
        letter = chr(65 + i)  # A-E
        occupation_map[letter] = occ["code"]
        label = occ["label"].title() if occ["label"].isupper() else occ["label"]
        table.add_row(letter, label, occ.get("description", ""))

    console.print(table)
    console.print("\n[bold]Select your preferences:[/]")

    while True:
        most = console.input("[bold green]Which would you MOST like to do? (A-E): [/]").strip().upper()
        if most in occupation_map:
            break
        print_error("Please enter a letter between A and E")

    while True:
        least = console.input("[bold red]Which would you LEAST like to do? (A-E): [/]").strip().upper()
        if least in occupation_map:
            if least != most:
                break
            print_error("You cannot select the same option for both MOST and LEAST")
        else:
            print_error("Please enter a letter between A and E")

    most_label  = next(o["label"] for o in occupations if o["code"] == occupation_map[most])
    least_label = next(o["label"] for o in occupations if o["code"] == occupation_map[least])
    most_label  = most_label.title()  if most_label.isupper()  else most_label
    least_label = least_label.title() if least_label.isupper() else least_label

    print_success(f"Most preferred:  {most} — {most_label}")
    print_success(f"Least preferred: {least} — {least_label}")

    return json.dumps({"type": "bws_response", "best": occupation_map[most], "worst": occupation_map[least]})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vignette_engine():
    """Test the vignette engine component."""
    print_header("Testing Vignette Engine")
    try:
        engine = VignetteEngine()
        print_success(f"Loaded {engine.get_total_vignettes_count()} vignettes")

        print_section("Vignette Categories")
        counts = engine.get_category_counts()
        cat_table = Table(box=box.SIMPLE)
        cat_table.add_column("Category", style="bold cyan")
        cat_table.add_column("Count", style="green")
        for category, count in counts.items():
            cat_table.add_row(category, f"{count} vignette(s)")
        console.print(cat_table)

        print_section("Browse Vignettes")
        for category, vignettes in engine._vignettes_by_category.items():
            console.print(f"\n[bold underline]{category.upper()}[/]")
            vp_table = Table(box=box.SIMPLE, show_header=False)
            vp_table.add_column("Details")
            for v in vignettes:
                details = (
                    f"[bold]{v.vignette_id}[/] (Diff: {v.difficulty_level})\n"
                    f"{v.scenario_text[:100]}...\n"
                    f"[dim]{len(v.options)} options[/]"
                )
                vp_table.add_row(details)
            console.print(vp_table)

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing vignette engine: {e}")
        import traceback
        traceback.print_exc()


@pytest.mark.asyncio
async def test_preference_vector():
    """Test preference vector creation and manipulation."""
    print_header("Testing Preference Vector")
    try:
        pv = PreferenceVector()
        print_success("Created default preference vector")
        display_preference_vector(pv)

        print_section("Modifying Preferences")
        pv.financial_importance         = 0.8
        pv.work_environment_importance  = 0.6
        pv.job_security_importance      = 0.7
        pv.social_impact_importance     = 0.9
        pv.confidence_score             = 0.65
        pv.n_vignettes_completed        = 5
        print_info("Updated financial, work environment, job security, and social impact importance")
        display_preference_vector(pv)

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing preference vector: {e}")
        import traceback
        traceback.print_exc()


@pytest.mark.asyncio
async def test_full_conversation(use_hybrid_mode: bool = False):
    """
    Test a full conversation with the agent.

    Conversation flow:
        INTRO → EXPERIENCE_QUESTIONS → VIGNETTES → [FOLLOW_UP] → GATE (3 questions) → BWS (12 tasks) → WRAPUP

    Commands during conversation:
        quit        — exit the session
        state       — show current agent state
        preferences — show current preference vector
        logs        — (hybrid mode only) show personalization logs
    """
    title = "Interactive Conversation Test — HYBRID MODE" if use_hybrid_mode else "Interactive Conversation Test"
    print_header(title)

    print_info("Commands: 'quit' | 'state' | 'preferences'" + (" | 'logs'" if use_hybrid_mode else ""))
    print_info("Flow: EXPERIENCE_QUESTIONS → VIGNETTES → GATE (3 free-form clarifications) → BWS (12 tasks) → WRAPUP\n")

    session_stats = SessionStats()
    last_phase: Optional[str] = None

    try:
        # Create agent
        if use_hybrid_mode:
            offline_output_dir = str(Path(__file__).parent.parent / "offline_output")
            agent = PreferenceElicitationAgent(
                use_offline_with_personalization=True,
                offline_output_dir=offline_output_dir
            )
            print_success("Created agent in HYBRID mode")
        else:
            agent = PreferenceElicitationAgent()
            print_success("Created preference elicitation agent")

        # Show sample experiences
        sample_experiences = create_sample_experiences()
        exp_table = Table(title="Sample Experiences", box=box.SIMPLE)
        exp_table.add_column("Title", style="bold")
        exp_table.add_column("Company")
        exp_table.add_column("Timeline")
        for exp in sample_experiences:
            start = exp.timeline.start if exp.timeline else "N/A"
            end   = exp.timeline.end   if exp.timeline else "N/A"
            exp_table.add_row(exp.experience_title, exp.company, f"{start} – {end}")
        console.print(exp_table)

        # Initialize state
        if use_hybrid_mode:
            import numpy as np
            state = PreferenceElicitationAgentState(
                session_id=12345,
                initial_experiences_snapshot=sample_experiences,
                use_db6_for_fresh_data=False,
                use_adaptive_selection=False,
                posterior_mean=np.zeros(7).tolist(),
                posterior_covariance=np.eye(7).tolist(),
                fisher_information_matrix=np.zeros((7, 7)).tolist()
            )
            print_info("Initialized Bayesian posterior (prior: μ=0, Σ=I)")
        else:
            state = PreferenceElicitationAgentState(
                session_id=12345,
                initial_experiences_snapshot=sample_experiences,
                use_db6_for_fresh_data=False
            )

        agent.set_state(state)
        print_success("Agent state initialized with sample experiences")
        conversation_history = ConversationHistory()

        # ---------------------------------------------------------------
        # Conversation loop
        # ---------------------------------------------------------------
        turn_index = 0
        while True:
            # --- Collect user input ---
            if turn_index == 0:
                user_message = ""
                print_info("Starting conversation (first turn is automatic)...")
            else:
                raw = get_user_input("You: ")

                if raw.lower() == "quit":
                    print_info("Ending conversation...")
                    break
                elif raw.lower() == "state":
                    display_state_info(state)
                    continue
                elif raw.lower() == "preferences":
                    display_preference_vector(state.preference_vector)
                    continue
                elif raw.lower() == "logs" and use_hybrid_mode:
                    if agent._personalization_logs:
                        log_table = Table(title="Personalization Logs", box=box.ROUNDED, show_lines=True)
                        log_table.add_column("#",                   style="dim")
                        log_table.add_column("Vignette ID",         style="cyan")
                        log_table.add_column("Success",             style="green")
                        log_table.add_column("Attributes Preserved",style="yellow")
                        for i, log in enumerate(agent._personalization_logs, 1):
                            log_table.add_row(
                                str(i), log.vignette_id,
                                "✓" if log.personalization_successful else "✗",
                                "✓" if log.attributes_preserved else "✗"
                            )
                        console.print(log_table)
                    else:
                        print_info("No personalization logs yet")
                    continue

                user_message = raw
                print_user(user_message)

            # --- Execute agent ---
            agent_input = AgentInput(message=user_message, is_artificial=(turn_index == 0))
            context = ConversationContext(
                all_history=conversation_history,
                history=conversation_history,
                summary=""
            )

            try:
                with console.status("[bold green]Agent is thinking...", spinner="dots"):
                    output = await agent.execute(agent_input, context)

                current_phase = state.conversation_phase

                # Show phase transition banner
                if current_phase != last_phase:
                    print_phase_banner(current_phase)
                    last_phase = current_phase

                # --- GATE phase: show GATE-specific progress ---
                if current_phase == "GATE":
                    gate_q = state.gate_interventions_completed
                    gate_label = f"[magenta]GATE Q{gate_q}/3[/]"
                    console.print(f"  {gate_label}  Respond naturally — the agent is probing your preferences.\n")
                    print_agent(output.message_for_user)

                # --- BWS phase: structured selection UI ---
                elif current_phase == "BWS" and not state.bws_phase_complete:
                    # Show the intro/question text first
                    print_agent(output.message_for_user)

                    # Build BWS metadata from state (AgentOutput doesn't carry it)
                    from app.agent.preference_elicitation_agent import bws_utils
                    tasks = bws_utils.load_bws_tasks()
                    current_task_idx = state.bws_tasks_completed - 1  # already incremented

                    if 0 <= current_task_idx < len(tasks):
                        current_task    = tasks[current_task_idx]
                        occupation_groups = bws_utils.load_occupation_groups()
                        occ_map         = {o["code"]: o for o in occupation_groups}

                        occupations_metadata = [
                            {
                                "code":        code,
                                "label":       occ_map.get(code, {}).get("label", f"Occupation {code}"),
                                "description": occ_map.get(code, {}).get("description", ""),
                            }
                            for code in current_task["occupations"]
                        ]
                        metadata = {
                            "interaction_type": "bws_task",
                            "task_number":      current_task_idx + 1,
                            "total_tasks":      len(tasks),
                            "occupations":      occupations_metadata,
                        }

                        bws_response = display_bws_task(metadata)
                        bws_input    = AgentInput(message=bws_response, is_artificial=False)

                        with console.status("[bold green]Agent is thinking...", spinner="dots"):
                            output = await agent.execute(bws_input, context)

                        # Update turn tracking for BWS sub-turn
                        input_tokens  = sum(s.prompt_token_count for s in output.llm_stats)
                        output_tokens = sum(s.response_token_count for s in output.llm_stats)
                        session_stats.add_turn(input_tokens, output_tokens, output.agent_response_time_in_sec, state.conversation_phase)
                        turn_index += 1
                        conversation_history.turns.append(ConversationTurn(index=turn_index, input=bws_input, output=output))

                    print_agent(output.message_for_user)

                else:
                    print_agent(output.message_for_user)

                # --- Stats bar ---
                input_tokens  = sum(s.prompt_token_count for s in output.llm_stats)
                output_tokens = sum(s.response_token_count for s in output.llm_stats)
                latency       = output.agent_response_time_in_sec
                session_stats.add_turn(input_tokens, output_tokens, latency, current_phase)

                emoji, color, label = PHASE_META.get(current_phase, ("📌", "white", current_phase))
                stats_text = (
                    f"Phase: [{color}]{emoji} {label}[/] | "
                    f"Turn: [bold]{state.conversation_turn_count}[/] | "
                    f"Latency: [bold]{latency:.2f}s[/] | "
                    f"Tokens: [green]{input_tokens:,}[/] in / [yellow]{output_tokens:,}[/] out"
                )
                console.print(Panel(stats_text, style="dim", box=box.ROUNDED))

                # Update conversation history
                conversation_history.turns.append(
                    ConversationTurn(index=turn_index, input=agent_input, output=output)
                )
                turn_index += 1

                if output.finished:
                    print_success("Conversation complete!")
                    print_section("Final Results")
                    display_state_info(state)
                    display_preference_vector(state.preference_vector)
                    break

            except Exception as e:
                print_error(f"Error during conversation: {e}")
                import traceback
                traceback.print_exc()
                break

        # End-of-session summary
        print_header("Session Completed")
        console.print(session_stats.get_summary_table())
        console.print(
            "\n[dim italic]Note: Includes conversation LLM + preference extraction LLM + GATE LLM + summary LLM.[/]"
        )
        console.print("[dim italic]Pre-warming LLM calls (background tasks) are not included.[/]")

    except Exception as e:
        print_error(f"Error setting up conversation test: {e}")
        import traceback
        traceback.print_exc()


@pytest.mark.asyncio
async def test_vignette_selection():
    """Test vignette selection logic."""
    print_header("Testing Vignette Selection Logic")
    try:
        engine = VignetteEngine()
        state  = PreferenceElicitationAgentState(session_id=123)
        print_info("Simulating vignette selection across multiple turns...")

        table = Table(box=box.ROUNDED)
        table.add_column("Turn",             style="dim")
        table.add_column("Selected Vignette",style="green")
        table.add_column("Category")
        table.add_column("Progress")

        for turn in range(1, 8):
            vignette = engine.select_next_vignette(state)
            if vignette is None:
                print_info("No more vignettes available")
                break
            state.completed_vignettes.append(vignette.vignette_id)
            if turn % 2 == 0:
                state.mark_category_covered(vignette.category)
            progress = f"{len(state.completed_vignettes)} completed"
            if state.categories_covered:
                progress += f"\nCovered: {', '.join(state.categories_covered)}"
            table.add_row(
                str(turn),
                f"{vignette.vignette_id}\n[dim]{vignette.scenario_text[:50]}...[/]",
                vignette.category,
                progress
            )
        console.print(table)
        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing vignette selection: {e}")
        import traceback
        traceback.print_exc()


@pytest.mark.asyncio
async def test_state_management():
    """Test state management and persistence."""
    print_header("Testing State Management")
    try:
        print_section("Creating State")
        state = PreferenceElicitationAgentState(
            session_id=999,
            conversation_phase="VIGNETTES",
            conversation_turn_count=5,
            completed_vignettes=["financial_001", "remote_commute_001"],
            categories_covered=["financial"],
            categories_to_explore=["work_environment", "job_security"]
        )
        print_success("Created state")
        display_state_info(state)

        print_section("Testing State Methods")
        log_table = Table(show_header=False, box=box.SIMPLE)

        initial_count = state.conversation_turn_count
        state.increment_turn_count()
        log_table.add_row("Increment Turn", f"{initial_count} -> {state.conversation_turn_count}")

        state.mark_category_covered("work_environment")
        log_table.add_row("Mark Covered", "Added 'work_environment'")

        next_cat = state.get_next_category_to_explore()
        log_table.add_row("Next Category", next_cat or "None")

        can_complete = state.can_complete()
        log_table.add_row("Can Complete", str(can_complete))

        # GATE fields
        log_table.add_row("GATE interventions", str(state.gate_interventions_completed))
        log_table.add_row("GATE complete",       str(state.gate_complete))
        console.print(log_table)

        print_section("Testing Serialization")
        state_dict     = state.model_dump()
        restored_state = PreferenceElicitationAgentState(**state_dict)
        print_success("Serialization round-trip OK")
        if state.session_id == restored_state.session_id:
            print_success(f"Session ID MATCH: {state.session_id}")
        else:
            print_error(f"Session ID MISMATCH: {state.session_id} != {restored_state.session_id}")

        # Verify GATE fields survive round-trip
        assert restored_state.gate_interventions_completed == state.gate_interventions_completed
        assert restored_state.gate_complete == state.gate_complete
        print_success("GATE fields survive serialization round-trip ✓")

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing state management: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

async def main_menu():
    print_header("Preference Elicitation Agent — Interactive Test")
    console.print(
        Panel(
            "Explore all components of the preference elicitation agent.\n\n"
            "[dim]Flow: EXPERIENCE_QUESTIONS → VIGNETTES → GATE (3 clarifying Qs) → BWS (12 tasks) → WRAPUP[/]",
            style="blue", box=box.ROUNDED
        )
    )

    print_section("Logging Configuration")
    log_choice = display_menu([
        "INFO — Standard output (recommended)",
        "DEBUG — Detailed debug output",
        "WARNING — Only warnings and errors"
    ])
    log_levels = {1: logging.INFO, 2: logging.DEBUG, 3: logging.WARNING}
    setup_logging(log_levels[log_choice])
    print_success(f"Logging set to {logging.getLevelName(log_levels[log_choice])}")

    while True:
        print_section("Main Menu")
        choice = display_menu([
            "Test Vignette Engine (browse vignettes & categories)",
            "Test Preference Vector (create, modify, view)",
            "Test State Management (state methods, serialization, GATE fields)",
            "Test Vignette Selection Logic",
            "Full Interactive Conversation (standard mode)",
            "Full Interactive Conversation (HYBRID mode — offline + personalization)",
            "Run All Component Tests",
            "Change Logging Level",
            "Exit"
        ])

        if choice == 1:
            await test_vignette_engine()
        elif choice == 2:
            await test_preference_vector()
        elif choice == 3:
            await test_state_management()
        elif choice == 4:
            await test_vignette_selection()
        elif choice == 5:
            await test_full_conversation(use_hybrid_mode=False)
        elif choice == 6:
            await test_full_conversation(use_hybrid_mode=True)
        elif choice == 7:
            print_info("Running all component tests...")
            await test_vignette_engine()
            await test_preference_vector()
            await test_state_management()
            await test_vignette_selection()
            print_success("All component tests complete!")
            print_info("Skipping full conversation test — run manually from menu")
            console.input("\n[dim]Press Enter to return to menu...[/]")
        elif choice == 8:
            print_section("Change Logging Level")
            new_choice = display_menu([
                "INFO — Standard output",
                "DEBUG — Detailed debug output",
                "WARNING — Only warnings and errors"
            ])
            setup_logging(log_levels[new_choice])
            print_success(f"Logging changed to {logging.getLevelName(log_levels[new_choice])}")
        elif choice == 9:
            print_info("Exiting...")
            break


async def main():
    try:
        await main_menu()
    except KeyboardInterrupt:
        print_info("\nInterrupted by user. Exiting...")
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
