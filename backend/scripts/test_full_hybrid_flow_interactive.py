#!/usr/bin/env python3
"""
Full Hybrid Preference Elicitation — Interactive Test.

Combines everything into one script:
  • Full agent conversation flow
      INTRO → EXPERIENCE_QUESTIONS → VIGNETTES → [FOLLOW_UP] → GATE (3 Qs) → BWS (8 tasks) → WRAPUP
  • Vignette personalization
      Offline D-optimal vignettes rewritten by LLM for the user's background
  • Bayesian math panels shown live after every vignette
      — Posterior distribution (mean, std, 95% CI per dimension)
      — Fisher Information Matrix (det, D-efficiency, eigenvalues)
      — Stopping criterion check (continue / stop + reason)
  • Personalization log after each vignette (success flag, attributes preserved)

Usage:
    poetry run python scripts/test_full_hybrid_flow_interactive.py
"""

import asyncio
import sys
import logging
import time
import json
from pathlib import Path
from datetime import timedelta
from typing import List, Optional

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
from rich.columns import Columns

# Add backend root to path  (script lives in backend/scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Agent
from app.agent.preference_elicitation_agent.agent import PreferenceElicitationAgent
from app.agent.preference_elicitation_agent.state import PreferenceElicitationAgentState
from app.agent.preference_elicitation_agent.vignette_engine import VignetteEngine
from app.agent.preference_elicitation_agent.types import PreferenceVector
from app.agent.agent_types import AgentInput, AgentOutput
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
    ConversationTurn,
)
from app.agent.experience.experience_entity import ExperienceEntity
from app.agent.experience import WorkType, Timeline

# Bayesian math (optional — only shown when adaptive mode is active)
try:
    import numpy as np
    from app.agent.preference_elicitation_agent.bayesian.posterior_manager import (
        PosteriorDistribution,
        PosteriorManager,
    )
    from app.agent.preference_elicitation_agent.information_theory.fisher_information import (
        FisherInformationCalculator,
    )
    from app.agent.preference_elicitation_agent.information_theory.stopping_criterion import (
        StoppingCriterion,
    )
    from app.agent.preference_elicitation_agent.config.adaptive_config import AdaptiveConfig
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# BWS helper
from app.agent.preference_elicitation_agent import bws_utils

# ─────────────────────────────────────────────────────────────────────────────
# Console + global display config
# ─────────────────────────────────────────────────────────────────────────────

install(show_locals=True)
console = Console()

PHASE_META = {
    "INTRO":                ("🌱", "blue",         "Introduction"),
    "EXPERIENCE_QUESTIONS": ("💼", "cyan",         "Experience Questions"),
    "VIGNETTES":            ("🎭", "green",        "Vignette Scenarios"),
    "FOLLOW_UP":            ("🔍", "yellow",       "Follow-Up Probe"),
    "GATE":                 ("🔑", "magenta",      "GATE Clarification (3 Qs)"),
    "BWS":                  ("⚖️",  "orange1",      "Best-Worst Scaling (8 tasks)"),
    "WRAPUP":               ("📋", "green",        "Wrap-Up Summary"),
    "COMPLETE":             ("✅", "bright_green", "Complete"),
}

BAYESIAN_DIMS = [
    "financial_importance",
    "work_environment_importance",
    "career_growth_importance",
    "work_life_balance_importance",
    "job_security_importance",
    "task_preference_importance",
    "values_culture_importance",
]

DIM_LABELS = {
    "financial_importance":        "Financial Compensation",
    "work_environment_importance": "Work Environment",
    "career_growth_importance":    "Career Growth",
    "work_life_balance_importance":"Work-Life Balance",
    "job_security_importance":     "Job Security",
    "task_preference_importance":  "Task Preferences",
    "values_culture_importance":   "Values & Culture",
}


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    for noisy in ("httpx", "httpcore", "google", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Session stats
# ─────────────────────────────────────────────────────────────────────────────

class SessionStats:
    """Track token usage, latency, and phase journey."""

    def __init__(self):
        self.start_time = time.time()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency = 0.0
        self.turns = 0
        self.phase_history: list[str] = []
        self.vignettes_personalized = 0
        self.vignettes_total = 0

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

    def get_summary_table(self) -> Table:
        t = Table(title="Session Summary", box=box.ROUNDED)
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="yellow")
        t.add_row("Total Duration",    str(timedelta(seconds=int(self.duration))))
        t.add_row("Total Turns",       str(self.turns))
        t.add_row("LLM Latency Total", f"{self.total_latency:.2f}s")
        t.add_row("Avg Latency/Turn",  f"{self.total_latency/self.turns:.2f}s" if self.turns else "—")
        t.add_row("Input Tokens",      f"{self.total_input_tokens:,}")
        t.add_row("Output Tokens",     f"{self.total_output_tokens:,}")
        t.add_row("Vignettes (total)", str(self.vignettes_total))
        t.add_row("Personalized",      str(self.vignettes_personalized))
        t.add_row("Phases Visited",    " → ".join(self.phase_history))
        return t


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_header(text: str):
    console.print(Panel(Text(text, justify="center", style="bold magenta"), box=box.DOUBLE))


def print_section(text: str):
    console.print(f"\n[bold blue]{text}[/]")
    console.print(f"[blue]{'-' * len(text)}[/]")


def print_agent(text: str):
    console.print(Panel(Markdown(text), title="[bold green]Agent[/]", border_style="green", box=box.ROUNDED))


def print_user(text: str):
    console.print(Panel(Text(text), title="[bold cyan]You[/]", border_style="cyan", box=box.ROUNDED))


def print_info(text: str):
    console.print(f"[bold yellow]ℹ  {text}[/]")


def print_error(text: str):
    console.print(f"[bold red]✗  {text}[/]")


def print_success(text: str):
    console.print(f"[bold green]✓  {text}[/]")


def get_user_input(prompt: str = "") -> str:
    return console.input(f"[bold cyan]{prompt}[/]")


def display_menu(options: List[str]) -> int:
    t = Table(show_header=False, box=box.SIMPLE)
    for i, o in enumerate(options, 1):
        t.add_row(f"[bold blue]{i}.[/]", o)
    console.print(t)
    while True:
        try:
            choice = int(console.input("\n[bold]Select option: [/]"))
            if 1 <= choice <= len(options):
                return choice
            print_error(f"Enter a number between 1 and {len(options)}")
        except ValueError:
            print_error("Enter a valid number")


def print_phase_banner(phase: str):
    emoji, color, label = PHASE_META.get(phase, ("📌", "white", phase))
    console.print(Rule(f"[bold {color}]{emoji}  {label}  {emoji}[/]", style=color))


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian display helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_posterior_from_state(state: PreferenceElicitationAgentState) -> Optional["PosteriorDistribution"]:
    """Reconstruct a PosteriorDistribution from state fields for display."""
    if not BAYESIAN_AVAILABLE:
        return None
    if state.posterior_mean is None or state.posterior_covariance is None:
        return None
    return PosteriorDistribution(
        mean=state.posterior_mean,
        covariance=state.posterior_covariance,
        dimensions=BAYESIAN_DIMS,
    )


def display_posterior_panel(state: PreferenceElicitationAgentState, title: str = "Bayesian Posterior"):
    """Show posterior distribution table from state."""
    posterior = build_posterior_from_state(state)
    if posterior is None:
        print_info("Bayesian posterior not available (adaptive mode off or numpy missing)")
        return

    t = Table(title=title, box=box.ROUNDED, show_lines=True)
    t.add_column("Dimension",  style="cyan",    no_wrap=True)
    t.add_column("Mean (μ)",   style="yellow",  justify="right")
    t.add_column("Std (σ)",    style="magenta", justify="right")
    t.add_column("95% CI",     style="green",   justify="center")

    mean_arr = np.array(posterior.mean)
    cov_arr  = np.array(posterior.covariance)

    for i, dim in enumerate(posterior.dimensions):
        mu  = mean_arr[i]
        var = cov_arr[i, i]
        std = float(np.sqrt(max(var, 0.0)))
        lo  = mu - 1.96 * std
        hi  = mu + 1.96 * std
        label = DIM_LABELS.get(dim, dim.replace("_", " ").title())
        t.add_row(label, f"{mu:+.3f}", f"±{std:.3f}", f"[{lo:+.3f}, {hi:+.3f}]")

    console.print(t)


def display_fim_panel(state: PreferenceElicitationAgentState, n_vignettes: int):
    """Show Fisher Information Matrix statistics from state."""
    if not BAYESIAN_AVAILABLE or state.fisher_information_matrix is None:
        return

    fim = np.array(state.fisher_information_matrix)
    eigenvalues = np.linalg.eigvalsh(fim)
    det = float(np.linalg.det(fim))
    d_eff = float(det ** (1.0 / len(eigenvalues))) if det > 0 else 0.0
    cond = float(eigenvalues.max() / eigenvalues.min()) if eigenvalues.min() > 1e-10 else float("inf")

    t = Table(title=f"Fisher Information Matrix (after {n_vignettes} vignette(s))", box=box.ROUNDED)
    t.add_column("Metric",  style="cyan")
    t.add_column("Value",   style="yellow", justify="right")
    t.add_row("Determinant  det(FIM)", f"{det:.2e}")
    t.add_row("D-Efficiency det^(1/7)", f"{d_eff:.4f}")
    t.add_row("Condition Number",       f"{cond:.2f}")
    t.add_row("Min Eigenvalue",         f"{eigenvalues.min():.4f}")
    t.add_row("Max Eigenvalue",         f"{eigenvalues.max():.4f}")

    eig_t = Table(title="Eigenvalues (info per direction)", box=box.SIMPLE)
    eig_t.add_column("#",   style="dim")
    eig_t.add_column("λ",  style="green", justify="right")
    for i, eig in enumerate(eigenvalues, 1):
        eig_t.add_row(str(i), f"{eig:.4f}")

    console.print(Columns([t, eig_t]))


def display_stopping_panel(state: PreferenceElicitationAgentState):
    """Show stopping criterion status from state."""
    if not BAYESIAN_AVAILABLE or state.fisher_information_matrix is None:
        return

    try:
        cfg = AdaptiveConfig.from_env()
        criterion = StoppingCriterion(
            min_vignettes=cfg.min_vignettes,
            max_vignettes=cfg.max_vignettes,
            det_threshold=cfg.fim_det_threshold,  # AdaptiveConfig.fim_det_threshold → StoppingCriterion.det_threshold
        )
        fim = np.array(state.fisher_information_matrix)
        posterior = build_posterior_from_state(state)
        if posterior is None:
            return

        should_go, reason = criterion.should_continue(
            posterior=posterior,
            fim=fim,
            n_vignettes_shown=len(state.completed_vignettes),
        )

        color  = "green" if should_go else "red"
        decision = "CONTINUE" if should_go else "STOP (adaptive phase done)"

        det_reg = float(np.linalg.det(fim + np.eye(fim.shape[0]) * 1e-8))
        max_var = max(posterior.get_variance(d) for d in posterior.dimensions)

        t = Table(title="Stopping Criterion", box=box.ROUNDED)
        t.add_column("Check",  style="cyan")
        t.add_column("Status", style=color)
        t.add_row("Decision",        f"[bold {color}]{decision}[/]")
        t.add_row("Reason",          reason)
        t.add_row("Vignettes shown", f"{len(state.completed_vignettes)} "
                                     f"(min {criterion.min_vignettes} / max {criterion.max_vignettes})")
        t.add_row("FIM det",         f"{det_reg:.2e}  (threshold: {criterion.det_threshold:.2e})")
        t.add_row("Max variance",    f"{max_var:.4f}  (threshold: {criterion.max_variance_threshold:.4f})")
        console.print(t)

    except Exception as exc:
        print_info(f"Stopping criterion display skipped: {exc}")


def display_bayesian_suite(state: PreferenceElicitationAgentState, vignette_id: str):
    """Display the full Bayesian math block after a vignette completes."""
    n = len(state.completed_vignettes)
    console.print(Rule(f"[dim]Bayesian update — {vignette_id} (vignette {n})[/]", style="dim"))
    display_posterior_panel(state, title=f"Posterior after vignette {n}: {vignette_id}")
    display_fim_panel(state, n)
    display_stopping_panel(state)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Preference vector display
# ─────────────────────────────────────────────────────────────────────────────

def display_preference_vector(pv: PreferenceVector):
    t = Table(title="Preference Vector (7 Dimensions)", box=box.ROUNDED, show_lines=True)
    t.add_column("Dimension",   style="cyan",   no_wrap=True)
    t.add_column("Score",       style="yellow", justify="center")
    t.add_column("Level",       style="white")

    def interp(v: float) -> str:
        if v >= 0.7:  return "[bold green]HIGH[/]"
        if v >= 0.5:  return "[yellow]MODERATE[/]"
        if v >= 0.3:  return "[dim]LOW[/]"
        return "[dim red]VERY LOW[/]"

    rows = [
        ("1. Financial Compensation",  pv.financial_importance),
        ("2. Work Environment",        pv.work_environment_importance),
        ("3. Career Advancement",      pv.career_advancement_importance),
        ("4. Work-Life Balance",       pv.work_life_balance_importance),
        ("5. Job Security",            pv.job_security_importance),
        ("6. Task Preferences",        pv.task_preference_importance),
        ("7. Social Impact",           pv.social_impact_importance),
    ]
    for label, val in rows:
        t.add_row(label, f"{val:.2f}", interp(val))
    t.add_row("", "", "")
    t.add_row("Confidence", f"[bold]{pv.confidence_score:.2f}[/]",
              f"{pv.n_vignettes_completed} vignettes completed")
    console.print(t)


# ─────────────────────────────────────────────────────────────────────────────
# State info display
# ─────────────────────────────────────────────────────────────────────────────

def display_state_info(state: PreferenceElicitationAgentState):
    emoji, color, label = PHASE_META.get(state.conversation_phase, ("📌", "white", state.conversation_phase))
    t = Table(title="Agent State", box=box.ROUNDED)
    t.add_column("Property", style="cyan")
    t.add_column("Value",    style="white")
    t.add_row("Session ID",    str(state.session_id))
    t.add_row("Phase",         f"[bold {color}]{emoji} {label}[/]")
    t.add_row("Turn Count",    str(state.conversation_turn_count))
    t.add_row("Adaptive Mode", "✓ ON" if state.use_adaptive_selection else "✗ OFF")

    completed = len(state.completed_vignettes)
    t.add_row("Vignettes Done",   f"{completed}")
    t.add_row("Categories Covered",   ", ".join(state.categories_covered) or "None")
    t.add_row("Categories Remaining", ", ".join(state.categories_to_explore) or "All done")

    gate_status = "[green]✓ Complete[/]" if state.gate_complete else f"{state.gate_interventions_completed}/3 asked"
    t.add_row("GATE Progress", gate_status)

    bws_status = "[green]✓ Complete[/]" if state.bws_phase_complete else f"{state.bws_tasks_completed}/8"
    t.add_row("BWS Progress", bws_status)

    if BAYESIAN_AVAILABLE and state.fisher_information_matrix is not None:
        fim = np.array(state.fisher_information_matrix)
        det = float(np.linalg.det(fim))
        t.add_row("FIM Determinant", f"{det:.2e}")

    console.print(t)

    if state.vignette_responses:
        ht = Table(title="Vignette Responses", box=box.SIMPLE)
        ht.add_column("#",          style="dim")
        ht.add_column("ID",         style="cyan")
        ht.add_column("Option",     style="yellow")
        ht.add_column("Confidence", style="green", justify="right")
        for i, r in enumerate(state.vignette_responses, 1):
            ht.add_row(str(i), r.vignette_id, r.chosen_option_id or "—", f"{r.confidence:.2f}")
        console.print(ht)


# ─────────────────────────────────────────────────────────────────────────────
# HB display helper
# ─────────────────────────────────────────────────────────────────────────────

def display_hb_scores(state: PreferenceElicitationAgentState):
    """Display HB utility scores table if available."""
    if not state.hb_scores:
        return

    wa_labels = bws_utils.load_wa_labels()
    t = Table(
        title="HB Utility Scores (Bayesian estimates)",
        box=box.ROUNDED,
        show_lines=True,
    )
    t.add_column("#",              style="bold",    width=4,  justify="right")
    t.add_column("WA_Element_ID",  style="dim",     width=16)
    t.add_column("Work Activity",  style="cyan",    width=44)
    t.add_column("Mean",           style="yellow",  width=8,  justify="right")
    t.add_column("SD",             style="magenta", width=6,  justify="right")
    t.add_column("95% CI",         style="green",   width=20, justify="center")

    # Sort by rank
    sorted_items = sorted(state.hb_scores.items(), key=lambda x: x[1]["rank"])
    for wa_id, scores in sorted_items:
        label  = wa_labels.get(wa_id, wa_id)
        mean   = scores["mean"]
        sd     = scores["sd"]
        ci_low = scores["ci_low"]
        ci_hi  = scores["ci_high"]
        rank   = scores["rank"]
        t.add_row(
            str(rank),
            wa_id,
            label,
            f"{mean:+.2f}",
            f"{sd:.2f}",
            f"[{ci_low:+.2f}, {ci_hi:+.2f}]",
        )
    console.print(t)


# ─────────────────────────────────────────────────────────────────────────────
# BWS UI
# ─────────────────────────────────────────────────────────────────────────────

def display_bws_task(metadata: dict) -> str:
    """Render a BWS task and collect best/worst selection. Returns JSON string."""
    task_num = metadata.get("task_number", 0)
    total    = metadata.get("total_tasks", 8)
    items    = metadata.get("items", [])

    t = Table(
        title=f"[bold]BWS Task {task_num} of {total}[/]",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    t.add_column("Option",        style="bold yellow", width=8)
    t.add_column("Work Activity", style="bold white",  width=60)

    item_map: dict[str, str] = {}
    for i, item in enumerate(items):
        letter = chr(65 + i)
        item_map[letter] = item["id"]
        t.add_row(letter, item["label"])
    console.print(t)

    console.print("\n[bold]Select your preferences:[/]")
    while True:
        most = console.input("[bold green]Which would you MOST enjoy? (A-E): [/]").strip().upper()
        if most in item_map:
            break
        print_error("Please enter a letter between A and E")

    while True:
        least = console.input("[bold red]Which would you LEAST enjoy? (A-E): [/]").strip().upper()
        if least in item_map:
            if least != most:
                break
            print_error("You cannot pick the same option for both MOST and LEAST")
        else:
            print_error("Please enter a letter between A and E")

    most_label  = next(o["label"] for o in items if o["id"] == item_map[most])
    least_label = next(o["label"] for o in items if o["id"] == item_map[least])
    print_success(f"Most preferred:  {most} — {most_label}")
    print_success(f"Least preferred: {least} — {least_label}")

    return json.dumps({"type": "bws_response", "best": item_map[most], "worst": item_map[least]})


# ─────────────────────────────────────────────────────────────────────────────
# Sample data
# ─────────────────────────────────────────────────────────────────────────────

def create_sample_experiences() -> List[ExperienceEntity]:
    return [
        ExperienceEntity(
            uuid="exp-1",
            experience_title="High School Teacher (Mathematics & Physics)",
            company="Alliance High School",
            location="Kikuyu",
            timeline=Timeline(start="2018", end="2023"),
            work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT,
        ),
        ExperienceEntity(
            uuid="exp-2",
            experience_title="Software Developer",
            company="Self-employed",
            location="Nairobi",
            timeline=Timeline(start="2023", end="Present"),
            work_type=WorkType.SELF_EMPLOYMENT,
        ),
        ExperienceEntity(
            uuid="exp-3",
            experience_title="Electrician",
            company="KPLC",
            location="Thika",
            timeline=Timeline(start="2017", end="2017"),
            work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Personalization log display
# ─────────────────────────────────────────────────────────────────────────────

def display_personalization_logs(agent: PreferenceElicitationAgent):
    """Show vignette personalization log table."""
    logs = agent._personalization_logs
    if not logs:
        print_info("No personalization logs yet")
        return

    t = Table(title="Vignette Personalization Log", box=box.ROUNDED, show_lines=True)
    t.add_column("#",                    style="dim",    width=4)
    t.add_column("Vignette ID",          style="cyan",   width=24)
    t.add_column("Personalized?",        style="green",  width=14)
    t.add_column("Attributes Preserved", style="yellow", width=20)
    t.add_column("Strategy",             style="white")

    for i, log in enumerate(logs, 1):
        success   = "✓ Yes" if log.personalization_successful else "✗ No"
        preserved = "✓ Yes" if log.attributes_preserved       else "✗ No"
        strategy  = getattr(log, "personalization_strategy", "—")
        t.add_row(str(i), log.vignette_id, success, preserved, strategy)
    console.print(t)


# ─────────────────────────────────────────────────────────────────────────────
# Main full-hybrid conversation
# ─────────────────────────────────────────────────────────────────────────────

async def run_full_hybrid_session(use_adaptive: bool = True):
    """
    Run the complete preference elicitation flow with personalization and Bayesian tracking.

    Args:
        use_adaptive: Whether to enable adaptive D-optimal vignette selection
                      (requires numpy + Bayesian modules)
    """
    title = "Full Hybrid Preference Elicitation"
    if use_adaptive:
        title += " + Adaptive Bayesian"
    print_header(title)

    console.print(Panel(
        "This session runs the [bold]complete[/] preference agent flow.\n\n"
        "• Vignettes are rewritten by LLM to match your background\n"
        + ("• Bayesian posterior + FIM are updated and displayed after each vignette\n"
           "• Stopping criterion checked live after each vignette\n"
           if use_adaptive else "") +
        "• GATE asks 3 targeted clarifying questions after vignettes\n"
        "• BWS ranks 37 work activities across 8 tasks\n\n"
        "[dim]Commands during chat:  quit | state | preferences | logs[/]",
        style="blue", box=box.ROUNDED,
    ))

    if use_adaptive and not BAYESIAN_AVAILABLE:
        print_error("Adaptive mode requested but numpy / Bayesian modules not installed. "
                    "Falling back to non-adaptive.")
        use_adaptive = False

    session_stats = SessionStats()
    last_phase: Optional[str] = None

    try:
        # ── Agent ───────────────────────────────────────────────────────────
        offline_output_dir = str(Path(__file__).parent.parent / "offline_output")
        agent = PreferenceElicitationAgent(
            use_offline_with_personalization=True,
            offline_output_dir=offline_output_dir,
        )
        print_success("Agent created — hybrid mode (offline D-optimal + LLM personalization)")

        # ── Experiences ─────────────────────────────────────────────────────
        sample_experiences = create_sample_experiences()
        exp_t = Table(title="Sample Experiences (used for vignette personalization)", box=box.SIMPLE)
        exp_t.add_column("Title",    style="bold")
        exp_t.add_column("Company")
        exp_t.add_column("Period")
        for exp in sample_experiences:
            start = exp.timeline.start if exp.timeline else "?"
            end   = exp.timeline.end   if exp.timeline else "?"
            exp_t.add_row(exp.experience_title, exp.company, f"{start} – {end}")
        console.print(exp_t)

        # ── State ───────────────────────────────────────────────────────────
        state_kwargs: dict = dict(
            session_id=54321,
            initial_experiences_snapshot=sample_experiences,
            use_db6_for_fresh_data=False,
            use_adaptive_selection=use_adaptive,
        )
        if use_adaptive:
            state_kwargs["posterior_mean"]           = np.zeros(7).tolist()
            state_kwargs["posterior_covariance"]     = np.eye(7).tolist()
            state_kwargs["fisher_information_matrix"] = np.zeros((7, 7)).tolist()

        state = PreferenceElicitationAgentState(**state_kwargs)
        agent.set_state(state)
        print_success("State initialized" + (" (prior: μ=0, Σ=I)" if use_adaptive else ""))

        if use_adaptive:
            print_section("Prior Distribution (Before Any Vignettes)")
            display_posterior_panel(state, "Prior Distribution")

        conversation_history = ConversationHistory()
        prev_vignette_count  = 0   # detect newly completed vignettes
        turn_index = 0

        # ── Conversation loop ────────────────────────────────────────────────
        while True:
            # --- Input ---
            if turn_index == 0:
                user_message = ""
                print_info("Starting conversation (first turn is automatic)...")
            else:
                raw = get_user_input("You: ")

                if raw.lower() == "quit":
                    print_info("Ending session...")
                    break
                elif raw.lower() == "state":
                    display_state_info(state)
                    continue
                elif raw.lower() == "preferences":
                    display_preference_vector(state.preference_vector)
                    continue
                elif raw.lower() == "logs":
                    display_personalization_logs(agent)
                    continue
                elif raw.lower() == "bayes" and use_adaptive:
                    n = len(state.completed_vignettes)
                    if n:
                        display_bayesian_suite(state, f"current ({n} done)")
                    else:
                        print_info("No vignettes completed yet")
                    continue

                user_message = raw
                print_user(user_message)

            # --- Execute agent turn ---
            agent_input = AgentInput(message=user_message, is_artificial=(turn_index == 0))
            context = ConversationContext(
                all_history=conversation_history,
                history=conversation_history,
                summary="",
            )

            try:
                with console.status("[bold green]Agent is thinking...", spinner="dots"):
                    output = await agent.execute(agent_input, context)

                current_phase = state.conversation_phase

                # Phase transition banner
                if current_phase != last_phase:
                    print_phase_banner(current_phase)
                    last_phase = current_phase

                # ── Bayesian panel: show after vignette completions ──────────
                new_vignette_count = len(state.completed_vignettes)
                if use_adaptive and new_vignette_count > prev_vignette_count:
                    # One or more vignettes just completed — show Bayesian math
                    for vid in state.completed_vignettes[prev_vignette_count:]:
                        display_bayesian_suite(state, vid)
                    session_stats.vignettes_personalized += sum(
                        1 for log in agent._personalization_logs
                        if log.personalization_successful
                    ) - session_stats.vignettes_personalized
                    session_stats.vignettes_total = new_vignette_count
                prev_vignette_count = new_vignette_count

                # ── Phase-specific rendering ─────────────────────────────────
                if current_phase == "GATE":
                    gate_q = state.gate_interventions_completed
                    console.print(
                        f"  [magenta]GATE Q{gate_q}/3[/]  "
                        "Respond naturally — the agent is probing preference nuances.\n"
                    )
                    print_agent(output.message_for_user)

                elif current_phase == "BWS" and not state.bws_phase_complete:
                    print_agent(output.message_for_user)

                    # Build BWS task metadata from state
                    tasks = bws_utils.load_wa_tasks()
                    current_task_idx = state.bws_tasks_completed - 1  # already incremented
                    if 0 <= current_task_idx < len(tasks):
                        current_task = tasks[current_task_idx]
                        wa_labels    = bws_utils.load_wa_labels()
                        items_meta   = [
                            {"id": wa_id, "label": wa_labels.get(wa_id, wa_id)}
                            for wa_id in current_task["items"]
                        ]
                        bws_meta = {
                            "interaction_type": "bws_task",
                            "task_number":      current_task_idx + 1,
                            "total_tasks":      len(tasks),
                            "items":            items_meta,
                        }
                        bws_response = display_bws_task(bws_meta)
                        bws_input    = AgentInput(message=bws_response, is_artificial=False)

                        with console.status("[bold green]Recording BWS response...", spinner="dots"):
                            output = await agent.execute(bws_input, context)

                        in_t  = sum(s.prompt_token_count for s in output.llm_stats)
                        out_t = sum(s.response_token_count for s in output.llm_stats)
                        session_stats.add_turn(in_t, out_t, output.agent_response_time_in_sec,
                                               state.conversation_phase)
                        turn_index += 1
                        conversation_history.turns.append(
                            ConversationTurn(index=turn_index, input=bws_input, output=output)
                        )
                    print_agent(output.message_for_user)

                else:
                    print_agent(output.message_for_user)

                # ── Stats bar ────────────────────────────────────────────────
                in_t    = sum(s.prompt_token_count  for s in output.llm_stats)
                out_t   = sum(s.response_token_count for s in output.llm_stats)
                latency = output.agent_response_time_in_sec
                session_stats.add_turn(in_t, out_t, latency, current_phase)

                emoji, color, label = PHASE_META.get(current_phase, ("📌", "white", current_phase))
                console.print(Panel(
                    f"Phase: [{color}]{emoji} {label}[/] | "
                    f"Turn: [bold]{state.conversation_turn_count}[/] | "
                    f"Latency: [bold]{latency:.2f}s[/] | "
                    f"Tokens: [green]{in_t:,}[/] in / [yellow]{out_t:,}[/] out"
                    + (f" | Vignettes: [cyan]{len(state.completed_vignettes)}[/]"
                       if current_phase in ("VIGNETTES", "FOLLOW_UP", "GATE") else ""),
                    style="dim", box=box.ROUNDED,
                ))

                conversation_history.turns.append(
                    ConversationTurn(index=turn_index, input=agent_input, output=output)
                )
                turn_index += 1

                if output.finished:
                    print_success("Conversation complete!")
                    break

            except Exception as exc:
                print_error(f"Error during turn: {exc}")
                import traceback
                traceback.print_exc()
                break

        # ── Final summary ────────────────────────────────────────────────────
        print_header("Session Complete")

        print_section("Final Preference Vector")
        display_preference_vector(state.preference_vector)

        if use_adaptive and len(state.completed_vignettes) > 0:
            print_section("Final Bayesian State")
            display_posterior_panel(state, "Final Posterior Distribution")
            display_fim_panel(state, len(state.completed_vignettes))
            display_stopping_panel(state)

        if agent._personalization_logs:
            print_section("Personalization Log")
            display_personalization_logs(agent)

        if state.hb_scores:
            print_section("HB Utility Scores")
            display_hb_scores(state)

        print_section("Session Stats")
        console.print(session_stats.get_summary_table())
        console.print(
            "\n[dim italic]Tokens include: conversation LLM + preference extractor + "
            "GATE LLM + metadata extractor + vignette personalizer.[/]"
        )

    except Exception as exc:
        print_error(f"Setup failed: {exc}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# BWS-only session (skip INTRO / VIGNETTES / GATE)
# ─────────────────────────────────────────────────────────────────────────────

async def run_bws_only_session():
    """
    Jump straight into the BWS phase — skips INTRO, EXPERIENCE_QUESTIONS,
    VIGNETTES, FOLLOW_UP, and GATE by pre-setting state.
    """
    print_header("BWS Phase — Direct Test")
    console.print(Panel(
        "Skips to BWS immediately.\n\n"
        "• 8 tasks × 5 ONET work activities\n"
        "• Scores keyed by WA_Element_ID\n"
        "• Final top-8 printed at the end\n\n"
        "[dim]Commands: quit | state[/]",
        style="orange1", box=box.ROUNDED,
    ))

    session_stats = SessionStats()

    try:
        agent = PreferenceElicitationAgent(
            use_offline_with_personalization=False,
        )

        state = PreferenceElicitationAgentState(
            session_id=99999,
            use_db6_for_fresh_data=False,
            use_adaptive_selection=False,
            # Skip straight to BWS
            conversation_phase="BWS",
            gate_complete=True,
        )
        agent.set_state(state)
        print_success("State initialized — phase set to BWS, GATE marked complete")

        conversation_history = ConversationHistory()
        turn_index = 0

        while True:
            if turn_index == 0:
                user_message = ""
                print_info("Starting BWS phase (first turn is automatic)...")
            else:
                raw = get_user_input("You: ")
                if raw.lower() == "quit":
                    print_info("Ending session...")
                    break
                elif raw.lower() == "state":
                    display_state_info(state)
                    continue
                user_message = raw
                print_user(user_message)

            agent_input = AgentInput(message=user_message, is_artificial=(turn_index == 0))
            context = ConversationContext(
                all_history=conversation_history,
                history=conversation_history,
                summary="",
            )

            try:
                with console.status("[bold green]Agent is thinking...", spinner="dots"):
                    output = await agent.execute(agent_input, context)

                current_phase = state.conversation_phase
                print_phase_banner(current_phase)

                if current_phase == "BWS" and not state.bws_phase_complete:
                    print_agent(output.message_for_user)

                    tasks = bws_utils.load_wa_tasks()
                    current_task_idx = state.bws_tasks_completed - 1
                    if 0 <= current_task_idx < len(tasks):
                        current_task = tasks[current_task_idx]
                        wa_labels    = bws_utils.load_wa_labels()
                        items_meta   = [
                            {"id": wa_id, "label": wa_labels.get(wa_id, wa_id)}
                            for wa_id in current_task["items"]
                        ]
                        bws_meta = {
                            "interaction_type": "bws_task",
                            "task_number":      current_task_idx + 1,
                            "total_tasks":      len(tasks),
                            "items":            items_meta,
                        }
                        bws_response = display_bws_task(bws_meta)
                        bws_input    = AgentInput(message=bws_response, is_artificial=False)

                        with console.status("[bold green]Recording BWS response...", spinner="dots"):
                            output = await agent.execute(bws_input, context)

                        in_t  = sum(s.prompt_token_count for s in output.llm_stats)
                        out_t = sum(s.response_token_count for s in output.llm_stats)
                        session_stats.add_turn(in_t, out_t, output.agent_response_time_in_sec,
                                               state.conversation_phase)
                        turn_index += 1
                        conversation_history.turns.append(
                            ConversationTurn(index=turn_index, input=bws_input, output=output)
                        )
                else:
                    print_agent(output.message_for_user)

                in_t    = sum(s.prompt_token_count  for s in output.llm_stats)
                out_t   = sum(s.response_token_count for s in output.llm_stats)
                latency = output.agent_response_time_in_sec
                session_stats.add_turn(in_t, out_t, latency, current_phase)

                conversation_history.turns.append(
                    ConversationTurn(index=turn_index, input=agent_input, output=output)
                )
                turn_index += 1

                if output.finished or state.bws_phase_complete:
                    print_success("BWS phase complete!")
                    break

            except Exception as exc:
                print_error(f"Error during turn: {exc}")
                import traceback
                traceback.print_exc()
                break

        # ── Final BWS results ────────────────────────────────────────────────
        print_header("BWS Results")

        if state.bws_scores:
            wa_labels = bws_utils.load_wa_labels()
            t = Table(title="BWS Scores (all items)", box=box.ROUNDED, show_lines=True)
            t.add_column("WA_Element_ID",  style="dim",    width=16)
            t.add_column("Work Activity",  style="cyan",   width=50)
            t.add_column("Score",          style="yellow", justify="right", width=8)
            for wa_id, score in sorted(state.bws_scores.items(), key=lambda x: -x[1]):
                t.add_row(wa_id, wa_labels.get(wa_id, wa_id), f"{score:+.0f}")
            console.print(t)

        if state.top_10_bws:
            wa_labels = bws_utils.load_wa_labels()
            t2 = Table(title="Top 8 Work Activities", box=box.ROUNDED)
            t2.add_column("#",             style="bold",   width=4)
            t2.add_column("WA_Element_ID", style="dim",    width=16)
            t2.add_column("Work Activity", style="cyan",   width=50)
            for rank, wa_id in enumerate(state.top_10_bws, 1):
                t2.add_row(str(rank), wa_id, wa_labels.get(wa_id, wa_id))
            console.print(t2)

        if state.hb_scores:
            print_section("HB Utility Scores")
            display_hb_scores(state)

        print_section("Session Stats")
        console.print(session_stats.get_summary_table())

    except Exception as exc:
        print_error(f"Setup failed: {exc}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Menu
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    print_header("Full Hybrid Preference Flow — Interactive Test")
    console.print(Panel(
        "Runs the complete preference agent with:\n"
        "  • LLM vignette personalization (offline D-optimal + rewriting)\n"
        "  • Live Bayesian math panels after each vignette\n"
        "  • Full GATE + BWS phases\n\n"
        "[dim]Commands during chat:  quit | state | preferences | logs | bayes[/]",
        style="blue", box=box.ROUNDED,
    ))

    print_section("Logging Level")
    log_choice = display_menu([
        "INFO — Standard (recommended)",
        "DEBUG — Verbose debug output",
        "WARNING — Errors only",
    ])
    log_levels = {1: logging.INFO, 2: logging.DEBUG, 3: logging.WARNING}
    setup_logging(log_levels[log_choice])

    print_section("Mode")
    mode_choice = display_menu([
        "Hybrid + Adaptive Bayesian  (personalization + live posterior/FIM panels)",
        "Hybrid only                 (personalization, no Bayesian panels)",
        "BWS only                    (skip straight to BWS phase — fast testing)",
    ])

    try:
        if mode_choice == 3:
            await run_bws_only_session()
        else:
            await run_full_hybrid_session(use_adaptive=(mode_choice == 1))
    except KeyboardInterrupt:
        print_info("\nInterrupted. Exiting.")
    except Exception as exc:
        print_error(f"Unexpected error: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
