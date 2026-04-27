#!/usr/bin/env python3
"""
Focused test for the preference agent experience-title phrasing fix.

Verifies that when experience_title is a place name (e.g. "cooking school")
the agent does NOT produce "working as cooking school" in either:
  - The deterministic fallback (turn 1, no LLM call)
  - The LLM prompt (turn 2, one real LLM call — checks the guard instruction works)

Usage:
    poetry run python scripts/test_preference_agent_title_phrasing.py
"""

import asyncio
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.preference_elicitation_agent.agent import PreferenceElicitationAgent
from app.agent.preference_elicitation_agent.state import PreferenceElicitationAgentState
from app.agent.agent_types import AgentInput
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
    ConversationTurn,
)
from app.agent.experience.experience_entity import ExperienceEntity
from app.agent.experience import WorkType, Timeline

logging.basicConfig(level=logging.WARNING)  # suppress agent internals


BAD_PHRASE = "working as cooking school"


def make_context(history: ConversationHistory) -> ConversationContext:
    return ConversationContext(all_history=history, history=history, summary="")


def check(label: str, message: str):
    lower = message.lower()
    if BAD_PHRASE in lower:
        print(f"  FAIL  [{label}]: produced '{BAD_PHRASE}'")
        print(f"         Message: {message!r}")
    else:
        print(f"  PASS  [{label}]")
        print(f"         Message: {message!r}")


async def run_scenario(label: str, experiences: list[ExperienceEntity], run_turn2: bool = False):
    """Run turn 1 (fallback) and optionally turn 2 (LLM) for a set of experiences."""
    print(f"\n{'='*60}")
    print(f"Scenario: {label}")
    print(f"{'='*60}")
    for exp in experiences:
        print(f"  experience_title          = {exp.experience_title!r}")
        print(f"  normalized_experience_title = {exp.normalized_experience_title!r}")
        print(f"  company                   = {exp.company!r}")

    agent = PreferenceElicitationAgent()
    state = PreferenceElicitationAgentState(
        session_id=1,
        initial_experiences_snapshot=experiences,
        use_db6_for_fresh_data=False,
    )
    agent.set_state(state)
    history = ConversationHistory()

    # Turn 1: always uses the deterministic fallback (no LLM call)
    out1 = await agent.execute(AgentInput(message="", is_artificial=True), make_context(history))
    check("turn 1 fallback", out1.message_for_user)
    history.turns.append(ConversationTurn(index=0, input=AgentInput(message=""), output=out1))

    if not run_turn2:
        return

    # Turn 2: user gives any answer → agent calls LLM with the guard-instrumented prompt
    user_reply = "I really enjoyed the hands-on cooking."
    print(f"\n  [turn 2 — LLM call] user says: {user_reply!r}")
    out2 = await agent.execute(
        AgentInput(message=user_reply, is_artificial=False),
        make_context(history),
    )
    check("turn 2 LLM", out2.message_for_user)


async def main():
    print("\nPreference Agent — Experience Title Phrasing Fix\n")

    # Scenario A: place name as title, company present → fallback should say "working at [company]"
    await run_scenario(
        label="A — place title, company present",
        experiences=[
            ExperienceEntity(
                uuid="exp-a",
                experience_title="cooking school",
                normalized_experience_title=None,
                company="Mama Rocks Culinary Institute",
                location="Nairobi",
                timeline=Timeline(start="2021", end="2023"),
                work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT,
            )
        ],
        run_turn2=False,  # deterministic only
    )

    # Scenario B: place name, no company, normalized title available → "working as [normalized]"
    await run_scenario(
        label="B — place title, no company, normalized title present",
        experiences=[
            ExperienceEntity(
                uuid="exp-b",
                experience_title="cooking school",
                normalized_experience_title="Cook",
                company=None,
                location="Nairobi",
                timeline=Timeline(start="2021", end="2023"),
                work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT,
            )
        ],
        run_turn2=False,
    )

    # Scenario C: proper job title (regression check) → existing behaviour preserved
    await run_scenario(
        label="C — proper job title (regression guard)",
        experiences=[
            ExperienceEntity(
                uuid="exp-c",
                experience_title="Mathematics Teacher",
                normalized_experience_title=None,
                company="Alliance High School",
                location="Kikuyu",
                timeline=Timeline(start="2018", end="2023"),
                work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT,
            )
        ],
        run_turn2=False,
    )

    # Scenario D: full LLM call with place-name title + company — does the LLM follow the guard?
    print("\n[Scenario D requires one real LLM call — skip with Ctrl-C if offline]")
    try:
        await run_scenario(
            label="D — LLM guard check (place title, company present)",
            experiences=[
                ExperienceEntity(
                    uuid="exp-d",
                    experience_title="cooking school",
                    normalized_experience_title=None,
                    company="Mama Rocks Culinary Institute",
                    location="Nairobi",
                    timeline=Timeline(start="2021", end="2023"),
                    work_type=WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT,
                )
            ],
            run_turn2=True,
        )
    except KeyboardInterrupt:
        print("\n  [skipped by user]")

    print("\nDone.\n")


if __name__ == "__main__":
    asyncio.run(main())
