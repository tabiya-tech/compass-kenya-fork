from __future__ import annotations

from enum import Enum
from typing import Sequence

from pydantic import BaseModel, ConfigDict


class JourneyPhase(str, Enum):
    """
    High-level journey phases used to decide
    where to enter the counseling flow for a user.

    This is intentionally decoupled from concrete agents or data
    sources so it can be reused across backends (Kobo, DB6, etc.).
    """

    SKILLS_ELICITATION = "SKILLS_ELICITATION"
    PREFERENCE_ELICITATION = "PREFERENCE_ELICITATION"
    MATCHING = "MATCHING"
    RECOMMENDATION = "RECOMMENDATION"


_DEFAULT_PHASE_ORDER: tuple[JourneyPhase, ...] = (
    JourneyPhase.SKILLS_ELICITATION,
    JourneyPhase.PREFERENCE_ELICITATION,
    JourneyPhase.MATCHING,
    JourneyPhase.RECOMMENDATION,
)


class PhaseDataStatus(BaseModel):
    """
    Snapshot of which upstream artifacts already exist for a user.

    Integration layers are responsible for populating these flags
    based on concrete stores like ApplicationState, DB6, Kobo, etc.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    has_skills_elicitation: bool = False
    has_preference_elicitation: bool = False
    has_matching: bool = False
    has_recommendation: bool = False


def determine_start_phase(
    data: PhaseDataStatus,
    allowed_phases: Sequence[JourneyPhase] | None = None,
) -> JourneyPhase:
    """
    Decide which journey phase a user should start in.

    The algorithm is:
    - Take the canonical phase order (skills → preferences → matching → recommendation)
    - Intersect it with allowed_phases (if provided) to support cohort-specific flows
    - Walk from the end backwards and pick the furthest phase we can safely enter
      given the existing data snapshot.

    Examples:
        - If recommendation data already exists, start at RECOMMENDATION.
        - If preferences exist but no recommendations, start at MATCHING.
        - If skills exist but no preferences, start at PREFERENCE_ELICITATION.
        - If nothing exists, start at SKILLS_ELICITATION.
    """
    if allowed_phases is None:
        ordered_allowed = list(_DEFAULT_PHASE_ORDER)
    else:
        allowed_set = set(allowed_phases)
        ordered_allowed = [p for p in _DEFAULT_PHASE_ORDER if p in allowed_set]
        if not ordered_allowed:
            raise ValueError("allowed_phases cannot be empty")

    for phase in reversed(ordered_allowed):
        if _can_start_at_phase(phase, data):
            return phase

    # Fallback: if for some reason no phase satisfied its own preconditions,
    # start at the earliest allowed phase.
    return ordered_allowed[0]


def _can_start_at_phase(phase: JourneyPhase, data: PhaseDataStatus) -> bool:
    """
    Phase-specific preconditions for entering the journey at that phase.
    """
    if phase is JourneyPhase.RECOMMENDATION:
        # If we already have recommendations (e.g. from Kobo cron),
        # we can jump straight to discussing them.
        return data.has_recommendation

    if phase is JourneyPhase.MATCHING:
        # We can start at matching if:
        # - matching results already exist, or
        # - both skills and preferences are available so matching can run.
        return data.has_matching or (
            data.has_skills_elicitation and data.has_preference_elicitation
        )

    if phase is JourneyPhase.PREFERENCE_ELICITATION:
        # We can start at preference elicitation if:
        # - a preference vector already exists, or
        # - skills elicitation has been completed so we have inputs.
        return data.has_preference_elicitation or data.has_skills_elicitation

    if phase is JourneyPhase.SKILLS_ELICITATION:
        # Baseline phase, always safe to enter.
        return True

    return False

