#!/usr/bin/env python3
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic_settings import BaseSettings

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from app.application_state import ApplicationState
from app.agent.experience.experience_entity import ExperienceEntity
from app.conversations.phase_data import apply_entry_phase
from app.conversations.phase_state_machine import JourneyPhase
from app.server_dependencies.database_collections import Collections
from app.store.database_application_state_store import DatabaseApplicationStateStore
from app.users.repositories import UserPreferenceRepository
from app.agent.preference_elicitation_agent.types import PreferenceVector

from populate_sample_conversation import (
    create_collect_experience_state,
    create_conversation_memory_manager_state,
    create_explore_experiences_director_state,
    create_sample_conversation_history_for_preference_skip,
    create_skills_explorer_agent_state_from_explore,
    create_welcome_agent_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ScriptSettings(BaseSettings):
    application_mongodb_uri: str = ""
    application_database_name: str = ""

    class Config:
        extra = "ignore"


PHASE_CHOICES = {
    "PREFERENCE_ELICITATION": JourneyPhase.PREFERENCE_ELICITATION,
    "MATCHING": JourneyPhase.MATCHING,
    "RECOMMENDATIONS": JourneyPhase.RECOMMENDATION,
}


def _populate_for_preference_elicitation(state: ApplicationState) -> None:
    session_id = state.session_id
    conversation_history = create_sample_conversation_history_for_preference_skip()
    explore_state = create_explore_experiences_director_state(session_id)

    state.explore_experiences_director_state = explore_state
    state.collect_experience_state = create_collect_experience_state(session_id)
    state.skills_explorer_agent_state = create_skills_explorer_agent_state_from_explore(
        explore_state
    )
    state.conversation_memory_manager_state = create_conversation_memory_manager_state(
        session_id, conversation_history
    )
    state.welcome_agent_state = create_welcome_agent_state(session_id)

    pref = state.preference_elicitation_agent_state
    explored = explore_state.explored_experiences or []
    pref.initial_experiences_snapshot = [
        ExperienceEntity(
            **exp.model_dump(exclude={"top_skills", "remaining_skills"}),
            top_skills=[s for _, s in exp.top_skills] if exp.top_skills else [],
            remaining_skills=[s for _, s in exp.remaining_skills] if exp.remaining_skills else [],
        )
        for exp in explored
    ]


def _populate_for_matching(state: ApplicationState) -> None:
    _populate_for_preference_elicitation(state)
    pref = state.preference_elicitation_agent_state
    pref.preference_vector = PreferenceVector(confidence_score=0.5)
    pref.conversation_phase = "COMPLETE"


async def set_phase(mongo_uri: str, db_name: str, user_id: str, phase: JourneyPhase) -> None:
    client = AsyncIOMotorClient(mongo_uri, tlsAllowInvalidCertificates=True)
    await client.server_info()
    db = client.get_database(db_name)

    user_prefs_repo = UserPreferenceRepository(db=db)
    user_prefs = await user_prefs_repo.get_user_preference_by_user_id(user_id)
    if not user_prefs or not user_prefs.sessions:
        logger.error("User %s has no sessions. Create a session first.", user_id)
        client.close()
        sys.exit(1)

    session_id = user_prefs.sessions[0]
    logger.info("Using latest session_id=%s for user %s", session_id, user_id)

    if phase == JourneyPhase.RECOMMENDATION:
        recs = db.get_collection(Collections.USER_RECOMMENDATIONS)
        doc = await recs.find_one({"user_id": user_id})
        if not doc:
            logger.error(
                "User %s has no recommendations. Run import_user_recommendations first.",
                user_id,
            )
            client.close()
            sys.exit(1)
        logger.info("User %s has recommendations. Will skip to recommendations on next message.", user_id)
        client.close()
        return

    store = DatabaseApplicationStateStore(db=db)
    existing = await store.get_state(session_id)
    if existing:
        state = existing
    else:
        from app.countries import Country
        state = ApplicationState.new_state(session_id=session_id, country_of_user=Country.UNSPECIFIED)
        await store.save_state(state)

    if phase == JourneyPhase.PREFERENCE_ELICITATION:
        _populate_for_preference_elicitation(state)
    elif phase == JourneyPhase.MATCHING:
        _populate_for_matching(state)

    apply_entry_phase(state, phase)

    await store.save_state(state)
    logger.info("Populated session %s for user %s to skip to phase: %s", session_id, user_id, phase.value)
    client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set session journey phase by populating session state.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("user_id", help="Compass user ID")
    parser.add_argument(
        "--phase",
        "-p",
        required=True,
        choices=list(PHASE_CHOICES.keys()),
        help="Phase to skip to (skills_elicitation not allowed)",
    )
    parser.add_argument("--hot-run", action="store_true", help="Apply changes to the database")
    args = parser.parse_args()

    settings = ScriptSettings()
    if not args.hot_run:
        logger.info(
            "Dry run. Would set user %s to phase %s. Use --hot-run to apply.",
            args.user_id,
            args.phase,
        )
        return

    if not settings.application_mongodb_uri or not settings.application_database_name:
        logger.error(
            "Set APPLICATION_MONGODB_URI and APPLICATION_DATABASE_NAME in .env"
        )
        sys.exit(1)

    phase = PHASE_CHOICES[args.phase]
    if phase == JourneyPhase.RECOMMENDATION:
        logger.info(
            "For recommendations phase, verifying user has recommendations."
        )

    asyncio.run(set_phase(
        settings.application_mongodb_uri,
        settings.application_database_name,
        args.user_id,
        phase,
    ))


if __name__ == "__main__":
    main()
