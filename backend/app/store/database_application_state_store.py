import asyncio
import logging
from typing import AsyncIterator

from motor.motor_asyncio import AsyncIOMotorDatabase

from ._utils import filter_explored_experiences

from app.agent.agent_director.abstract_agent_director import (
    AgentDirectorState, ConversationPhase, CounselingSubPhase,
)
from app.agent.sentry_trace import trace
from app.conversations.phase_state_machine import JourneyPhase
from app.agent.collect_experiences_agent import CollectExperiencesAgentState
from app.agent.explore_experiences_agent_director import ExploreExperiencesAgentDirectorState
from app.agent.skill_explorer_agent import SkillsExplorerAgentState
from app.agent.welcome_agent import WelcomeAgentState
from app.agent.preference_elicitation_agent import PreferenceElicitationAgentState
from app.agent.recommender_advisor_agent import RecommenderAdvisorAgentState
from app.application_state import ApplicationStateStore, ApplicationState
from app.server_dependencies.database_collections import Collections
from app.conversation_memory.conversation_memory_types import ConversationMemoryManagerState


class DatabaseApplicationStateStore(ApplicationStateStore):
    """
    A MongoDB store for application state.
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self._agent_director_collection = db.get_collection(Collections.AGENT_DIRECTOR_STATE)
        self._welcome_agent_state = db.get_collection(Collections.WELCOME_AGENT_STATE)
        self._explore_experiences_director_state_collection = db.get_collection(Collections.EXPLORE_EXPERIENCES_DIRECTOR_STATE)
        self._conversation_memory_manager_state_collection = db.get_collection(Collections.CONVERSATION_MEMORY_MANAGER_STATE)
        self._collect_experience_state_collection = db.get_collection(Collections.COLLECT_EXPERIENCE_STATE)
        self._skills_explorer_agent_state_collection = db.get_collection(Collections.SKILLS_EXPLORER_AGENT_STATE)
        self._preference_elicitation_agent_state_collection = db.get_collection(Collections.PREFERENCE_ELICITATION_AGENT_STATE)
        self._recommender_advisor_agent_state_collection = db.get_collection(Collections.RECOMMENDER_ADVISOR_AGENT_STATE)
        self._logger = logging.getLogger(self.__class__.__name__)

    async def get_state(self, session_id: int) -> ApplicationState | None:
        """
        Get the application state for a session from the databaseProtected Attributes and memory.
        """
        try:

            # Get the states of the different components from the database
            # Using $eq to prevent NoSQL injection
            results = await asyncio.gather(
                self._agent_director_collection.find_one({"session_id": {"$eq": session_id}}, {'_id': False}),
                self._welcome_agent_state.find_one({"session_id": {"$eq": session_id}}, {'_id': False}),
                self._explore_experiences_director_state_collection.find_one({"session_id": {"$eq": session_id}}, {'_id': False}),
                self._conversation_memory_manager_state_collection.find_one({"session_id": {"$eq": session_id}}, {'_id': False}),
                self._collect_experience_state_collection.find_one({"session_id": {"$eq": session_id}}, {'_id': False}),
                self._skills_explorer_agent_state_collection.find_one({"session_id": {"$eq": session_id}}, {'_id': False}),
                self._preference_elicitation_agent_state_collection.find_one({"session_id": {"$eq": session_id}}, {'_id': False}),
                self._recommender_advisor_agent_state_collection.find_one({"session_id": {"$eq": session_id}}, {'_id': False})
            )
            if all(_state_part is None for _state_part in results):
                # If all the states are None, return None
                self._logger.info("No application state found for session ID %s", session_id)
                return None

            collection_names = [
                Collections.AGENT_DIRECTOR_STATE,
                Collections.WELCOME_AGENT_STATE,
                Collections.EXPLORE_EXPERIENCES_DIRECTOR_STATE,
                Collections.CONVERSATION_MEMORY_MANAGER_STATE,
                Collections.COLLECT_EXPERIENCE_STATE,
                Collections.SKILLS_EXPLORER_AGENT_STATE,
                Collections.PREFERENCE_ELICITATION_AGENT_STATE,
                Collections.RECOMMENDER_ADVISOR_AGENT_STATE
            ]

            if len(collection_names) != len(results):
                self._logger.error(
                    "Mismatch between collection names and results for session ID %s. "
                    "Expected %d results, got %d.",
                    session_id,
                    len(collection_names),
                    len(results)
                )
                return None

            missing_parts = [name for name, result in zip(collection_names, results) if result is None]
            # Allow recommender_advisor_agent_state to be missing for backward compatibility
            critical_missing_parts = [name for name in missing_parts if name != Collections.RECOMMENDER_ADVISOR_AGENT_STATE]
            if critical_missing_parts:
                self._logger.error(
                    "Missing critical application state part(s) for session ID %s. Missing part(s): %s",
                    session_id,
                    critical_missing_parts
                )
                return None

            # Successfully retrieved all states
            (agent_director_state,
             welcome_agent_state,
             explore_experiences_director_state,
             conversation_memory_manager_state,
             collect_experience_state,
             skills_explorer_agent_state,
             preference_elicitation_agent_state,
             recommender_advisor_agent_state) = results

            # Backward compatibility: Create default recommender state if missing
            if recommender_advisor_agent_state is None:
                self._logger.info("Creating default RecommenderAdvisorAgentState for session ID %s (backward compatibility)", session_id)
                recommender_advisor_agent_state_obj = RecommenderAdvisorAgentState(session_id=session_id)
            else:
                recommender_advisor_agent_state_obj = RecommenderAdvisorAgentState.from_document(recommender_advisor_agent_state)

            state = ApplicationState(session_id=session_id,
                                     agent_director_state=AgentDirectorState.from_document(agent_director_state),
                                     welcome_agent_state=WelcomeAgentState.from_document(welcome_agent_state),
                                     explore_experiences_director_state=ExploreExperiencesAgentDirectorState.from_document(explore_experiences_director_state),
                                     conversation_memory_manager_state=ConversationMemoryManagerState.from_document(conversation_memory_manager_state),
                                     collect_experience_state=CollectExperiencesAgentState.from_document(collect_experience_state),
                                     skills_explorer_agent_state=SkillsExplorerAgentState.from_document(skills_explorer_agent_state),
                                     preference_elicitation_agent_state=PreferenceElicitationAgentState.from_document(preference_elicitation_agent_state),
                                     recommender_advisor_agent_state=recommender_advisor_agent_state_obj)

            # Upgrade the state if necessary
            state = await self._upgrade_state(state)

            return state

        except Exception as e:  # pylint: disable=broad-except
            self._logger.error("Failed to get application state for session ID %s: %s", session_id, e, exc_info=True)
            return None

    async def save_state(self, state: ApplicationState):
        """
        Save the application state for a session.
        """
        try:
            # look through all the states to check that they use the same session_id
            # since all the session_ids should be the same, we can use any of them
            # here we use the agent_director_state.session_id
            session_id = state.agent_director_state.session_id
            if not all([state.explore_experiences_director_state.session_id == session_id,
                        state.welcome_agent_state.session_id == session_id,
                        state.conversation_memory_manager_state.session_id == session_id,
                        state.collect_experience_state.session_id == session_id,
                        state.skills_explorer_agent_state.session_id == session_id,
                        state.preference_elicitation_agent_state.session_id == session_id,
                        state.recommender_advisor_agent_state.session_id == session_id]):
                raise ValueError("All states must have the same session_id")
            # Write the component states to the database
            # Using $eq to prevent NoSQL injection
            await asyncio.gather(
                self._agent_director_collection.update_one({"session_id": {"$eq": session_id}}, {"$set": state.agent_director_state.model_dump()}, upsert=True),
                self._welcome_agent_state.update_one({"session_id": {"$eq": session_id}}, {"$set": state.welcome_agent_state.model_dump()}, upsert=True),
                self._explore_experiences_director_state_collection.update_one({"session_id": {"$eq": session_id}},
                                                                               {"$set": state.explore_experiences_director_state.model_dump()}, upsert=True),
                self._conversation_memory_manager_state_collection.update_one({"session_id": {"$eq": session_id}},
                                                                              {"$set": state.conversation_memory_manager_state.model_dump()}, upsert=True),
                self._collect_experience_state_collection.update_one({"session_id": {"$eq": session_id}}, {"$set": state.collect_experience_state.model_dump()},
                                                                     upsert=True),
                self._skills_explorer_agent_state_collection.update_one({"session_id": {"$eq": session_id}},
                                                                        {"$set": state.skills_explorer_agent_state.model_dump()}, upsert=True),
                self._preference_elicitation_agent_state_collection.update_one({"session_id": {"$eq": session_id}},
                                                                               {"$set": state.preference_elicitation_agent_state.model_dump()}, upsert=True),
                self._recommender_advisor_agent_state_collection.update_one({"session_id": {"$eq": session_id}},
                                                                            {"$set": state.recommender_advisor_agent_state.model_dump()}, upsert=True)
            )

        except Exception as e:  # pylint: disable=broad-except
            # Log the error and raise an exception so that the caller can handle it
            self._logger.error("Failed to save application state for session ID %s: %s", state.agent_director_state.session_id, e, exc_info=True)
            raise

    async def delete_state(self, session_id: int) -> None:
        """
        Delete the application state for a session.
        """
        try:
            # Delete the states from the database
            # Using $eq to prevent NoSQL injection
            await asyncio.gather(
                self._agent_director_collection.delete_one({"session_id": {"$eq": session_id}}),
                self._welcome_agent_state.delete_one({"session_id": {"$eq": session_id}}),
                self._explore_experiences_director_state_collection.delete_one({"session_id": {"$eq": session_id}}),
                self._conversation_memory_manager_state_collection.delete_one({"session_id": {"$eq": session_id}}),
                self._collect_experience_state_collection.delete_one({"session_id": {"$eq": session_id}}),
                self._skills_explorer_agent_state_collection.delete_one({"session_id": {"$eq": session_id}}),
                self._preference_elicitation_agent_state_collection.delete_one({"session_id": {"$eq": session_id}}),
                self._recommender_advisor_agent_state_collection.delete_one({"session_id": {"$eq": session_id}})
            )

        except Exception as e:  # pylint: disable=broad-except
            # Log the error and raise an exception so that the caller can handle it
            self._logger.error("Failed to delete application state for session ID %s: %s", session_id, e, exc_info=True)
            raise

    async def get_all_session_ids(self) -> AsyncIterator[int]:
        """
        Stream all application states.
        Returns an async generator of ApplicationState objects.
        """
        try:
            # Create a cursor for streaming conversation memory manager documents
            cursor = self._conversation_memory_manager_state_collection.find(
                {}, {'_id': False, 'session_id': True}
            )

            async for doc in cursor:
                session_id = doc.get('session_id')
                if session_id is None:
                    self._logger.error("Session ID not found in document: %s", doc)
                    continue
                yield session_id

        except Exception as e:
            self._logger.error("Failed to stream application states: %s", e, exc_info=True)
            raise

    async def _upgrade_state(self, state: ApplicationState) -> ApplicationState:
        """
        Upgrade the state to the latest version if necessary.
        Saves it andy returns the upgraded state.

        This method should not raise an exception but log it and return the state as is.
        As we didn't upgrade the state, it will be returned as is.
        """

        try:
            _changes = False

            # The field `state.explore_experiences_director_state.explored_experiences` was added in a later version
            # if it is empty, and we have explored experiences, we populate it
            # with the experiences that have been processed
            if state.explore_experiences_director_state.explored_experiences is None:
                self._logger.info("upgrading state: populating explored_experiences field")
                state.explore_experiences_director_state.explored_experiences = filter_explored_experiences(state)
                _changes = True

            # Populate preference agent's initial_experiences_snapshot if empty
            # This ensures the preference agent can reference existing experiences
            if (state.preference_elicitation_agent_state.initial_experiences_snapshot is None and
                state.explore_experiences_director_state.explored_experiences):
                explored_count = len(state.explore_experiences_director_state.explored_experiences)
                explored_titles = [exp.experience_title for exp in state.explore_experiences_director_state.explored_experiences]
                self._logger.info(
                    f"upgrading state: populating preference agent initial_experiences_snapshot "
                    f"with {explored_count} experiences: {explored_titles}"
                )
                # Copy explored experiences to preference agent snapshot
                # Note: explored_experiences has tuple-format skills [(score, skill), ...]
                # but initial_experiences_snapshot needs plain ExperienceEntity with skills as dicts
                # Strip the tuple wrapping from top_skills during the copy
                from app.agent.experience.experience_entity import ExperienceEntity
                state.preference_elicitation_agent_state.initial_experiences_snapshot = [
                    ExperienceEntity(
                        **exp.model_dump(exclude={'top_skills', 'remaining_skills'}),
                        top_skills=[skill for _, skill in exp.top_skills] if exp.top_skills else [],
                        remaining_skills=[skill for _, skill in exp.remaining_skills] if exp.remaining_skills else []
                    )
                    for exp in state.explore_experiences_director_state.explored_experiences
                ]
                _changes = True
                self._logger.info(
                    "Successfully populated initial_experiences_snapshot with %d experiences",
                    explored_count
                )
            elif state.preference_elicitation_agent_state.initial_experiences_snapshot is None:
                self._logger.warning(
                    "Cannot populate initial_experiences_snapshot: explored_experiences is empty or None"
                )
            else:
                # Already populated, log for debugging
                snapshot_count = len(state.preference_elicitation_agent_state.initial_experiences_snapshot)
                snapshot_titles = [exp.experience_title for exp in state.preference_elicitation_agent_state.initial_experiences_snapshot]
                self._logger.debug(
                    f"initial_experiences_snapshot already populated with {snapshot_count} experiences: {snapshot_titles}"
                )

            # ========== Recommender Agent Data Transfer ==========
            # Sync PEAS values into RAAS so the recommender has what it needs.
            # Gate semantics mirror service._prepare_recommender_state_if_needed:
            # only sync when the recommender is actually about to run (sub-phase
            # RECOMMENDER_ADVISOR), post-COUNSELING phases (CHECKOUT/ENDED self-heal
            # for stuck production sessions), or a step-skip into RECOMMENDATION /
            # MATCHING. ConversationPhase.COUNSELING alone is too coarse because it
            # also covers EXPLORE_EXPERIENCES and PREFERENCE_ELICITATION sub-phases
            # where PEAS is still at defaults.
            director_state = state.agent_director_state
            pref_state = state.preference_elicitation_agent_state
            rec_state = state.recommender_advisor_agent_state
            in_recommender_phase = (
                (director_state.current_phase == ConversationPhase.COUNSELING
                 and director_state.counseling_sub_phase == CounselingSubPhase.RECOMMENDER_ADVISOR)
                or director_state.current_phase in (
                    ConversationPhase.CHECKOUT, ConversationPhase.ENDED,
                )
                or director_state.skip_to_phase in (
                    JourneyPhase.RECOMMENDATION, JourneyPhase.MATCHING,
                )
            )
            # n_vignettes_completed > 0 means PEAS has actually done meaningful work;
            # without this the CHECKOUT/ENDED self-heal path would copy PEAS defaults
            # into RAAS for sessions that never went through vignettes.
            peas_has_data = pref_state.preference_vector.n_vignettes_completed > 0
            if in_recommender_phase and peas_has_data:
                _rec_changes = False
                # Content-only equality on PV: its last_updated field is a fresh
                # datetime.now() on default instances, so Pydantic's ``==`` would
                # falsely report differences whenever the two halves are constructed
                # separately (e.g. on a load before any sync has fired yet).
                if not pref_state.preference_vector.content_equals(rec_state.preference_vector):
                    rec_state.preference_vector = pref_state.preference_vector
                    _rec_changes = True
                    trace(
                        "recommender.preference_vector.synced",
                        session_id=state.session_id,
                        source="state_load",
                        n_vignettes_completed=int(pref_state.preference_vector.n_vignettes_completed),
                        confidence_score=float(pref_state.preference_vector.confidence_score),
                    )

                if pref_state.hb_scores and pref_state.hb_ranking:
                    new_bws = {wa_id: entry["mean"] for wa_id, entry in pref_state.hb_scores.items()}
                    new_top10 = list(pref_state.hb_ranking)
                    source = "hb"
                else:
                    new_bws = pref_state.bws_scores
                    new_top10 = list(pref_state.top_10_bws) if pref_state.top_10_bws else None
                    source = "fallback"
                if rec_state.bws_scores != new_bws or rec_state.top_10_bws != new_top10:
                    rec_state.bws_scores = new_bws
                    rec_state.top_10_bws = new_top10
                    _rec_changes = True
                    trace(
                        "recommender.bws.synced",
                        session_id=state.session_id,
                        source_event="state_load",
                        bws_items=len(new_bws) if new_bws else 0,
                        top_10_bws_len=len(new_top10) if new_top10 else 0,
                        source=source,
                    )

                if rec_state.skills_vector is None:
                    rec_state.skills_vector = self._extract_skills_from_experiences(
                        pref_state.initial_experiences_snapshot
                    )
                    _rec_changes = True

                # Education experiences are recommender input; mirror the service-layer
                # behavior so consumers that come in through state load (export script,
                # metrics jobs) get the same payload.
                new_education = [
                    e for e in state.collect_experience_state.collected_data
                    if e.source == "education"
                ]
                if rec_state.education_experiences != new_education:
                    rec_state.education_experiences = new_education
                    _rec_changes = True

                if rec_state.youth_id is None:
                    rec_state.youth_id = f"youth_{state.session_id}"
                    _rec_changes = True

                if _rec_changes:
                    _changes = True
                    self._logger.info("Synced PEAS data into recommender state on load")

            # after the upgrade, we save the state
            if _changes:
                await self.save_state(state)

            # Currently, no upgrades are needed, but this method can be extended in the future
            return state
        except Exception as e:  # pylint: disable=broad-except
            self._logger.error("Failed to upgrade application state: %s", e, exc_info=True)
            return state

    def _extract_skills_from_experiences(self, experiences):
        """
        Extract skills vector from experiences snapshot.

        Args:
            experiences: List of ExperienceEntity objects or None

        Returns:
            Dictionary with skills vector data compatible with Node2Vec
        """
        if not experiences:
            return {"skills": [], "total_experiences": 0}

        try:
            # Use SkillsExtractor to aggregate skills
            from app.agent.recommender_advisor_agent.skills_extractor import SkillsExtractor
            extractor = SkillsExtractor()
            return extractor.extract_skills_vector(experiences)
        except Exception as e:  # pylint: disable=broad-except
            self._logger.error("Failed to extract skills from experiences: %s", e, exc_info=True)
            # Return empty skills vector on error
            return {"skills": [], "total_experiences": 0}
