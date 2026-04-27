import asyncio
import string
from typing import Optional, Mapping, Any

from pydantic import BaseModel, Field, field_serializer, field_validator

from app.agent.agent import Agent
from app.agent.agent_types import AgentInput, AgentOutput
from app.agent.agent_types import AgentType
from app.agent.collect_experiences_agent._bridge_llm import generate_bridge_to_work_type
from app.agent.collect_experiences_agent._conversation_llm import _ConversationLLM, ConversationLLMAgentOutput, \
    _get_experience_type, fill_incomplete_fields_as_declined
from app.agent.persona_detector import PersonaType
from app.agent.collect_experiences_agent._dataextraction_llm import _DataExtractionLLM
from app.agent.collect_experiences_agent._transition_decision_tool import TransitionDecisionTool, TransitionDecision
from app.agent.collect_experiences_agent._types import CollectedData
from app.agent.experience.experience_entity import ExperienceEntity, ResponsibilitiesData
from app.agent.experience.timeline import Timeline
from app.agent.experience.work_type import WorkType
from app.agent.linking_and_ranking_pipeline import ExperiencePipelineConfig
from app.agent.linking_and_ranking_pipeline.infer_occupation_tool import InferOccupationTool
from app.conversation_memory.conversation_memory_types import ConversationContext
from app.countries import Country
from app.i18n.translation_service import t
from app.vector_search.esco_entities import OccupationSkillEntity
from app.vector_search.vector_search_dependencies import SearchServices


# Confirmation tokens that — when sent on their own — trigger the data-extraction
# fast-path. Skipping data extraction on these saves ~750ms (IntentAnalyzer +
# any per-operation entity/temporal calls) on the most common turn type.
#
# Trade-off: a yes/no answer to a data question (e.g. "Was this paid work?") will
# be missed and the agent will re-ask the question on a future turn. The agent
# asks open-ended questions for most fields (title, dates, company, location), so
# the only field exposed to this regression is `paid_work`. Acceptable for the
# latency win on every other confirmation turn.
#
# Keep this list small. Expand carefully — every new entry expands the surface
# area where data signals can be lost.
_CONFIRMATION_TOKENS: frozenset[str] = frozenset({
    # English
    "yes", "yeah", "yep", "y",
    "no", "nope", "n",
    "ok", "okay", "k",
    "sure",
    # Swahili
    "ndio", "ndiyo",
    "hapana",
    "sawa",
})


def _is_simple_confirmation(message: str) -> bool:
    """True if the user's input is a bare confirmation token (yes/no/ok variants)."""
    if not message:
        return False
    normalized = message.strip().lower().rstrip(string.punctuation).strip()
    return normalized in _CONFIRMATION_TOKENS


def _deserialize_work_types(value: list[str] | list[WorkType]) -> list[WorkType]:
    if isinstance(value, list):
        result = []
        for x in value:
            if isinstance(x, str):
                wt = WorkType.from_string_key(x)
                if wt is not None:
                    result.append(wt)
            else:
                result.append(x)
        return result
    return value


def _select_normalized_title(*,
                             original_title: str,
                             contextual_titles: list[str],
                             esco_occupations: list[OccupationSkillEntity]) -> Optional[str]:
    cleaned_original = (original_title or "").strip().lower()
    candidates: list[str] = []

    for title in contextual_titles:
        cleaned = (title or "").strip()
        if cleaned:
            candidates.append(cleaned)
    for occupation_skill in esco_occupations:
        label = occupation_skill.occupation.preferredLabel.strip()
        if label:
            candidates.append(label)

    if not candidates:
        return None

    for candidate in candidates:
        if candidate.lower() != cleaned_original:
            return candidate

    return candidates[0]


class CollectExperiencesAgentState(BaseModel):
    """
    Stores the user-specific state for this agent. Managed centrally.
    """

    session_id: int
    """
    The session id of the conversation.
    """

    country_of_user: Country = Field(default=Country.UNSPECIFIED)
    """
    The country of the user.
    """

    persona_type: PersonaType = Field(default=PersonaType.INFORMAL)
    """
    The detected persona type for adapting prompts.
    """

    collected_data: list[CollectedData] = Field(default_factory=list)
    """
    The data collected during the conversation.
    """

    unexplored_types: list[WorkType] = Field(default_factory=lambda: [WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT,
                                                                      WorkType.SELF_EMPLOYMENT,
                                                                      WorkType.FORMAL_SECTOR_UNPAID_TRAINEE_WORK,
                                                                      WorkType.UNSEEN_UNPAID])
    """
    The types of work experiences that have not been explored yet.
    """

    explored_types: list[WorkType] = Field(default_factory=list)
    """
    The questions asked by the conversational LLM.
    """

    first_time_visit: bool = True
    """
    Whether this is the first time the agent is visited during the conversation.
    """

    education_phase_done: bool = False
    """
    Whether the education collection phase has been completed.
    Education is asked on first_time_visit before the work type loop begins.
    """

    class Config:
        """
        Disallow extra fields in the model
        """
        extra = "forbid"

    @field_serializer("country_of_user")
    def serialize_country_of_user(self, country_of_user: Country, _info):
        return country_of_user.name

    @field_validator("country_of_user", mode='before')
    def deserialize_country_of_user(cls, value: str | Country) -> Country:
        if isinstance(value, str):
            return Country[value]
        return value

    @field_serializer("persona_type")
    def serialize_persona_type(self, persona_type: PersonaType, _info):
        return persona_type.name

    @field_validator("persona_type", mode='before')
    def deserialize_persona_type(cls, value: str | PersonaType) -> PersonaType:
        if isinstance(value, str):
            return PersonaType[value]
        return value

    # use a field serializer to serialize the explored_types
    # we use the name of the Enum instead of the value because that makes the code less brittle
    @field_serializer("explored_types")
    def serialize_explored_types(self, explored_types: list[WorkType], _info):
        # We serialize the explored_types to a list of strings (the names of the Enum)
        return [x.name for x in explored_types]

    # Deserialize the explored_types from the enum name
    @field_validator("explored_types", mode='before')
    def deserialize_explored_types(cls, value: list[str] | list[WorkType]) -> list[WorkType]:
        return _deserialize_work_types(value)

    # use a field serializer to serialize the unexplored_types
    # we use the name of the Enum instead of the value because that makes the code less brittle
    @field_serializer("unexplored_types")
    def serialize_unexplored_types(self, unexplored_types: list[WorkType], _info):
        # We serialize the unexplored_types to a list of strings (the names of the Enum)
        return [x.name for x in unexplored_types]

    # Deserialize the unexplored_types from the enum name
    @field_validator("unexplored_types", mode='before')
    def deserialize_unexplored_types(cls, value: list[str] | list[WorkType]) -> list[WorkType]:
        return _deserialize_work_types(value)

    @staticmethod
    def from_document(_doc: Mapping[str, Any]) -> "CollectExperiencesAgentState":
        return CollectExperiencesAgentState(session_id=_doc["session_id"],
                                            # For backward compatibility with old documents that don't have the country_of_user field, set it to UNSPECIFIED
                                            country_of_user=_doc.get("country_of_user", Country.UNSPECIFIED),
                                            persona_type=_doc.get("persona_type", PersonaType.INFORMAL),
                                            collected_data=_doc["collected_data"],
                                            unexplored_types=_doc["unexplored_types"],
                                            explored_types=_doc["explored_types"],
                                            first_time_visit=_doc["first_time_visit"],
                                            education_phase_done=_doc.get("education_phase_done", False))


class CollectExperiencesAgent(Agent):
    """
    This agent converses with user and collects basic information about their work experiences.
    """

    _MAX_NORMALIZATION_ATTEMPTS = 2

    def __init__(self,
                 *,
                 search_services: SearchServices | None = None,
                 experience_pipeline_config: ExperiencePipelineConfig | None = None):
        super().__init__(agent_type=AgentType.COLLECT_EXPERIENCES_AGENT,
                         is_responsible_for_conversation_history=False)
        self._experiences: list[ExperienceEntity] = []
        self._state: Optional[CollectExperiencesAgentState] = None
        self._search_services = search_services
        self._experience_pipeline_config = experience_pipeline_config
        # Per-experience attempt counter for occupation normalization. Kept as an
        # instance attribute (not on CollectExperiencesAgentState) so it never gets
        # serialized into LLM prompts via `model_dump_json()` or persisted to MongoDB.
        # Reset on `set_state()` so it does not leak across sessions.
        self._normalization_attempts: dict[str, int] = {}
        # Last referenced experience index, preserved across the confirmation
        # fast-path so the conversation LLM keeps context on simple yes/no/ok
        # turns where data extraction is skipped. Reset on set_state().
        self._last_referenced_experience_index: int = -1

    async def _normalize_experience_titles(
        self,
        *,
        collected_data: list[CollectedData],
        target_uuids: Optional[set[str]] = None,
    ):
        if not self._search_services or self._state is None:
            return

        config = self._experience_pipeline_config or ExperiencePipelineConfig()
        infer_tool = InferOccupationTool(
            occupation_skill_search_service=self._search_services.occupation_skill_search_service,
            occupation_search_service=self._search_services.occupation_search_service
        )

        targets: list[CollectedData] = []
        tasks = []
        for elem in collected_data:
            if not elem.experience_title or elem.normalized_experience_title:
                continue
            if target_uuids is not None and elem.uuid not in target_uuids:
                continue
            if self._normalization_attempts.get(elem.uuid, 0) >= self._MAX_NORMALIZATION_ATTEMPTS:
                continue
            self._normalization_attempts[elem.uuid] = self._normalization_attempts.get(elem.uuid, 0) + 1
            targets.append(elem)
            tasks.append(infer_tool.execute(
                experience_title=elem.experience_title,
                company=elem.company,
                work_type=WorkType.from_string_key(elem.work_type),
                responsibilities=[],
                country_of_interest=self._state.country_of_user,
                number_of_titles=config.number_of_occupation_alt_titles,
                top_k=config.number_of_occupations_per_cluster,
                top_p=config.number_of_occupations_candidates_per_title
            ))

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for elem, result in zip(targets, results):
            if isinstance(result, Exception):
                self.logger.warning("Failed to infer normalized title for '%s': %s",
                                    elem.experience_title, result)
                continue
            normalized_title = _select_normalized_title(
                original_title=elem.experience_title,
                contextual_titles=result.contextual_titles,
                esco_occupations=result.esco_occupations
            )
            if normalized_title:
                elem.normalized_experience_title = normalized_title

    def _prune_stale_orphan_experiences(
        self,
        *,
        last_referenced_experience_index: int,
        current_turn_count: int,
    ) -> int:
        """
        Remove titleless experiences that were created in a previous turn.

        Such entries are data-extraction artifacts (a fleeting mention that never got
        named, or a phantom left behind because a later ADD created a sibling instead
        of updating it). They cannot be meaningfully completed and would otherwise
        block work-type transitions via _find_incomplete_required_for_work_type.

        Fresh titleless entries (defined_at_turn_number >= current turn count) are
        preserved so the conversation LLM can ask for a title in its response.

        Returns the (possibly re-mapped) last_referenced_experience_index.
        """
        collected_data = self._state.collected_data

        def _is_keepable(exp: CollectedData) -> bool:
            if exp.experience_title and exp.experience_title.strip():
                return True
            if exp.defined_at_turn_number is not None and exp.defined_at_turn_number >= current_turn_count:
                return True
            return False

        if all(_is_keepable(exp) for exp in collected_data):
            return last_referenced_experience_index

        last_ref_uuid = (
            collected_data[last_referenced_experience_index].uuid
            if 0 <= last_referenced_experience_index < len(collected_data)
            else None
        )
        pruned_count = sum(1 for exp in collected_data if not _is_keepable(exp))
        kept = [exp for exp in collected_data if _is_keepable(exp)]
        for new_idx, exp in enumerate(kept):
            exp.index = new_idx
        self._state.collected_data = kept
        self.logger.info(
            "Pruned %d stale orphan experience(s) without titles (current turn count=%d).",
            pruned_count, current_turn_count,
        )

        if last_ref_uuid is None:
            return -1
        for i, exp in enumerate(kept):
            if exp.uuid == last_ref_uuid:
                return i
        return -1

    def set_state(self, state: CollectExperiencesAgentState):
        """
        Set the state of the agent.
        This method should be called before the agent's execute() method is called.
        """
        self._state = state
        # Reset the per-experience normalization attempt counter so it does not
        # leak across sessions when the agent instance is reused by the director.
        self._normalization_attempts = {}
        self._last_referenced_experience_index = -1

    @staticmethod
    def _has_incomplete_required_fields_for_type(
        *,
        collected_data: list[CollectedData],
        exploring_type: WorkType | None,
    ) -> bool:
        """
        Required fields for transitioning a work type are title and work_type.
        If any experience of the current exploring type is missing these,
        we must keep collecting.
        """
        if exploring_type is None:
            return False
        key = exploring_type.name
        for exp in collected_data:
            if exp.work_type and exp.work_type.strip() == key:
                if not (exp.experience_title and exp.experience_title.strip()):
                    return True
                if not (exp.work_type and exp.work_type.strip()):
                    return True
        return False

    async def execute(self, user_input: AgentInput,
                      context: ConversationContext) -> AgentOutput:

        if self._state is None:
            raise ValueError("CollectExperiencesAgent: execute() called before state was initialized")

        collected_data = self._state.collected_data
        # Preserve the last referenced experience across confirmation turns where
        # data extraction is skipped. On full extraction this is overwritten by
        # the returned value below.
        last_referenced_experience_index = self._last_referenced_experience_index
        data_extraction_llm_stats = []
        newly_titled_uuids: list[str] = []

        # Determine if we are in the education phase
        is_education_phase = not self._state.education_phase_done

        if user_input.message == "":
            # If the user input is empty, set it to "(silence)"
            # This is to avoid the agent failing to respond to an empty input
            user_input.message = "(silence)"
        elif _is_simple_confirmation(user_input.message):
            # Fast-path: skip data extraction entirely on bare confirmations
            # ("yes", "no", "ok", "sawa", etc.). Saves ~750ms on the most common
            # turn type. The conversation LLM and transition tool still run, so
            # phase transitions and follow-up questions are unaffected. Trade-off:
            # a yes/no answer to a `paid_work` question is missed and the agent
            # will re-ask on a future turn. See _CONFIRMATION_TOKENS for details.
            self.logger.debug(
                "Confirmation fast-path: skipping data extraction for input %r",
                user_input.message,
            )
        else:
            # The data extraction LLM is responsible for extracting the experience data from the conversation
            data_extraction_llm = _DataExtractionLLM(self.logger)
            # TODO: the LLM can and will fail with an exception or even return None, we need to handle this
            (
                last_referenced_experience_index,
                data_extraction_llm_stats,
                newly_titled_uuids,
            ) = await data_extraction_llm.execute(user_input=user_input,
                                                   context=context,
                                                   collected_experience_data_so_far=collected_data)
            self._last_referenced_experience_index = last_referenced_experience_index
            # Tag education-phase entries with source="education"
            if is_education_phase:
                for elem in collected_data:
                    if elem.source is None:
                        elem.source = "education"

        # Prune stale orphan experiences — titleless entries defined in a previous turn.
        # Without this, a phantom (created when the user first mentions an experience but
        # never gets a title assigned — e.g. because a later ADD created a sibling entry
        # instead of updating it) traps TransitionDecisionTool in perpetual CONTINUE,
        # because _find_incomplete_required_for_work_type treats missing titles as a
        # blocker. Fresh titleless entries from this turn are kept — the conversation LLM
        # is presumed to be asking for the title right now.
        last_referenced_experience_index = self._prune_stale_orphan_experiences(
            last_referenced_experience_index=last_referenced_experience_index,
            current_turn_count=len(context.all_history.turns),
        )
        self._last_referenced_experience_index = last_referenced_experience_index
        collected_data = self._state.collected_data

        # TODO: Keep track of the last_referenced_experience_index and if it has changed it means that the user has
        #   provided a new experience, we need to handle this as
        #   a) if the user has not finished with the previous one we should ask them to complete it first
        #   b) the model may have made a mistake interpreting the user input as we need to clarify
        conversation_llm = _ConversationLLM()
        exploring_type = self._state.unexplored_types[0] if len(self._state.unexplored_types) > 0 else None

        transition_decision_tool = TransitionDecisionTool(self.logger)

        # Run normalization concurrently with the conversation+transition pair.
        # Normalization (~2.3s on expensive turns) only mutates `normalized_experience_title`
        # on the experience records; the conversation and transition LLMs build their
        # prompts synchronously before awaiting, so they see a stable snapshot of the
        # data. Downstream readers of `normalized_experience_title` use the
        # `normalized_experience_title or experience_title` pattern (None-tolerant),
        # so a write that lands mid-turn cannot corrupt their output.
        async def _maybe_normalize_titles():
            if newly_titled_uuids:
                await self._normalize_experience_titles(
                    collected_data=collected_data,
                    target_uuids=set(newly_titled_uuids),
                )

        # Conversation, transition, and normalization are all pure readers of
        # collected_data/context/user_input (or, in normalization's case, only
        # mutate fields the others tolerate as None) -- safe to parallelize.
        (
            conversation_llm_output,
            (transition_decision, transition_reasoning, transition_llm_stats),
            _normalization_done,
        ) = await asyncio.gather(
            conversation_llm.execute(
                first_time_visit=self._state.first_time_visit,
                is_education_phase=is_education_phase,
                context=context,
                user_input=user_input,
                country_of_user=self._state.country_of_user,
                persona_type=self._state.persona_type,
                collected_data=collected_data,
                last_referenced_experience_index=last_referenced_experience_index,
                exploring_type=exploring_type,
                unexplored_types=self._state.unexplored_types,
                explored_types=self._state.explored_types,
                logger=self.logger),
            transition_decision_tool.execute(
                collected_data=collected_data,
                exploring_type=exploring_type,
                unexplored_types=self._state.unexplored_types,
                explored_types=self._state.explored_types,
                conversation_context=context,
                user_input=user_input),
            _maybe_normalize_titles(),
        )

        self._state.first_time_visit = False

        conversation_llm_output.llm_stats = data_extraction_llm_stats + conversation_llm_output.llm_stats + transition_llm_stats
        reasoning_text = transition_reasoning.reasoning if transition_reasoning else "No reasoning provided"

        # Handle education phase transitions.
        # TransitionDecisionTool evaluates against the first unexplored WORK type
        # (FORMAL_SECTOR_WAGED_EMPLOYMENT) and is unaware of the education phase, so its
        # END_WORKTYPE / END_CONVERSATION verdicts are unreliable here and can fire on a
        # single partial education entry. The conversation LLM is prompted to emit
        # <END_OF_WORKTYPE> (surfaced as exploring_type_finished) only when the user
        # confirms they have no more education entries, so trust that signal exclusively.
        if is_education_phase and conversation_llm_output.exploring_type_finished:
            self._state.education_phase_done = True
            self.logger.info(
                "Education phase complete (transition=%s). Collected %d education entries. Transitioning to work type loop.",
                transition_decision.value,
                len([e for e in collected_data if e.source == "education"])
            )
            # Transition message to work types
            transition_text = t("messages", "collectExperiences.education.transitionToWork")
            next_exploring_type = self._state.unexplored_types[0] if self._state.unexplored_types else None
            if next_exploring_type is not None:
                next_type_description = _get_experience_type(next_exploring_type)
                work_type_text = await generate_bridge_to_work_type(
                    next_work_type_description=next_type_description,
                    last_agent_message=conversation_llm_output.message_for_user,
                    logger=self.logger,
                )
                if not work_type_text:
                    work_type_text = t(
                        'messages', 'collectExperiences.askAboutType',
                        experience_type=next_type_description,
                    )
                transition_text = f"{transition_text}\n\n{work_type_text}"
            conversation_llm_output.message_for_user = (
                f"{conversation_llm_output.message_for_user}\n\n{transition_text}"
            )
            conversation_llm_output.finished = False  # Don't end conversation, move to work types
            return conversation_llm_output

        # During the education phase the transition tool's verdicts are unreliable
        # (see comment above at lines 435-441) — only the conversation LLM's
        # <END_OF_WORKTYPE> signal can advance us out of education. If line 442
        # didn't fire we are still in education; coerce any END_WORKTYPE /
        # END_CONVERSATION verdict from the transition tool to CONTINUE so we
        # don't accidentally append a work-type bridge while the user is still
        # giving education entries.
        if is_education_phase and transition_decision in (
            TransitionDecision.END_WORKTYPE,
            TransitionDecision.END_CONVERSATION,
        ):
            self.logger.info(
                "Ignoring transition tool verdict %s during education phase; "
                "conversation LLM did not signal <END_OF_WORKTYPE>. Treating as CONTINUE.",
                transition_decision.value,
            )
            transition_decision = TransitionDecision.CONTINUE

        # Normal work type loop handling (existing logic)
        if (
            not is_education_phase
            and transition_decision == TransitionDecision.CONTINUE
            and conversation_llm_output.exploring_type_finished
            and not self._has_incomplete_required_fields_for_type(
                collected_data=collected_data,
                exploring_type=exploring_type,
            )
        ):
            self.logger.info(
                "Conversation LLM signaled END_OF_WORKTYPE while transition tool returned CONTINUE; "
                "overriding decision to END_WORKTYPE for exploring_type=%s.",
                exploring_type.name if exploring_type else "None",
            )
            transition_decision = TransitionDecision.END_WORKTYPE

        if transition_decision == TransitionDecision.END_WORKTYPE:
            did_update = False
            # if decision is to end the exploration of the current work type, we update null fields to ""
            if exploring_type is not None and exploring_type in self._state.unexplored_types:
                fill_incomplete_fields_as_declined(
                    self._state.collected_data, exploring_type
                )
                self._state.unexplored_types.remove(exploring_type)
                self._state.explored_types.append(exploring_type)
                did_update = True
                self.logger.info(
                    "Transition decision: END_WORKTYPE - Explored work type: %s"
                    "\n  - remaining types: %s"
                    "\n  - discovered experiences so far: %s"
                    "\n  - reasoning: %s",
                    exploring_type,
                    self._state.unexplored_types,
                    self._state.collected_data,
                    reasoning_text
                )
            # exit if no unexplored types left
            if not did_update and not self._state.unexplored_types:
                conversation_llm_output.finished = True
                self.logger.info(
                    "Transition decision: END_WORKTYPE with no unexplored types - treating as END_CONVERSATION"
                )
                return conversation_llm_output

            next_exploring_type = self._state.unexplored_types[0] if self._state.unexplored_types else None
            if next_exploring_type is not None:
                next_type_description = _get_experience_type(next_exploring_type)
                transition_text = await generate_bridge_to_work_type(
                    next_work_type_description=next_type_description,
                    last_agent_message=conversation_llm_output.message_for_user,
                    logger=self.logger,
                )
                if not transition_text:
                    transition_text = t(
                        'messages', 'collectExperiences.askAboutType',
                        experience_type=next_type_description,
                    )
            else:
                transition_text = t('messages', 'collectExperiences.recapPrompt')

            conversation_llm_output.message_for_user = (
                f"{conversation_llm_output.message_for_user}\n\n{transition_text}"
            )

            # If we just closed the last remaining type, finish collection now so
            # the director can transition to the dive-in phase next turn.
            if not self._state.unexplored_types:
                conversation_llm_output.finished = True
                self.logger.info(
                    "All work types explored after END_WORKTYPE; finishing collect phase."
                )

        elif transition_decision == TransitionDecision.END_CONVERSATION:
            conversation_llm_output.finished = True
            self.logger.info(
                "Transition decision: END_CONVERSATION"
                "\n  - all work types explored: %s"
                "\n  - discovered experiences: %s"
                "\n  - reasoning: %s",
                len(self._state.explored_types) == 4,
                self._state.collected_data,
                reasoning_text
            )
        elif transition_decision == TransitionDecision.CONTINUE:
            self.logger.info(
                "Transition decision: CONTINUE"
                "\n  - exploring type: %s"
                "\n  - unexplored types: %s"
                "\n  - collected experiences: %d"
                "\n  - reasoning: %s",
                exploring_type.name if exploring_type else "None",
                [wt.name for wt in self._state.unexplored_types],
                len(collected_data),
                reasoning_text
            )

        return conversation_llm_output

    def get_experiences(self) -> list[ExperienceEntity]:
        """
        Get the experiences extracted by the agent.
        If this method is called before the agent has finished its task, the list will be empty or incomplete.
        :return:
        """
        experiences = []
        # The conversation is completed when the LLM has finished and all work types have been explored
        for elem in self._state.collected_data:
            self.logger.debug("Experience data collected: %s", elem)
            try:
                entity = ExperienceEntity(
                    uuid=elem.uuid if elem.uuid else None,
                    experience_title=elem.experience_title if elem.experience_title else '',
                    normalized_experience_title=elem.normalized_experience_title,
                    company=elem.company,
                    location=elem.location,
                    timeline=Timeline(start=elem.start_date, end=elem.end_date),
                    work_type=WorkType.from_string_key(elem.work_type),
                    responsibilities=ResponsibilitiesData(responsibilities=elem.responsibilities),
                    source=elem.source,
                )
                experiences.append(entity)
            except Exception as e:  # pylint: disable=broad-except
                self.logger.warning("Could not parse experience entity from: %s. Error: %s", elem, e)

        return experiences
