import logging

from app.agent.collect_experiences_agent._types import CollectedData
from app.agent.experience.work_type import WorkType
from app.users.cv.types import CVExtractedExperience

logger = logging.getLogger(__name__)


def map_cv_to_collected_data(
    cv_experiences: list[CVExtractedExperience],
    existing_data: list[CollectedData],
) -> list[CollectedData]:
    """
    Convert structured CV experiences to CollectedData,
    deduplicating against already-collected conversational data.

    Returns only NEW (non-duplicate) CollectedData items.
    """
    new_items: list[CollectedData] = []
    next_index = len(existing_data)

    for cv_exp in cv_experiences:
        # Map work_type string to valid WorkType enum name, clearing if invalid
        work_type_name = cv_exp.work_type
        wt = WorkType.from_string_key(work_type_name) if work_type_name else None
        if work_type_name and wt is None:
            logger.warning(
                "cv_to_agent_mapper: unrecognised work_type '%s' for experience '%s', setting to None",
                work_type_name, cv_exp.experience_title
            )
            work_type_name = None

        # Infer paid_work from work_type
        paid_work = None
        if wt is not None:
            if wt == WorkType.FORMAL_SECTOR_WAGED_EMPLOYMENT:
                paid_work = True
            elif wt == WorkType.UNSEEN_UNPAID:
                paid_work = False
            elif wt == WorkType.FORMAL_SECTOR_UNPAID_TRAINEE_WORK:
                paid_work = False

        candidate = CollectedData(
            index=next_index + len(new_items),
            defined_at_turn_number=0,
            experience_title=cv_exp.experience_title,
            company=cv_exp.company,
            location=cv_exp.location,
            start_date=cv_exp.start_date,
            end_date=cv_exp.end_date,
            paid_work=paid_work,
            work_type=work_type_name,
            source="cv",
            responsibilities=cv_exp.responsibilities,
        )

        # Deduplicate against existing + already-mapped
        is_duplicate = any(
            CollectedData.compare_relaxed(candidate, existing)
            for existing in [*existing_data, *new_items]
        )
        if not is_duplicate:
            new_items.append(candidate)

    return new_items
