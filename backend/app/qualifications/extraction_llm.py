import logging
import re
from textwrap import dedent
from typing import Optional

from pydantic import BaseModel, Field

from app.agent.llm_caller import LLMCaller
from app.agent.penalty import get_penalty
from app.agent.prompt_template import sanitize_input
from common_libs.llm.generative_models import GeminiGenerativeLLM
from common_libs.llm.models_utils import LLMConfig, JSON_GENERATION_CONFIG, ZERO_TEMPERATURE_GENERATION_CONFIG
from common_libs.retry import Retry
from app.qualifications.types import QualificationEntity, QualificationType

_TRIGGER_PATTERN = re.compile(
    r"\b(certificate|diploma|degree|license|licence|qualified|certif|trained|certified|"
    r"NITA|KNEC|grade\s+[IVX1-3]+|cheti|shahada|course|artisan|trade\s+test|"
    r"BSc|MSc|PhD|BA|BEd|BCom|HND|CPA|ACCA|CIPS)\b",
    re.IGNORECASE,
)

_TAGS_TO_FILTER = ["System Instructions", "Conversation History", "User Input"]


class _DetectedQualificationsResponse(BaseModel):
    qualifications: list[dict] = Field(default_factory=list)

    class Config:
        extra = "allow"


_SYSTEM_INSTRUCTIONS = dedent("""\
    <System Instructions>
    You are an expert at detecting qualifications mentioned in conversations.

    Task: From the user's message, extract any qualifications mentioned (certificates, diplomas, degrees, licenses, trade tests).
    Only extract qualifications with HIGH confidence. Do not over-extract casual mentions.

    Qualification types: CERTIFICATE, DIPLOMA, DEGREE, TRADE_LICENSE, PROFESSIONAL_LICENSE, TRAINING_COMPLETION, OTHER

    Kenya-specific: Recognize NITA trade tests, KNEC certificates, Grade I/II/III artisan qualifications.
    Swahili terms: "cheti" = certificate, "shahada" = degree/diploma.

    JSON Output:
    {{"qualifications": [
      {{
        "name": "string (required - full qualification name)",
        "qualification_type": "string (one of the types above)",
        "institution": "string or null",
        "date_obtained": "string or null",
        "field_of_study": "string or null",
        "level": "string or null"
      }}
    ]}}

    If no qualifications are mentioned, return {{"qualifications": []}}.
    No prose. JSON only.
    </System Instructions>
""")


class QualificationsDetector:
    """Lightweight LLM-based detector for qualifications mentioned in conversation turns."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._llm_caller = LLMCaller[_DetectedQualificationsResponse](
            model_response_type=_DetectedQualificationsResponse
        )

    def _should_run(self, user_message: str) -> bool:
        """Quick regex pre-filter to avoid LLM calls when no qualification terms are present."""
        return bool(_TRIGGER_PATTERN.search(user_message))

    async def detect(self, user_message: str, existing_uuids: set[str]) -> list[QualificationEntity]:
        """Detect qualifications mentioned in a single user message.

        Returns newly detected qualifications not already tracked.
        Skips the LLM call if no trigger words are found.
        """
        if not user_message or not self._should_run(user_message):
            return []

        prompt = dedent(f"""
            <User Input>
            {sanitize_input(user_message, _TAGS_TO_FILTER)}
            </User Input>
        """)

        async def _callback(attempt: int, max_retries: int) -> tuple[list[QualificationEntity], float, BaseException | None]:
            llm = GeminiGenerativeLLM(
                system_instructions=_SYSTEM_INSTRUCTIONS,
                config=LLMConfig(
                    generation_config=ZERO_TEMPERATURE_GENERATION_CONFIG | JSON_GENERATION_CONFIG | {"max_output_tokens": 512}
                )
            )
            try:
                model_response, _ = await self._llm_caller.call_llm(
                    llm=llm, llm_input=prompt, logger=self._logger
                )
            except Exception as e:
                return [], get_penalty(1), e

            if not model_response or not model_response.qualifications:
                return [], 0.0, None

            results = []
            for q_dict in model_response.qualifications:
                try:
                    # Normalize qualification_type to a valid enum key, defaulting to OTHER
                    raw_type = str(q_dict.get("qualification_type", "OTHER")).upper()
                    try:
                        q_type = QualificationType[raw_type]
                    except KeyError:
                        q_type = QualificationType.OTHER

                    entity = QualificationEntity(
                        qualification_type=q_type,
                        name=q_dict.get("name", ""),
                        institution=q_dict.get("institution"),
                        date_obtained=q_dict.get("date_obtained"),
                        field_of_study=q_dict.get("field_of_study"),
                        level=q_dict.get("level"),
                        source="conversation",
                    )
                    if entity.name and entity.uuid not in existing_uuids:
                        results.append(entity)
                except Exception as parse_err:
                    self._logger.warning("Failed to parse qualification dict: %s — %s", q_dict, parse_err)

            return results, 0.0, None

        detected, _penalty, _error = await Retry[list[QualificationEntity]].call_with_penalty(
            callback=_callback, logger=self._logger
        )
        if detected:
            self._logger.info("Qualifications detected {count=%s}", len(detected))
        return detected or []
