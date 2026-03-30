import logging
from textwrap import dedent
from typing import Optional

from pydantic import BaseModel

from app.agent.llm_caller import LLMCaller
from app.agent.penalty import get_penalty
from app.agent.prompt_template import sanitize_input
from app.agent.experience.work_type import WORK_TYPE_DEFINITIONS_FOR_PROMPT
from common_libs.llm.generative_models import GeminiGenerativeLLM
from common_libs.llm.models_utils import LLMConfig, JSON_GENERATION_CONFIG, get_config_variation
from common_libs.retry import Retry
from app.users.cv.types import CVExtractedExperience, CVExtractedQualification, CVStructuredExtractionResponse

_TAGS_TO_FILTER = [
    "CV Markdown",
    "System Instructions",
    "User's Last Input",
    "Conversation History",
]

_QUALIFICATION_TYPE_KEYS = (
    "CERTIFICATE, DIPLOMA, DEGREE, TRADE_LICENSE, PROFESSIONAL_LICENSE, TRAINING_COMPLETION, OTHER"
)


class CVStructuredExtractor:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._llm_caller: LLMCaller[CVStructuredExtractionResponse] = LLMCaller[CVStructuredExtractionResponse](
            model_response_type=CVStructuredExtractionResponse
        )
        self._penalty_level = 1

    @staticmethod
    def _prompt(markdown_cv: str) -> str:
        clean_md = sanitize_input(markdown_cv, _TAGS_TO_FILTER)
        return dedent("""
            <CV Markdown>
            {markdown}
            </CV Markdown>
        """).format(markdown=clean_md)

    @staticmethod
    def _json_system_instructions() -> str:
        return dedent("""
            <System Instructions>
            You are an expert CV parser that extracts structured work experience and qualification data.

            Task: From the provided <CV Markdown>, extract ALL work/livelihood experiences AND qualifications as structured JSON.

            JSON Output Schema (must strictly follow):
            {{
              "experiences": [
                {{
                  "experience_title": "string (required - job title or role)",
                  "company": "string or null (company/organization name)",
                  "location": "string or null (city, region, or country)",
                  "start_date": "string or null (year or month/year)",
                  "end_date": "string or null (year, month/year, or 'Present')",
                  "work_type": "string or null (one of the work type keys below)",
                  "responsibilities": ["string", ...]
                }}
              ],
              "qualifications": [
                {{
                  "name": "string (required - full qualification name)",
                  "qualification_type": "string (one of: {qualification_type_keys})",
                  "institution": "string or null (awarding institution)",
                  "date_obtained": "string or null (year or month/year)",
                  "field_of_study": "string or null",
                  "level": "string or null (e.g. Grade I, Level 3, NQF 4)"
                }}
              ]
            }}

            Work Type Classification (use the .name key):
            {work_type_definitions}

            If the work type is unclear, default to "FORMAL_SECTOR_WAGED_EMPLOYMENT".

            Qualification Type Classification:
            - CERTIFICATE: Professional certificates (CompTIA, CCNA, NITA trade test certificates, KNEC certificates)
            - DIPLOMA: Diplomas (Diploma in Nursing, Business Administration)
            - DEGREE: University degrees (BSc, MSc, PhD, BA)
            - TRADE_LICENSE: Artisan/trade licenses (electrician license, plumber license, Grade I/II/III trade tests)
            - PROFESSIONAL_LICENSE: Professional licenses (CPA, nursing license, teaching license)
            - TRAINING_COMPLETION: Training program completions
            - OTHER: Anything not fitting above categories

            Kenya-specific: Recognize NITA (National Industrial Training Authority) trade tests, KNEC certificates,
            Grade I/II/III artisan qualifications, and Swahili qualification names (e.g., "Cheti cha..." = Certificate of).

            Rules:
            - Extract EVERY distinct work experience (unique by role + company + timeframe)
            - Extract EVERY distinct qualification listed in the CV
            - Skip completely duplicated entries
            - Do NOT include personal data: no person names, emails, phone numbers, street addresses, or profile links
            - Company/organization names and city/country locations ARE allowed
            - Each experience must have at minimum an experience_title; each qualification must have at minimum a name
            - Responsibilities should be concise bullet-point descriptions of what the person did
            - If the CV has no discernible experiences, return {{"experiences": []}}
            - If the CV has no discernible qualifications, return {{"qualifications": []}}
            - No prose outside the JSON. Respond with JSON only.
            </System Instructions>
        """).format(
            work_type_definitions=WORK_TYPE_DEFINITIONS_FOR_PROMPT,
            qualification_type_keys=_QUALIFICATION_TYPE_KEYS,
        )

    async def extract_structured(self, markdown_cv: str) -> tuple[list[CVExtractedExperience], list[CVExtractedQualification]]:
        """Extract structured experiences and qualifications from CV markdown.

        Returns a tuple of (experiences, qualifications).
        """
        self._logger.info("Extracting structured data from markdown {md_length_chars=%s}", len(markdown_cv or ""))
        prompt = self._prompt((markdown_cv or "").strip())

        _result: CVStructuredExtractionResponse | None = None

        async def _callback(attempt: int, max_retries: int) -> tuple[CVStructuredExtractionResponse, float, BaseException | None]:
            temperature_cfg = get_config_variation(
                start_temperature=0.0, end_temperature=0.3,
                start_top_p=0.9, end_top_p=1.0,
                attempt=attempt, max_retries=max_retries
            )
            llm = GeminiGenerativeLLM(
                system_instructions=self._json_system_instructions(),
                config=LLMConfig(
                    generation_config=temperature_cfg | JSON_GENERATION_CONFIG | {
                        "max_output_tokens": 4096
                    }
                )
            )
            try:
                model_response, _ = await self._llm_caller.call_llm(
                    llm=llm, llm_input=prompt, logger=self._logger
                )
            except Exception as e:
                return CVStructuredExtractionResponse(), get_penalty(self._penalty_level), e

            if not model_response:
                return CVStructuredExtractionResponse(), get_penalty(self._penalty_level), ValueError("LLM returned no model response")

            if not model_response.experiences and not model_response.qualifications:
                return CVStructuredExtractionResponse(), get_penalty(self._penalty_level), ValueError("LLM returned empty experiences and qualifications")

            return model_response, 0.0, None

        result, _penalty, _error = await Retry[CVStructuredExtractionResponse].call_with_penalty(
            callback=_callback, logger=self._logger
        )

        experiences = result.experiences if result else []
        qualifications = result.qualifications if result else []

        self._logger.info(
            "Structured extraction complete {experiences=%s, qualifications=%s}",
            len(experiences), len(qualifications)
        )
        return experiences, qualifications
