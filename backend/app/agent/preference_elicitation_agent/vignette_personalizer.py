"""
Vignette Personalizer for Preference Elicitation Agent.

Generates personalized vignettes based on user context and vignette templates.
"""

import logging
import json
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from app.agent.preference_elicitation_agent.types import (
    UserContext,
    VignetteTemplate,
    PersonalizedVignette,
    Vignette,
    VignetteOption
)
from app.agent.llm_caller import LLMCaller
from common_libs.llm.models_utils import (
    BasicLLM,
    LLMConfig,
    LOW_TEMPERATURE_GENERATION_CONFIG,
    JSON_GENERATION_CONFIG
)


class GeneratedVignetteContent(BaseModel):
    """LLM response model for generated vignette content."""

    scenario_intro: str = Field(
        description="Brief framing for the choice (1-2 sentences). Neutral — does not hint at which option is 'better'."
    )

    option_a_description: str = Field(
        description="Lived-experience narrative for option A (2-4 sentences). No job titles, no company names."
    )

    option_b_description: str = Field(
        description="Lived-experience narrative for option B (2-4 sentences). No job titles, no company names."
    )

    reasoning: str = Field(
        default="",
        description="Brief explanation of how this was personalized (for debugging)"
    )


class VignettePersonalizer:
    """
    Personalizes vignette templates to match user's background.

    Uses LLM to generate job scenarios relevant to the user's role,
    industry, and experience level while maintaining the core trade-offs
    defined in the template.
    """

    def __init__(self, llm: BasicLLM, templates_config_path: Optional[str] = None):
        """
        Initialize the VignettePersonalizer.

        Args:
            llm: Language model to use for generation (base LLM without system instructions)
            templates_config_path: Path to vignette templates JSON file
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._base_llm = llm
        self._templates: list[VignetteTemplate] = []
        self._templates_by_id: dict[str, VignetteTemplate] = {}
        self._templates_by_category: dict[str, list[VignetteTemplate]] = {}

        # Create LLM with system instructions for vignette generation
        from common_libs.llm.generative_models import GeminiGenerativeLLM

        system_instructions = """
You are helping write career preference questions for Kenyan youth.

Your task: Generate TWO short job-scenario descriptions (Option A and Option B) that:
1. Test the trade-off specified in the template
2. Feel grounded in everyday Kenyan working life
3. Read as lived experience, not as job ads
4. Create a real dilemma — neither option is obviously better

CRITICAL — DO NOT NAME THE JOB
==============================
Never write a job title, role name, or occupation (no "Sales Manager",
"Developer", "Teacher", "Driver", etc.).
Never name a specific company (no "Safaricom", "Equity Bank", "M-PESA",
no startup names).
Never label the option with the trade-off it represents (no "The Stable
Path", "The Risky Bet", "The Flexible Option"). The label is just A or B
in the rendered UI — your job is to describe what the day-to-day
actually feels like and let the user infer the trade-off themselves.

WRITE LIVED EXPERIENCE
======================
Describe what the person's week looks like — concretely. Anchor on:
- Money: monthly take-home in KES, how predictable it is, whether
  benefits (NHIF, NSSF, leave) are part of the deal
- Time: hours per week, when they happen, whether weekends are off
- Place: office, field, home, mixed; commute reality
- Autonomy: who tells them what to do, how much is theirs to decide
- Pace and pressure: steady vs spiky, who carries the risk when things
  go wrong
- Growth: what gets better over a year of doing this

Use second person ("you'd...") consistently across both options. Keep
it to 2-4 sentences.

SHOW, DON'T TELL
================
Anchor the description in concrete moments, not abstract job traits.
Pick one or two specific images per option — what does Tuesday morning
actually look like? What does the person see, hear, or do?

Weak (abstract): "Your days involve interacting with many different
people, often explaining technical details clearly."
Strong (concrete): "Most mornings start with someone at your desk
looking confused — you talk them through it until their face clears."

Weak: "It's physically demanding, often requiring you to be on your
feet and handle heavy tasks."
Strong: "By midday your shirt is sticking to your back; your hands
know the weight of the work without you thinking about it."

Keep it tight — one or two images, then back to the trade-off. Do not
over-write or get poetic; the goal is to put the reader inside the
moment, not to perform.

HARD RULE: At least ONE sentence per option must be a specific scene
— a moment in time, an action in progress, something the person sees
or does. The other sentences can summarize. A description with zero
scenes (only patterns and traits) is a failure, regardless of how
accurate it is.

ENFORCE A REAL TRADE-OFF
========================
Each option must have clear advantages AND clear sacrifices. The user
should feel conflicted.

Salary balance rules:
- The more stable option should pay 20-40% more on average than the
  less stable one
- The riskier option should have a meaningfully higher ceiling
  (50-100% above the stable ceiling) to compensate for the downside
- Never make the "exciting" option also pay more guaranteed — that's
  unrealistic

KENYAN CONTEXT
==============
- Salaries in KES per month, realistic ranges (15K-180K depending on
  level)
- Local texture where it fits naturally: matatu commutes, NHIF/NSSF,
  M-PESA payments, upcountry travel — but only if it serves the
  description, not as decoration. Do NOT mention specific cities or
  locations.

OUTPUT SCHEMA
=============
Return a JSON object with exactly these fields:
- scenario_intro (string): 1-2 sentence neutral framing
- option_a_description (string): 2-4 sentence lived-experience narrative
- option_b_description (string): 2-4 sentence lived-experience narrative
- reasoning (string): brief note on how this was personalized

EXAMPLE OUTPUT
==============
{
  "scenario_intro": "Two paths are open to you, with very different rhythms.",
  "option_a_description": "Your salary is KES 95,000 every month, on the same day, with NHIF, NSSF, and pension deductions already handled. You're in an office from 8 to 5, Monday to Friday — the commute eats two hours of your day. The work is steady and the people above you decide most of what you'll do; you'll get a small raise next year and a slightly bigger one the year after.",
  "option_b_description": "Some months you clear KES 140,000, other months KES 45,000 — it depends on what you bring in. You set your own schedule and work from wherever; nobody asks where you were on Tuesday. There are no benefits, no safety net, and the slow stretches are quietly stressful. But when something works, the upside is yours.",
  "reasoning": "Personalized for a mid-level professional. Neither option names a role; the trade-off (predictability vs upside) emerges from the texture, not from a label."
}
"""

        # Use proper JSON generation config
        llm_config = LLMConfig(
            generation_config=LOW_TEMPERATURE_GENERATION_CONFIG | JSON_GENERATION_CONFIG
        )

        self._llm = GeminiGenerativeLLM(
            system_instructions=system_instructions,
            config=llm_config
        )

        # Load templates from config
        if templates_config_path is None:
            # Default path relative to backend root
            config_dir = Path(__file__).parent.parent.parent / "config"
            templates_config_path = str(config_dir / "vignette_templates.json")

        self._load_templates(templates_config_path)

    def _load_templates(self, config_path: str) -> None:
        """
        Load vignette templates from JSON configuration file.

        Args:
            config_path: Path to templates configuration file
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                templates_data = json.load(f)

            for template_data in templates_data:
                template = VignetteTemplate(**template_data)
                self._templates.append(template)
                self._templates_by_id[template.template_id] = template

                # Index by category
                category = template.category
                if category not in self._templates_by_category:
                    self._templates_by_category[category] = []
                self._templates_by_category[category].append(template)

            self._logger.info(f"Loaded {len(self._templates)} templates from {config_path}")

        except FileNotFoundError:
            self._logger.error(f"Templates config file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            self._logger.error(f"Invalid JSON in templates config: {e}")
            raise
        except Exception as e:
            self._logger.error(f"Error loading templates: {e}")
            raise

    def get_templates_by_category(self, category: str) -> list[VignetteTemplate]:
        """
        Get all templates for a specific category.

        Args:
            category: Category name (e.g., "financial", "work_environment")

        Returns:
            List of templates in that category
        """
        return self._templates_by_category.get(category, [])

    def get_template_by_id(self, template_id: str) -> Optional[VignetteTemplate]:
        """
        Get a specific template by ID.

        Args:
            template_id: Unique identifier for the template

        Returns:
            VignetteTemplate or None if not found
        """
        return self._templates_by_id.get(template_id)

    async def personalize_vignette(
        self,
        template: VignetteTemplate,
        user_context: UserContext,
        previous_vignettes: Optional[list[str]] = None
    ) -> PersonalizedVignette:
        """
        Generate a personalized vignette from a template.

        Args:
            template: The vignette template to personalize
            user_context: User's background context
            previous_vignettes: List of previous vignette scenarios (to avoid repetition)

        Returns:
            PersonalizedVignette with generated content
        """
        # Generate personalized content using LLM
        generated = await self._generate_vignette_content(
            template=template,
            user_context=user_context,
            previous_vignettes=previous_vignettes or []
        )

        # Build VignetteOption objects (titles are no longer rendered;
        # neutral A/B labels are applied at presentation time)
        option_a = VignetteOption(
            option_id="A",
            title="Option A",
            description=generated.option_a_description,
            attributes=template.option_a  # Use template attributes
        )

        option_b = VignetteOption(
            option_id="B",
            title="Option B",
            description=generated.option_b_description,
            attributes=template.option_b  # Use template attributes
        )

        # Build personalized Vignette with unique ID per generation
        # Use template_id + a unique suffix to ensure each generated vignette is tracked separately
        unique_suffix = str(uuid.uuid4())[:8]
        vignette = Vignette(
            vignette_id=f"{template.template_id}_{unique_suffix}",
            category=template.category,
            scenario_text=generated.scenario_intro,
            options=[option_a, option_b],
            follow_up_questions=template.follow_up_prompts,
            targeted_dimensions=template.targeted_dimensions,
            difficulty_level=template.difficulty_level
        )

        return PersonalizedVignette(
            template_id=template.template_id,
            vignette=vignette,
            generation_context={
                "user_role": user_context.current_role,
                "user_industry": user_context.industry,
                "user_level": user_context.experience_level,
                "reasoning": generated.reasoning
            }
        )

    async def _generate_vignette_content(
        self,
        template: VignetteTemplate,
        user_context: UserContext,
        previous_vignettes: list[str]
    ) -> GeneratedVignetteContent:
        """
        Call LLM to generate personalized vignette content.

        Args:
            template: The template defining trade-offs
            user_context: User's background
            previous_vignettes: Previously shown scenarios

        Returns:
            Generated vignette content
        """
        # Build context description (pass previous_vignettes so the background
        # selection instruction can reference what has already been used)
        user_background = self._format_user_context(user_context, previous_vignettes)

        # Build template description
        template_description = self._format_template(template)

        # Build previous vignettes list
        previous_desc = (
            "\n".join(f"- {v}" for v in previous_vignettes)
            if previous_vignettes
            else "None yet"
        )

        prompt = f"""
**User Background:**
{user_background}

**Template Trade-Off:**
{template_description}

**Previously Shown Scenarios (MUST be VERY different — also use these to identify which backgrounds have already been used):**
{previous_desc}

**CRITICAL REQUIREMENTS:**
1. The job scenarios you generate MUST be SUBSTANTIALLY DIFFERENT from all previous scenarios
2. If previous scenarios used freelance/permanent contrast, use a DIFFERENT contrast (startup vs corporate, field work vs office, client-facing vs backend)
3. If previous scenarios used a specific industry/company, use a COMPLETELY DIFFERENT industry/company
4. Vary the job settings: if previous were tech/office jobs, try field work, retail, healthcare, education, etc.
5. The trade-off dimensions from the template are what matter - not the specific job types

Generate a personalized vignette that:
- Tests the TEMPLATE'S trade-off dimensions
- Uses job scenarios VERY DIFFERENT from previous ones
- Feels relevant to user's skills but in a different context
"""

        caller = LLMCaller[GeneratedVignetteContent](
            model_response_type=GeneratedVignetteContent
        )

        response, _ = await caller.call_llm(
            llm=self._llm,
            llm_input=prompt,
            logger=self._logger
        )

        if response is None:
            # Failed to generate, raise error to be caught by caller
            raise Exception("Failed to generate vignette content")

        return response

    def _format_user_context(
        self,
        context: UserContext,
        previous_vignettes: Optional[list[str]] = None
    ) -> str:
        """Format user context for LLM prompt."""
        parts = []

        if context.current_role:
            parts.append(f"Most Recent Role: {context.current_role}")
        else:
            parts.append("Most Recent Role: Not specified (use general junior-level roles)")

        if context.industry:
            parts.append(f"Most Recent Industry: {context.industry}")
        else:
            parts.append("Most Recent Industry: Not specified (use general industries)")

        parts.append(f"Experience Level: {context.experience_level}")

        if context.background_summary:
            parts.append(f"Summary: {context.background_summary}")

        if context.all_backgrounds and len(context.all_backgrounds) > 1:
            parts.append("")
            parts.append("All Past Experience Contexts (Role | Industry):")
            for bg in context.all_backgrounds:
                parts.append(f"  - {bg}")
            parts.append("")
            parts.append(
                "Use any of these backgrounds to create relevant job scenarios. "
                "Scenarios should feel realistic for someone with this overall professional experience."
            )

        return "\n".join(parts)

    def _format_template(self, template: VignetteTemplate) -> str:
        """Format template trade-off for LLM prompt."""
        parts = [
            f"Category: {template.category}",
            f"Testing: {template.trade_off.get('dimension_a', 'unknown')} vs {template.trade_off.get('dimension_b', 'unknown')}",
            "",
            "Option A should have:",
            f"- High: {', '.join(template.option_a.get('high_dimensions', []))}",
            f"- Low: {', '.join(template.option_a.get('low_dimensions', []))}",
            f"- Salary range: {template.option_a.get('salary_range', [])} KES/month",
            "",
            "Option B should have:",
            f"- High: {', '.join(template.option_b.get('high_dimensions', []))}",
            f"- Low: {', '.join(template.option_b.get('low_dimensions', []))}",
            f"- Salary range: {template.option_b.get('salary_range', [])} KES/month",
        ]

        return "\n".join(parts)

    def get_total_templates_count(self) -> int:
        """Get total number of available templates."""
        return len(self._templates)

    def get_category_counts(self) -> dict[str, int]:
        """Get count of templates per category."""
        return {
            category: len(templates)
            for category, templates in self._templates_by_category.items()
        }

    async def personalize_concrete_vignette(
        self,
        vignette: Vignette,
        user_context: UserContext
    ) -> tuple[Vignette, "PersonalizationLog"]:
        """
        Personalize a concrete vignette (from offline optimization)
        while preserving exact attribute values.

        Takes a vignette with fixed attributes and generates personalized
        text descriptions that match the user's background, without changing
        any of the numerical/categorical attribute values.

        Args:
            vignette: Concrete vignette from offline optimization
            user_context: User's background context

        Returns:
            Tuple of (personalized_vignette, personalization_log)

        Raises:
            Exception: If personalization fails after retries (handled by LLMCaller)
        """
        from app.agent.preference_elicitation_agent.types import PersonalizationLog

        # Store original values for logging
        original_data = {
            "scenario_text": vignette.scenario_text,
            "option_a_description": vignette.options[0].description,
            "option_b_description": vignette.options[1].description
        }

        # Extract trade-off from attribute differences
        trade_off_desc = self._extract_trade_off_from_vignette(vignette)

        # Build personalization prompt
        prompt = f"""You are personalizing a career preference question for a Kenyan youth.

**User Background:**
{self._format_user_context(user_context)}

**Vignette Attributes (DO NOT CHANGE THESE VALUES):**
Option A: {self._format_attributes(vignette.options[0].attributes)}
Option B: {self._format_attributes(vignette.options[1].attributes)}

**Trade-Off Being Tested (for your reference only — DO NOT name it in the output):**
{trade_off_desc}

**Your Task:**
Write two short lived-experience descriptions that:
1. Feel grounded in the user's working life
2. Maintain the EXACT trade-off shown in the attributes above
3. Make both options feel like real, attractive choices (real dilemma)
4. Do NOT name a job title, occupation, or company
5. Do NOT label the option with the trade-off it represents

**CRITICAL:**
- DO NOT invent different attribute values
- The descriptions must MATCH the attributes exactly
- If wage=25000 in attributes, the description must surface "KES 25,000/month"
- If physical_demand=1, the description must convey high physical demands
- If flexibility=0, the description must convey fixed schedules

**Example of Good Personalization:**
Attributes: wage=30000, flexibility=1, remote_work=1
✓ Description: "You set your own hours and work from wherever — some days a café, some days from bed. KES 30,000 lands in your account at the end of each month, but there are no benefits and no one above you to catch you when work dries up."

**Example of Bad Personalization (inventing different values):**
❌ Description: "Earn KES 50,000/month..." (wage was 30000!)
❌ Description: "Fixed 9-5 schedule" (flexibility was 1!)

Generate personalized content now:
"""

        caller = LLMCaller[GeneratedVignetteContent](
            model_response_type=GeneratedVignetteContent
        )

        response, _ = await caller.call_llm(
            llm=self._llm,
            llm_input=prompt,
            logger=self._logger
        )

        personalization_log = PersonalizationLog(
            vignette_id=vignette.vignette_id,
            original=original_data,
            user_context={
                "role": user_context.current_role,
                "industry": user_context.industry,
                "experience_level": user_context.experience_level
            }
        )

        if response is None:
            # Personalization failed after retries - use original vignette
            self._logger.warning(
                f"Personalization failed for vignette {vignette.vignette_id}, using original text"
            )
            personalization_log.personalization_successful = False
            personalization_log.error_message = "LLM personalization failed after retries"
            personalization_log.attributes_preserved = True  # No changes made
            return vignette, personalization_log

        # Create personalized vignette with new text but same attributes.
        # Titles are preserved from the original (offline) vignette since they
        # are no longer rendered to users; neutral A/B labels are applied at
        # presentation time.
        personalized_options = []
        for original_opt, new_desc in [
            (vignette.options[0], response.option_a_description),
            (vignette.options[1], response.option_b_description)
        ]:
            personalized_opt = VignetteOption(
                option_id=original_opt.option_id,
                title=original_opt.title,
                description=new_desc,
                attributes=original_opt.attributes.copy()  # Preserve exact attributes
            )
            personalized_options.append(personalized_opt)

        personalized_vignette = Vignette(
            vignette_id=vignette.vignette_id,
            category=vignette.category,
            scenario_text=response.scenario_intro,
            options=personalized_options,
            follow_up_questions=vignette.follow_up_questions,
            targeted_dimensions=vignette.targeted_dimensions,
            difficulty_level=vignette.difficulty_level
        )

        # Validate attributes were preserved
        attrs_preserved = (
            personalized_vignette.options[0].attributes == vignette.options[0].attributes and
            personalized_vignette.options[1].attributes == vignette.options[1].attributes
        )

        if not attrs_preserved:
            self._logger.error(
                f"CRITICAL: Attributes changed during personalization for {vignette.vignette_id}! "
                f"Using original vignette."
            )
            personalization_log.personalization_successful = False
            personalization_log.error_message = "Attribute validation failed"
            personalization_log.attributes_preserved = False
            return vignette, personalization_log

        # Log successful personalization
        personalization_log.personalized = {
            "scenario_text": response.scenario_intro,
            "option_a_description": response.option_a_description,
            "option_b_description": response.option_b_description,
            "reasoning": response.reasoning
        }
        personalization_log.personalization_successful = True
        personalization_log.attributes_preserved = True

        self._logger.info(
            f"Successfully personalized vignette {vignette.vignette_id}: {response.reasoning}"
        )

        return personalized_vignette, personalization_log

    def _extract_trade_off_from_vignette(self, vignette: Vignette) -> str:
        """
        Extract the key trade-off being tested by comparing option attributes.

        Args:
            vignette: Vignette with two options

        Returns:
            Human-readable description of the trade-off
        """
        attrs_a = vignette.options[0].attributes
        attrs_b = vignette.options[1].attributes

        differences = []
        for key in attrs_a.keys():
            val_a = attrs_a.get(key)
            val_b = attrs_b.get(key)
            if val_a != val_b:
                differences.append(f"{key}: A={val_a}, B={val_b}")

        if not differences:
            return "Options are identical (no trade-off)"

        return "Key differences:\n- " + "\n- ".join(differences)

    def _format_attributes(self, attributes: dict) -> str:
        """Format attribute dictionary for display in prompts."""
        lines = []
        for key, value in attributes.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
