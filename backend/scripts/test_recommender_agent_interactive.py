#!/usr/bin/env python3
"""
Interactive test script for the Recommender/Advisor Agent.

This script allows you to test all phases of the recommender agent
in an interactive mode without needing full backend integration.

Usage:
    poetry run python scripts/test_recommender_agent_interactive.py
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

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.recommender_advisor_agent.state import RecommenderAdvisorAgentState
from app.agent.recommender_advisor_agent.types import (
    ConversationPhase,
    Node2VecRecommendations,
    OccupationRecommendation,
    OpportunityRecommendation,
    SkillsTrainingRecommendation,
    ActionType,
    CommitmentLevel
)
from app.agent.preference_elicitation_agent.types import PreferenceVector
from app.agent.recommender_advisor_agent.phase_handlers.intro_handler import IntroPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.present_handler import PresentPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.exploration_handler import ExplorationPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.concerns_handler import ConcernsPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.action_handler import ActionPhaseHandler
from app.agent.recommender_advisor_agent.intent_classifier import IntentClassifier
from app.agent.recommender_advisor_agent.llm_response_models import (
    ConversationResponse,
    ResistanceClassification,
    UserIntentClassification,
    ActionExtractionResult
)
from app.agent.recommender_advisor_agent.recommendation_interface import RecommendationInterface
from app.agent.llm_caller import LLMCaller
from app.conversation_memory.conversation_memory_types import (
    ConversationContext,
    ConversationHistory,
    ConversationTurn
)
from app.conversation_memory.conversation_formatter import ConversationHistoryFormatter
from app.agent.agent_types import AgentInput, AgentOutput
from common_libs.llm.generative_models import GeminiGenerativeLLM
from app.countries import Country
from app.vector_search.esco_search_service import OccupationSearchService, VectorSearchConfig
from app.vector_search.embeddings_model import GoogleEmbeddingService
from app.server_dependencies.db_dependencies import CompassDBProvider
from app.app_config import get_application_config
from common_libs.environment_settings.constants import EmbeddingConfig
import os

# Install rich traceback handler
install(show_locals=True)

# Initialize console
console = Console()

# Test configuration
TEST_COUNTRY = Country.KENYA  # Configure country for localization testing


class SessionStats:
    """Track session statistics."""
    def __init__(self):
        self.start_time = time.time()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency = 0.0
        self.turns = 0

        # Estimated costs (Gemini Flash 2.0 pricing)
        self.input_cost_per_1m = 0.075  # $0.075 / 1M input tokens
        self.output_cost_per_1m = 0.30  # $0.30 / 1M output tokens

    def add_turn(self, input_tokens: int, output_tokens: int, latency: float):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency += latency
        self.turns += 1

    @property
    def duration(self) -> float:
        return time.time() - self.start_time

    @property
    def estimated_cost(self) -> float:
        input_cost = (self.total_input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_cost_per_1m
        return input_cost + output_cost

    def get_summary_table(self) -> Table:
        table = Table(title="Session Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Total Duration", f"{timedelta(seconds=int(self.duration))}")
        table.add_row("Total Turns", str(self.turns))
        table.add_row("Total Latency (LLM)", f"{self.total_latency:.2f}s")
        table.add_row("Avg Latency/Turn", f"{self.total_latency/self.turns:.2f}s" if self.turns > 0 else "0s")
        table.add_row("Total Input Tokens", f"{self.total_input_tokens:,}")
        table.add_row("Total Output Tokens", f"{self.total_output_tokens:,}")
        table.add_row("Total Tokens", f"{self.total_input_tokens + self.total_output_tokens:,}")
        table.add_row("Estimated Cost", f"${self.estimated_cost:.6f}")

        return table


# Configure logging
def setup_logging(level=logging.INFO):
    """Configure logging with RichHandler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
    )

    # Suppress noisy loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def print_header(text: str):
    """Print a formatted header."""
    console.print(Panel(Text(text, justify="center", style="bold magenta"), box=box.DOUBLE))


def print_section(text: str):
    """Print a formatted section header."""
    console.print(f"\n[bold blue]{text}[/]")
    console.print(f"[blue]{'-'*len(text)}[/]")


def print_agent(text: str):
    """Print agent message."""
    console.print(Panel(Markdown(text), title="[bold green]Recommender Agent[/]", border_style="green", box=box.ROUNDED))


def print_user(text: str):
    """Print user message."""
    console.print(Panel(Text(text), title="[bold cyan]You[/]", border_style="cyan", box=box.ROUNDED))


def print_info(text: str):
    """Print info message."""
    console.print(f"[bold yellow]ℹ {text}[/]")


def print_error(text: str):
    """Print error message."""
    console.print(f"[bold red]✗ {text}[/]")


def print_success(text: str):
    """Print success message."""
    console.print(f"[bold green]✓ {text}[/]")


def get_user_input(prompt: str = "") -> str:
    """Get input from user."""
    return console.input(f"[bold cyan]{prompt}[/]")


def display_menu(options: List[str]) -> int:
    """Display a menu and get user selection."""
    table = Table(show_header=False, box=box.SIMPLE)
    for i, option in enumerate(options, 1):
        table.add_row(f"[bold blue]{i}.[/]", option)

    console.print(table)

    while True:
        try:
            choice = int(console.input("\n[bold]Select option (number): [/]"))
            if 1 <= choice <= len(options):
                return choice
            print_error(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print_error("Please enter a valid number")


def create_sample_recommendations() -> Node2VecRecommendations:
    """
    Create sample recommendations for testing.

    PERSONA: Hassan, 24, Mombasa
    - Completed Form 4, some technical college
    - Has worked casual jobs at the port, helped uncle with electrical repairs
    - Good with hands, basic phone/mobile money skills
    - Wants stable income but values flexibility
    - Family expects him to contribute financially
    """
    return Node2VecRecommendations(
        youth_id="test_user_123",
        generated_at="2026-01-09T10:30:00Z",
        recommended_by=["Algorithm"],
        occupation_recommendations=[
            OccupationRecommendation(
                uuid="occ_001_uuid",
                originUuid="kesco_7411_origin",
                rank=1,
                occupation_id="KESCO_7411",
                occupation_code="7411",
                occupation="Fundi wa Stima (Electrician)",
                confidence_score=0.88,
                skills_match_score=0.82,
                preference_match_score=0.85,
                labor_demand_score=0.92,
                graph_proximity_score=0.88,
                labor_demand_category="high",
                salary_range="KES 800-2,000/day (job-based) or KES 25,000-45,000/month",
                justification="Your hands-on experience helping your uncle with electrical work gives you a strong foundation. High demand in Mombasa's growing construction and hotel sector.",
                essential_skills=[
                    "Basic wiring and installation",
                    "Reading electrical diagrams",
                    "Safety procedures",
                    "Using multimeter and tools",
                    "Customer communication"
                ],
                description="Electricians install, maintain, and repair electrical wiring and systems in homes, hotels, and businesses.",
                typical_tasks=[
                    "Install and repair electrical wiring in buildings",
                    "Fix faulty sockets, switches, and lighting",
                    "Install ceiling fans and water heaters",
                    "Troubleshoot electrical problems",
                    "Quote jobs and collect payment from clients"
                ],
                career_path_next_steps=[
                    "Apprentice/Helper → Fundi (1-2 years)",
                    "Fundi → Certified Electrician (Grade Test)",
                    "Certified → Contractor/Own business",
                    "Specialize in solar installation (growing demand)"
                ],
                skill_gaps=["Formal certification (Grade Test)", "Industrial wiring"],
                user_skill_coverage=0.55
            ),
            OccupationRecommendation(
                uuid="occ_002_uuid",
                originUuid="kesco_8322_origin",
                rank=2,
                occupation_id="KESCO_8322",
                occupation_code="8322",
                occupation="Boda-Boda Rider / Delivery Driver",
                confidence_score=0.79,
                skills_match_score=0.70,
                preference_match_score=0.88,
                labor_demand_score=0.85,
                graph_proximity_score=0.72,
                labor_demand_category="high",
                salary_range="KES 500-1,500/day depending on hustle",
                justification="Offers immediate income and flexibility you value. Your knowledge of Mombasa streets is an asset. Can start quickly while building other skills.",
                essential_skills=[
                    "Motorcycle riding (valid license)",
                    "Knowledge of local routes",
                    "Basic phone/M-Pesa skills",
                    "Customer service",
                    "Time management"
                ],
                description="Boda-boda riders provide passenger transport and delivery services using motorcycles.",
                typical_tasks=[
                    "Transport passengers around the city",
                    "Deliver food, packages, and goods",
                    "Navigate traffic efficiently",
                    "Manage daily earnings and fuel costs",
                    "Maintain motorcycle in good condition"
                ],
                career_path_next_steps=[
                    "Rider (employed) → Own motorcycle",
                    "Join delivery apps (Glovo, Uber Eats)",
                    "Build regular customer base",
                    "Grow to 2-3 bikes with riders (fleet owner)"
                ],
                skill_gaps=["Motorcycle license (if not yet obtained)"],
                user_skill_coverage=0.75
            ),
            OccupationRecommendation(
                uuid="occ_003_uuid",
                originUuid="kesco_9329_origin",
                rank=3,
                occupation_id="KESCO_9329",
                occupation_code="9329",
                occupation="Port Cargo Handler / Stevedore",
                confidence_score=0.74,
                skills_match_score=0.78,
                preference_match_score=0.65,
                labor_demand_score=0.80,
                graph_proximity_score=0.75,
                labor_demand_category="medium",
                salary_range="KES 600-1,200/day (casual) or KES 20,000-35,000/month",
                justification="Your experience with casual port work is valuable. More organized positions offer better pay and some job security.",
                essential_skills=[
                    "Physical fitness and stamina",
                    "Following safety protocols",
                    "Basic cargo handling",
                    "Teamwork",
                    "Punctuality and reliability"
                ],
                description="Cargo handlers load, unload, and move goods at the port, warehouses, and shipping yards.",
                typical_tasks=[
                    "Load and unload cargo from ships/trucks",
                    "Operate basic cargo equipment",
                    "Sort and stack containers/goods",
                    "Follow safety procedures strictly",
                    "Work in shifts (day/night)"
                ],
                career_path_next_steps=[
                    "Casual laborer → Registered handler",
                    "Get forklift/equipment certification",
                    "Handler → Supervisor/Tally clerk",
                    "Move to logistics/clearing agent roles"
                ],
                skill_gaps=["Forklift certification", "Container handling training"],
                user_skill_coverage=0.70
            ),
            OccupationRecommendation(
                uuid="occ_004_uuid",
                originUuid="kesco_7233_origin",
                rank=4,
                occupation_id="KESCO_7233",
                occupation_code="7233",
                occupation="Boat/Marine Equipment Fundi",
                confidence_score=0.71,
                skills_match_score=0.68,
                preference_match_score=0.75,
                labor_demand_score=0.70,
                graph_proximity_score=0.72,
                labor_demand_category="medium",
                salary_range="KES 1,000-3,000/job or KES 20,000-40,000/month (busy season)",
                justification="Mombasa's fishing and tourism boat industry needs repair skills. Combines your electrical knowledge with marine work.",
                essential_skills=[
                    "Outboard motor repair",
                    "Basic electrical troubleshooting",
                    "Fiberglass patching",
                    "Engine maintenance",
                    "Customer negotiation"
                ],
                description="Marine fundis repair and maintain boats, outboard motors, and marine electrical systems.",
                typical_tasks=[
                    "Repair outboard motors for fishermen",
                    "Fix electrical systems on boats",
                    "Patch and maintain boat hulls",
                    "Install marine equipment",
                    "Travel to different landing sites for jobs"
                ],
                career_path_next_steps=[
                    "Learn from experienced marine fundi",
                    "Specialize in outboard motors (Yamaha, etc.)",
                    "Build reputation at fish landing sites",
                    "Open marine repair shop"
                ],
                skill_gaps=["Marine engine training", "Fiberglass work"],
                user_skill_coverage=0.45
            ),
            OccupationRecommendation(
                uuid="occ_005_uuid",
                originUuid="kesco_5221_origin",
                rank=5,
                occupation_id="KESCO_5221",
                occupation_code="5221",
                occupation="Market Vendor / Trader",
                confidence_score=0.68,
                skills_match_score=0.60,
                preference_match_score=0.80,
                labor_demand_score=0.75,
                graph_proximity_score=0.65,
                labor_demand_category="medium",
                salary_range="KES 300-1,000/day profit (depends on product and location)",
                justification="Low startup cost, flexible hours, and potential to grow. Your M-Pesa skills help with transactions.",
                essential_skills=[
                    "Basic math and pricing",
                    "Customer service",
                    "M-Pesa transactions",
                    "Negotiation",
                    "Stock management"
                ],
                description="Market vendors sell goods (food, household items, phone accessories, etc.) at markets, streets, or small stalls.",
                typical_tasks=[
                    "Source and buy goods for resale",
                    "Set up stall and display products",
                    "Negotiate prices with customers",
                    "Manage daily cash and M-Pesa payments",
                    "Track what sells well"
                ],
                career_path_next_steps=[
                    "Start small (phone accessories, fruits)",
                    "Build regular customers",
                    "Get permanent stall/kiosk",
                    "Grow to wholesale or multiple stalls"
                ],
                skill_gaps=["Sourcing goods at good prices", "Business record-keeping"],
                user_skill_coverage=0.65
            )
        ],
        opportunity_recommendations=[
            OpportunityRecommendation(
                uuid="opp_001_uuid",
                originUuid="job_001_origin",
                rank=1,
                opportunity_title="Electrical Apprenticeship - Nyali Construction Site",
                location="Nyali, Mombasa",
                employer="Nyali Heights Development",
                contract_type="contract",
                salary_range="KES 500-800/day + skills training",
                justification="Learn from certified electricians while earning. The foreman is known to train serious workers.",
                essential_skills=["Basic wiring", "Willingness to learn", "Physical work"]
            ),
            OpportunityRecommendation(
                uuid="opp_002_uuid",
                originUuid="job_002_origin",
                rank=2,
                opportunity_title="Glovo Delivery Partner",
                location="Mombasa (various zones)",
                employer="Glovo Kenya",
                contract_type="freelance",
                salary_range="KES 100-200 per delivery",
                posting_url="https://glovoapp.com/ke/riders",
                justification="Flexible hours, paid per delivery. Good way to earn while exploring other opportunities.",
                essential_skills=["Motorcycle + license", "Smartphone", "M-Pesa"]
            ),
            OpportunityRecommendation(
                uuid="opp_003_uuid",
                originUuid="job_003_origin",
                rank=3,
                opportunity_title="Cargo Handler - Kilindini Port",
                location="Mombasa Port",
                employer="Various shipping agents",
                contract_type="contract",
                salary_range="KES 800-1,200/day",
                justification="Regular work available. Being registered with a gang gives more consistent income than casual pickup.",
                essential_skills=["Physical fitness", "Reliability", "Safety awareness"]
            )
        ],
        skillstraining_recommendations=[
            SkillsTrainingRecommendation(
                uuid="skill_001_uuid",
                originUuid="training_001_origin",
                rank=1,
                skill="Electrical Installation (Grade Test Preparation)",
                training_title="Electrician Grade III Certification",
                provider="Mombasa Technical Training Institute",
                estimated_hours=160,
                cost="KES 15,000-20,000",
                location="Mombasa Technical",
                delivery_mode="in_person",
                target_occupations=["Electrician", "Maintenance Technician"],
                fills_gap_for=["occ_001_uuid"],
                justification="The Grade Test certification opens doors to formal employment and higher-paying contracts. Many hotels and companies require certified electricians."
            ),
            SkillsTrainingRecommendation(
                uuid="skill_002_uuid",
                originUuid="training_002_origin",
                rank=2,
                skill="Solar Panel Installation",
                training_title="Solar PV Installation Training",
                provider="Kenya Power / Various NGOs",
                estimated_hours=40,
                cost="Free - KES 10,000 (NGO programs often subsidized)",
                location="Mombasa / Kilifi",
                delivery_mode="hybrid",
                target_occupations=["Solar Technician", "Electrician"],
                fills_gap_for=["occ_001_uuid"],
                justification="Solar is booming in Coast region. Adds to your electrical skills and pays very well."
            ),
            SkillsTrainingRecommendation(
                uuid="skill_003_uuid",
                originUuid="training_003_origin",
                rank=3,
                skill="Motorcycle Riding License",
                training_title="NTSA Motorcycle License (Class A)",
                provider="Approved Driving Schools",
                estimated_hours=20,
                cost="KES 3,000-5,000",
                location="Mombasa driving schools",
                delivery_mode="in_person",
                target_occupations=["Boda-Boda Rider", "Delivery Driver"],
                fills_gap_for=["occ_002_uuid"],
                justification="Required for legal boda-boda work and delivery apps. Protects you from police harassment and opens formal delivery opportunities."
            ),
            SkillsTrainingRecommendation(
                uuid="skill_004_uuid",
                originUuid="training_004_origin",
                rank=4,
                skill="Forklift Operation",
                training_title="Forklift Operator Certificate",
                provider="Industrial Training Centres",
                estimated_hours=40,
                cost="KES 8,000-12,000",
                location="Mombasa",
                delivery_mode="in_person",
                target_occupations=["Forklift Operator", "Warehouse Supervisor"],
                fills_gap_for=["occ_003_uuid"],
                justification="Certified forklift operators earn much more at the port. Opens path to supervisor roles."
            )
        ],
        confidence=0.82
    )


def create_sample_skills_vector() -> dict:
    """
    Create sample skills vector for testing.

    Matches Hassan's informal sector background:
    - Electrical work experience (from uncle)
    - Port casual labor experience
    - Mobile money/phone skills
    - Physical/hands-on work
    """
    return {
        "top_skills": [
            {"preferredLabel": "Basic Electrical Wiring", "proficiency": 0.6},
            {"preferredLabel": "Manual Handling / Physical Labor", "proficiency": 0.8},
            {"preferredLabel": "M-Pesa / Mobile Money", "proficiency": 0.85},
            {"preferredLabel": "Customer Service", "proficiency": 0.65},
            {"preferredLabel": "Tool Usage (hand tools)", "proficiency": 0.7},
            {"preferredLabel": "Motorcycle Riding", "proficiency": 0.5},
            {"preferredLabel": "Basic Math / Pricing", "proficiency": 0.7}
        ]
    }


def create_sample_preference_vector() -> PreferenceVector:
    """
    Create sample preference vector for testing.

    Matches Hassan's priorities:
    - Financial needs (family pressure) → HIGH
    - Work-life balance (values flexibility) → HIGH
    - Job security (wants stability) → MODERATE-HIGH
    - Career advancement → MODERATE (open to growth)
    - Work environment → MODERATE (okay with physical work)
    - Social impact → LOW (practical focus)
    """
    return PreferenceVector(
        financial_importance=0.85,         # High - needs to support family
        work_environment_importance=0.55,  # Moderate - okay with physical work outdoors
        career_advancement_importance=0.60, # Moderate - interested but not primary focus
        work_life_balance_importance=0.80, # High - values flexibility
        job_security_importance=0.70,      # Moderate-high - wants stable income
        task_preference_importance=0.65,   # Moderate - prefers hands-on work
        social_impact_importance=0.40      # Lower - practical/financial focus first
    )


def display_state_info(state: RecommenderAdvisorAgentState):
    """Display current agent state information."""
    table = Table(title="Agent State", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Session ID", str(state.session_id))
    table.add_row("Phase", f"[bold]{state.conversation_phase}[/]")
    table.add_row("Current Focus", state.current_focus_id or "None")
    table.add_row("Recommendation Type", state.current_recommendation_type or "None")

    # Engagement tracking
    table.add_row("Presented Occupations", str(len(state.presented_occupations)))
    table.add_row("Explored Items", str(len(state.explored_items)))
    table.add_row("Rejected Occupations", str(state.rejected_occupations))
    table.add_row("User Interest Signals", str(len(state.user_interest_signals)))

    # Concerns
    if state.concerns_raised:
        concerns_text = "\n".join([f"- {c.concern} ({c.resistance_type})" for c in state.concerns_raised[-3:]])
        table.add_row("Recent Concerns", concerns_text)

    console.print(table)


def display_recommendations(recs: Node2VecRecommendations):
    """Display recommendations in a table."""
    if recs.occupation_recommendations:
        occ_table = Table(title="Occupation Recommendations", box=box.ROUNDED, show_lines=True)
        occ_table.add_column("Rank", style="bold yellow", width=6)
        occ_table.add_column("Occupation", style="bold cyan")
        occ_table.add_column("Scores", style="green")
        occ_table.add_column("Details", style="white")

        for occ in recs.occupation_recommendations[:5]:
            scores = f"Overall: {occ.confidence_score:.0%}\n"
            if occ.skills_match_score:
                scores += f"Skills: {occ.skills_match_score:.0%}\n"
            if occ.preference_match_score:
                scores += f"Prefs: {occ.preference_match_score:.0%}\n"
            if occ.labor_demand_score:
                scores += f"Demand: {occ.labor_demand_score:.0%}"

            details = f"{occ.labor_demand_category or 'N/A'} demand\n"
            details += f"{occ.salary_range or 'Salary N/A'}"

            occ_table.add_row(
                str(occ.rank),
                occ.occupation,
                scores.strip(),
                details.strip()
            )

        console.print(occ_table)


def display_preference_vector(pv: PreferenceVector):
    """Display preference vector in a table."""
    table = Table(title="Preference Vector", box=box.ROUNDED)
    table.add_column("Dimension", style="cyan")
    table.add_column("Importance", style="yellow", justify="center")
    table.add_column("Interpretation", style="white")

    def interpret(value: float) -> str:
        if value >= 0.7:
            return "HIGH"
        elif value >= 0.5:
            return "MODERATE"
        elif value >= 0.3:
            return "LOW"
        else:
            return "VERY LOW"

    table.add_row("Financial Compensation", f"{pv.financial_importance:.2f}", interpret(pv.financial_importance))
    table.add_row("Work Environment", f"{pv.work_environment_importance:.2f}", interpret(pv.work_environment_importance))
    table.add_row("Career Advancement", f"{pv.career_advancement_importance:.2f}", interpret(pv.career_advancement_importance))
    table.add_row("Work-Life Balance", f"{pv.work_life_balance_importance:.2f}", interpret(pv.work_life_balance_importance))
    table.add_row("Job Security", f"{pv.job_security_importance:.2f}", interpret(pv.job_security_importance))
    table.add_row("Task Preferences", f"{pv.task_preference_importance:.2f}", interpret(pv.task_preference_importance))
    table.add_row("Social Impact", f"{pv.social_impact_importance:.2f}", interpret(pv.social_impact_importance))

    console.print(table)


async def initialize_handlers():
    """Initialize phase handlers with LLM."""
    print_info("Initializing phase handlers with Gemini LLM...")

    # Initialize LLM exactly like the agent does
    from common_libs.llm.models_utils import LLMConfig, LOW_TEMPERATURE_GENERATION_CONFIG, JSON_GENERATION_CONFIG
    from app.agent.recommender_advisor_agent.phase_handlers.tradeoffs_handler import TradeoffsPhaseHandler
    from app.agent.recommender_advisor_agent.phase_handlers.followup_handler import FollowupPhaseHandler
    from app.agent.recommender_advisor_agent.phase_handlers.skills_pivot_handler import SkillsPivotPhaseHandler
    from app.agent.recommender_advisor_agent.phase_handlers.wrapup_handler import WrapupPhaseHandler

    llm_config = LLMConfig(
        generation_config=LOW_TEMPERATURE_GENERATION_CONFIG | JSON_GENERATION_CONFIG
    )

    llm = GeminiGenerativeLLM(
        system_instructions="",
        config=llm_config
    )

    # Create LLM callers exactly like the agent does
    conversation_caller = LLMCaller[ConversationResponse](
        model_response_type=ConversationResponse
    )

    resistance_caller = LLMCaller[ResistanceClassification](
        model_response_type=ResistanceClassification
    )

    intent_caller = LLMCaller[UserIntentClassification](
        model_response_type=UserIntentClassification
    )

    action_caller = LLMCaller[ActionExtractionResult](
        model_response_type=ActionExtractionResult
    )

    # Initialize recommendation interface
    recommendation_interface = RecommendationInterface(node2vec_client=None)

    # Initialize IntentClassifier (centralized intent classification)
    intent_classifier = IntentClassifier(intent_caller=intent_caller)

    # Initialize occupation search service for out-of-list occupation handling
    # (manually create since we're outside FastAPI context)
    occupation_search_service = None  # Default to None
    try:
        console.print("[dim]Initializing occupation search service...[/]")
        app_config = get_application_config()
        embedding_config = EmbeddingConfig()

        # Get database connection
        taxonomy_db = await CompassDBProvider.get_taxonomy_db()

        # Create embedding service
        embedding_service = GoogleEmbeddingService(model_name=app_config.embeddings_model_name)

        # Create occupation search config
        occupation_vector_search_config = VectorSearchConfig(
            collection_name=embedding_config.occupation_collection_name,
            index_name=embedding_config.embedding_index,
            embedding_key=embedding_config.embedding_key,
        )

        # Create occupation search service
        occupation_search_service = OccupationSearchService(
            taxonomy_db,
            embedding_service,
            occupation_vector_search_config,
            app_config.taxonomy_model_id
        )
        console.print(f"[green]✓ Occupation search service initialized (service={type(occupation_search_service).__name__})[/]")
    except Exception as e:
        import traceback
        console.print(f"[red]✗ Failed to initialize occupation search service:[/]")
        console.print(f"[red]{e}[/]")
        console.print(f"[dim]{traceback.format_exc()}[/]")
        console.print("[yellow]  Out-of-list occupation search will be disabled[/]")
        occupation_search_service = None

    # Initialize handlers in dependency order (handlers with no dependencies first)
    concerns_handler = ConcernsPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        resistance_caller=resistance_caller,
        occupation_search_service=occupation_search_service
    )

    intro_handler = IntroPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        recommendation_interface=recommendation_interface,
        occupation_search_service=occupation_search_service
    )

    tradeoffs_handler = TradeoffsPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller
    )

    skills_pivot_handler = SkillsPivotPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller
    )

    wrapup_handler = WrapupPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        db6_client=None  # Optional DB6 client
    )

    followup_handler = FollowupPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier
    )

    # Handlers with dependencies
    exploration_handler = ExplorationPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
        concerns_handler=concerns_handler,
        tradeoffs_handler=tradeoffs_handler,
        occupation_search_service=occupation_search_service
    )

    present_handler = PresentPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        intent_classifier=intent_classifier,
        exploration_handler=exploration_handler,
        concerns_handler=concerns_handler,
        tradeoffs_handler=tradeoffs_handler,
        occupation_search_service=occupation_search_service
    )


    action_handler = ActionPhaseHandler(
        conversation_llm=llm,
        conversation_caller=conversation_caller,
        action_caller=action_caller,
        intent_classifier=intent_classifier
    )

    # Set up delegation chains after all handlers are initialized
    exploration_handler._action_handler = action_handler
    exploration_handler._tradeoffs_handler = tradeoffs_handler
    action_handler._present_handler = present_handler
    action_handler._concerns_handler = concerns_handler
    action_handler._wrapup_handler = wrapup_handler

    print_success("Phase handlers initialized!")

    return {
        ConversationPhase.INTRO: intro_handler,
        ConversationPhase.PRESENT_RECOMMENDATIONS: present_handler,
        ConversationPhase.CAREER_EXPLORATION: exploration_handler,
        ConversationPhase.ADDRESS_CONCERNS: concerns_handler,
        ConversationPhase.ACTION_PLANNING: action_handler,
        ConversationPhase.DISCUSS_TRADEOFFS: tradeoffs_handler,
        ConversationPhase.FOLLOW_UP: followup_handler,
        ConversationPhase.SKILLS_UPGRADE_PIVOT: skills_pivot_handler,
        ConversationPhase.WRAPUP: wrapup_handler,
        ConversationPhase.COMPLETE: wrapup_handler  # Use wrapup_handler.handle_complete
    }


async def test_intro_phase():
    """Test the INTRO phase."""
    print_header("Testing INTRO Phase")

    try:
        handlers = await initialize_handlers()

        # Create state
        state = RecommenderAdvisorAgentState(
            session_id="test_session_12345",
            youth_id="test_user_123",
            country_of_user=TEST_COUNTRY,
            conversation_phase=ConversationPhase.INTRO,
            recommendations=create_sample_recommendations(),
            skills_vector=create_sample_skills_vector(),
            preference_vector=create_sample_preference_vector()
        )

        # Create context
        conversation_history = ConversationHistory()
        context = ConversationContext(
            all_history=conversation_history,
            history=conversation_history,
            summary=""
        )

        # Execute INTRO handler
        print_info("Executing INTRO handler...")
        handler = handlers[ConversationPhase.INTRO]

        with console.status("[bold green]Processing...", spinner="dots"):
            response, llm_stats = await handler.handle("", state, context)

        # Display response
        print_agent(response.message)

        # Show stats
        print_section("LLM Stats")
        if llm_stats:
            stats_table = Table(box=box.SIMPLE)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            total_input = sum(s.prompt_token_count for s in llm_stats)
            total_output = sum(s.response_token_count for s in llm_stats)
            total_latency = sum(s.response_time_in_sec for s in llm_stats)

            stats_table.add_row("Input Tokens", f"{total_input:,}")
            stats_table.add_row("Output Tokens", f"{total_output:,}")
            stats_table.add_row("Latency", f"{total_latency:.2f}s")

            console.print(stats_table)
        else:
            print_info("No LLM calls made (INTRO uses static message)")

        # Show updated state
        print_section("Updated State")
        display_state_info(state)

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing INTRO phase: {e}")
        import traceback
        traceback.print_exc()


async def test_present_phase():
    """Test the PRESENT_RECOMMENDATIONS phase."""
    print_header("Testing PRESENT_RECOMMENDATIONS Phase")

    try:
        handlers = await initialize_handlers()

        # Create state
        recommendations = create_sample_recommendations()
        state = RecommenderAdvisorAgentState(
            session_id="test_session_12345",
            country_of_user=TEST_COUNTRY,
            youth_id="test_user_123",
            conversation_phase=ConversationPhase.PRESENT_RECOMMENDATIONS,
            recommendations=recommendations,
            skills_vector=create_sample_skills_vector(),
            preference_vector=create_sample_preference_vector()
        )

        # Display recommendations first
        print_section("Available Recommendations")
        display_recommendations(recommendations)

        # Create context
        conversation_history = ConversationHistory()
        context = ConversationContext(
            all_history=conversation_history,
            history=conversation_history,
            summary=""
        )

        # Execute PRESENT handler
        print_info("Executing PRESENT handler with LLM...")
        handler = handlers[ConversationPhase.PRESENT_RECOMMENDATIONS]

        with console.status("[bold green]LLM generating presentation...", spinner="dots"):
            response, llm_stats = await handler.handle("Ready to see recommendations", state, context)

        # Display response
        print_agent(response.message)

        # Show stats
        print_section("LLM Stats")
        if llm_stats:
            stats_table = Table(box=box.SIMPLE)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            total_input = sum(s.prompt_token_count for s in llm_stats)
            total_output = sum(s.response_token_count for s in llm_stats)
            total_latency = sum(s.response_time_in_sec for s in llm_stats)

            stats_table.add_row("Input Tokens", f"{total_input:,}")
            stats_table.add_row("Output Tokens", f"{total_output:,}")
            stats_table.add_row("Latency", f"{total_latency:.2f}s")

            console.print(stats_table)

        # Show metadata
        if response.metadata:
            print_section("Response Metadata")
            console.print(json.dumps(response.metadata, indent=2))

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing PRESENT phase: {e}")
        import traceback
        traceback.print_exc()


async def test_exploration_phase():
    """Test the CAREER_EXPLORATION phase."""
    print_header("Testing CAREER_EXPLORATION Phase")

    try:
        handlers = await initialize_handlers()

        # Create state with selected occupation
        recommendations = create_sample_recommendations()
        selected_occ = recommendations.occupation_recommendations[0]

        state = RecommenderAdvisorAgentState(
            session_id="test_session_12345",
            youth_id="test_user_123",
            country_of_user=TEST_COUNTRY,
            conversation_phase=ConversationPhase.CAREER_EXPLORATION,
            recommendations=recommendations,
            skills_vector=create_sample_skills_vector(),
            preference_vector=create_sample_preference_vector(),
            current_focus_id=selected_occ.uuid,
            current_recommendation_type="occupation"
        )

        print_info(f"Exploring: {selected_occ.occupation}")

        # Create context
        conversation_history = ConversationHistory()
        context = ConversationContext(
            all_history=conversation_history,
            history=conversation_history,
            summary=""
        )

        # Execute EXPLORATION handler
        print_info("Executing EXPLORATION handler with LLM...")
        handler = handlers[ConversationPhase.CAREER_EXPLORATION]

        with console.status("[bold green]LLM generating exploration...", spinner="dots"):
            response, llm_stats = await handler.handle("Tell me more about Data Analyst", state, context)

        # Display response
        print_agent(response.message)

        # Show stats
        print_section("LLM Stats")
        if llm_stats:
            stats_table = Table(box=box.SIMPLE)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            total_input = sum(s.prompt_token_count for s in llm_stats)
            total_output = sum(s.response_token_count for s in llm_stats)
            total_latency = sum(s.response_time_in_sec for s in llm_stats)

            stats_table.add_row("Input Tokens", f"{total_input:,}")
            stats_table.add_row("Output Tokens", f"{total_output:,}")
            stats_table.add_row("Latency", f"{total_latency:.2f}s")

            console.print(stats_table)

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing EXPLORATION phase: {e}")
        import traceback
        traceback.print_exc()


async def test_concerns_phase():
    """Test the ADDRESS_CONCERNS phase."""
    print_header("Testing ADDRESS_CONCERNS Phase")

    print_info("This will test the 2-step LLM process: classify → respond")

    # Get user concern
    print_section("Enter a concern to test")
    console.print("[dim]Examples:[/]")
    console.print("  - I don't think I have the technical skills for this")
    console.print("  - The salary is too low for my needs")
    console.print("  - I can't relocate to Nairobi")
    console.print("  - My family won't approve of this career\n")

    user_concern = get_user_input("Your concern: ")

    if not user_concern:
        print_error("No concern provided, using default")
        user_concern = "I don't think I have the technical skills for Data Analyst"

    try:
        handlers = await initialize_handlers()

        # Create state
        recommendations = create_sample_recommendations()
        selected_occ = recommendations.occupation_recommendations[0]

        state = RecommenderAdvisorAgentState(
            session_id="test_session_12345",
            youth_id="test_user_123",
            country_of_user=TEST_COUNTRY,
            conversation_phase=ConversationPhase.ADDRESS_CONCERNS,
            recommendations=recommendations,
            skills_vector=create_sample_skills_vector(),
            preference_vector=create_sample_preference_vector(),
            current_focus_id=selected_occ.uuid,
            current_recommendation_type="occupation"
        )

        # Create context
        conversation_history = ConversationHistory()
        # Add some history
        conversation_history.turns.append(
            ConversationTurn(
                index=0,
                input=AgentInput(message="Tell me about Data Analyst", is_artificial=False),
                output=AgentOutput(
                    message_for_user="Let's explore Data Analyst...",
                    finished=False,
                    llm_stats=[],
                    agent_response_time_in_sec=1.0
                )
            )
        )

        context = ConversationContext(
            all_history=conversation_history,
            history=conversation_history,
            summary=""
        )

        # Execute CONCERNS handler
        print_section("Step 1: Classifying Resistance")
        handler = handlers[ConversationPhase.ADDRESS_CONCERNS]

        with console.status("[bold green]LLM classifying concern...", spinner="dots"):
            response, llm_stats = await handler.handle(user_concern, state, context)

        # Show classification (from concerns recorded)
        if state.concerns_raised:
            latest_concern = state.concerns_raised[-1]
            class_table = Table(title="Resistance Classification", box=box.ROUNDED)
            class_table.add_column("Property", style="cyan")
            class_table.add_column("Value", style="yellow")

            class_table.add_row("Type", latest_concern.resistance_type)
            class_table.add_row("Concern", latest_concern.concern)

            console.print(class_table)

        print_section("Step 2: Generated Response")
        print_agent(response.message)

        # Show stats
        print_section("LLM Stats (Both Steps)")
        if llm_stats:
            stats_table = Table(box=box.SIMPLE)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            total_input = sum(s.prompt_token_count for s in llm_stats)
            total_output = sum(s.response_token_count for s in llm_stats)
            total_latency = sum(s.response_time_in_sec for s in llm_stats)

            stats_table.add_row("Total LLM Calls", str(len(llm_stats)))
            stats_table.add_row("Input Tokens", f"{total_input:,}")
            stats_table.add_row("Output Tokens", f"{total_output:,}")
            stats_table.add_row("Total Latency", f"{total_latency:.2f}s")

            console.print(stats_table)

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing CONCERNS phase: {e}")
        import traceback
        traceback.print_exc()


async def test_action_phase():
    """Test the ACTION_PLANNING phase."""
    print_header("Testing ACTION_PLANNING Phase")

    try:
        handlers = await initialize_handlers()

        # Create state with selected occupation and exploration completed
        recommendations = create_sample_recommendations()
        selected_occ = recommendations.occupation_recommendations[0]

        state = RecommenderAdvisorAgentState(
            session_id="test_session_12345",
            youth_id="test_user_123",
            country_of_user=TEST_COUNTRY,
            conversation_phase=ConversationPhase.ACTION_PLANNING,
            recommendations=recommendations,
            skills_vector=create_sample_skills_vector(),
            preference_vector=create_sample_preference_vector(),
            current_focus_id=selected_occ.uuid,
            current_recommendation_type="occupation"
        )

        print_info(f"Planning actions for: {selected_occ.occupation}")

        # Create context with previous exploration
        conversation_history = ConversationHistory()
        conversation_history.turns.append(
            ConversationTurn(
                index=0,
                input=AgentInput(message="Tell me about Data Analyst", is_artificial=False),
                output=AgentOutput(
                    message_for_user="Let's explore Data Analyst...",
                    finished=False,
                    llm_stats=[],
                    agent_response_time_in_sec=1.0
                )
            )
        )

        context = ConversationContext(
            all_history=conversation_history,
            history=conversation_history,
            summary=""
        )

        # Execute ACTION handler
        print_info("Executing ACTION_PLANNING handler with LLM...")
        handler = handlers[ConversationPhase.ACTION_PLANNING]

        with console.status("[bold green]LLM generating action plan...", spinner="dots"):
            response, llm_stats = await handler.handle("I want to apply for this", state, context)

        # Display response
        print_agent(response.message)

        # Show stats
        print_section("LLM Stats")
        if llm_stats:
            stats_table = Table(box=box.SIMPLE)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            total_input = sum(s.prompt_token_count for s in llm_stats)
            total_output = sum(s.response_token_count for s in llm_stats)
            total_latency = sum(s.response_time_in_sec for s in llm_stats)

            stats_table.add_row("Input Tokens", f"{total_input:,}")
            stats_table.add_row("Output Tokens", f"{total_output:,}")
            stats_table.add_row("Latency", f"{total_latency:.2f}s")

            console.print(stats_table)

        # Show action commitment
        if state.action_commitment:
            print_section("Action Commitment")
            commitment = state.action_commitment
            commit_table = Table(box=box.ROUNDED)
            commit_table.add_column("Property", style="cyan")
            commit_table.add_column("Value", style="yellow")

            commit_table.add_row("Recommendation", commitment.recommendation_title)
            commit_table.add_row("Action Type", commitment.action_type.value)
            commit_table.add_row("Commitment Level", commitment.commitment_level.value)
            if commitment.barriers_mentioned:
                commit_table.add_row("Barriers", ", ".join(commitment.barriers_mentioned))

            console.print(commit_table)

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing ACTION phase: {e}")
        import traceback
        traceback.print_exc()


async def test_tradeoffs_phase():
    """Test the DISCUSS_TRADEOFFS phase."""
    print_header("Testing DISCUSS_TRADEOFFS Phase")

    try:
        handlers = await initialize_handlers()

        # Create state with user preferring lower-demand option
        recommendations = create_sample_recommendations()
        # Assume user is interested in Marketing (rank 3, high demand)
        # but we'll show tradeoff with Data Analyst (rank 1, high demand)
        preferred_occ = recommendations.occupation_recommendations[2]  # Marketing

        state = RecommenderAdvisorAgentState(
            session_id="test_session_12345",
            youth_id="test_user_123",
            country_of_user=TEST_COUNTRY,
            conversation_phase=ConversationPhase.DISCUSS_TRADEOFFS,
            recommendations=recommendations,
            skills_vector=create_sample_skills_vector(),
            preference_vector=create_sample_preference_vector(),
            current_focus_id=preferred_occ.uuid,
            current_recommendation_type="occupation"
        )

        print_info(f"User prefers: {preferred_occ.occupation}")
        print_info("Will discuss tradeoffs vs higher-demand alternatives")

        # Create context
        conversation_history = ConversationHistory()
        context = ConversationContext(
            all_history=conversation_history,
            history=conversation_history,
            summary=""
        )

        # Execute TRADEOFFS handler
        print_info("Executing DISCUSS_TRADEOFFS handler with LLM...")
        handler = handlers[ConversationPhase.DISCUSS_TRADEOFFS]

        with console.status("[bold green]LLM generating tradeoff discussion...", spinner="dots"):
            response, llm_stats = await handler.handle("I really like Marketing", state, context)

        # Display response
        print_agent(response.message)

        # Show stats
        print_section("LLM Stats")
        if llm_stats:
            stats_table = Table(box=box.SIMPLE)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            total_input = sum(s.prompt_token_count for s in llm_stats)
            total_output = sum(s.response_token_count for s in llm_stats)
            total_latency = sum(s.response_time_in_sec for s in llm_stats)

            stats_table.add_row("Input Tokens", f"{total_input:,}")
            stats_table.add_row("Output Tokens", f"{total_output:,}")
            stats_table.add_row("Latency", f"{total_latency:.2f}s")

            console.print(stats_table)

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing TRADEOFFS phase: {e}")
        import traceback
        traceback.print_exc()


async def test_followup_phase():
    """Test the FOLLOW_UP phase."""
    print_header("Testing FOLLOW_UP Phase")

    # Get user input to test with
    print_section("Enter an ambiguous response to test")
    console.print("[dim]Examples:[/]")
    console.print("  - ok")
    console.print("  - hmm")
    console.print("  - maybe")
    console.print("  - I don't know\\n")

    user_response = get_user_input("Your ambiguous response: ")

    if not user_response:
        print_error("No response provided, using default")
        user_response = "hmm"

    try:
        handlers = await initialize_handlers()

        # Create state
        recommendations = create_sample_recommendations()

        state = RecommenderAdvisorAgentState(
            session_id="test_session_12345",
            youth_id="test_user_123",
            country_of_user=TEST_COUNTRY,
            conversation_phase=ConversationPhase.FOLLOW_UP,
            recommendations=recommendations,
            skills_vector=create_sample_skills_vector(),
            preference_vector=create_sample_preference_vector()
        )

        # Create context
        conversation_history = ConversationHistory()
        conversation_history.turns.append(
            ConversationTurn(
                index=0,
                input=AgentInput(message="Ready to see recommendations", is_artificial=False),
                output=AgentOutput(
                    message_for_user="Here are your top matches...",
                    finished=False,
                    llm_stats=[],
                    agent_response_time_in_sec=1.0
                )
            )
        )

        context = ConversationContext(
            all_history=conversation_history,
            history=conversation_history,
            summary=""
        )

        # Execute FOLLOW_UP handler
        print_section("Clarifying User Intent")
        handler = handlers[ConversationPhase.FOLLOW_UP]

        with console.status("[bold green]LLM clarifying intent...", spinner="dots"):
            response, llm_stats = await handler.handle(user_response, state, context)

        # Display response
        print_agent(response.message)

        # Show stats
        print_section("LLM Stats")
        if llm_stats:
            stats_table = Table(box=box.SIMPLE)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            total_input = sum(s.prompt_token_count for s in llm_stats)
            total_output = sum(s.response_token_count for s in llm_stats)
            total_latency = sum(s.response_time_in_sec for s in llm_stats)

            stats_table.add_row("Input Tokens", f"{total_input:,}")
            stats_table.add_row("Output Tokens", f"{total_output:,}")
            stats_table.add_row("Latency", f"{total_latency:.2f}s")

            console.print(stats_table)

        # Show phase transition
        print_section("Phase Transition")
        print_info(f"Next phase: {state.conversation_phase}")

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing FOLLOW_UP phase: {e}")
        import traceback
        traceback.print_exc()


async def test_skills_pivot_phase():
    """Test the SKILLS_UPGRADE_PIVOT phase."""
    print_header("Testing SKILLS_UPGRADE_PIVOT Phase")

    try:
        handlers = await initialize_handlers()

        # Create state with rejected occupations
        recommendations = create_sample_recommendations()

        state = RecommenderAdvisorAgentState(
            session_id="test_session_12345",
            youth_id="test_user_123",
            country_of_user=TEST_COUNTRY,
            conversation_phase=ConversationPhase.SKILLS_UPGRADE_PIVOT,
            recommendations=recommendations,
            skills_vector=create_sample_skills_vector(),
            preference_vector=create_sample_preference_vector(),
            rejected_occupations=3  # User has rejected 3 occupations
        )

        print_info("User has rejected 3+ occupations, pivoting to training")

        # Create context
        conversation_history = ConversationHistory()
        context = ConversationContext(
            all_history=conversation_history,
            history=conversation_history,
            summary=""
        )

        # Execute SKILLS_PIVOT handler
        print_info("Executing SKILLS_UPGRADE_PIVOT handler...")
        handler = handlers[ConversationPhase.SKILLS_UPGRADE_PIVOT]

        with console.status("[bold green]Generating training recommendations...", spinner="dots"):
            response, llm_stats = await handler.handle("None of those feel right", state, context)

        # Display response
        print_agent(response.message)

        # Show stats
        print_section("LLM Stats")
        if llm_stats:
            stats_table = Table(box=box.SIMPLE)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            total_input = sum(s.prompt_token_count for s in llm_stats)
            total_output = sum(s.response_token_count for s in llm_stats)
            total_latency = sum(s.response_time_in_sec for s in llm_stats)

            stats_table.add_row("Input Tokens", f"{total_input:,}")
            stats_table.add_row("Output Tokens", f"{total_output:,}")
            stats_table.add_row("Latency", f"{total_latency:.2f}s")

            console.print(stats_table)

        # Show pivot flag
        print_section("State Changes")
        print_info(f"Pivoted to training: {state.pivoted_to_training}")

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing SKILLS_PIVOT phase: {e}")
        import traceback
        traceback.print_exc()


async def test_wrapup_phase():
    """Test the WRAPUP phase."""
    print_header("Testing WRAPUP Phase")

    try:
        handlers = await initialize_handlers()

        # Create state with action commitment
        recommendations = create_sample_recommendations()
        selected_occ = recommendations.occupation_recommendations[0]

        # Create action commitment
        from app.agent.recommender_advisor_agent.types import ActionCommitment
        commitment = ActionCommitment(
            recommendation_id=selected_occ.uuid,
            recommendation_type="occupation",
            recommendation_title=selected_occ.occupation,
            action_type=ActionType.APPLY_TO_JOB,
            commitment_level=CommitmentLevel.WILL_DO_THIS_WEEK,
            barriers_mentioned=[]
        )

        state = RecommenderAdvisorAgentState(
            session_id="test_session_12345",
            youth_id="test_user_123",
            country_of_user=TEST_COUNTRY,
            conversation_phase=ConversationPhase.WRAPUP,
            recommendations=recommendations,
            skills_vector=create_sample_skills_vector(),
            preference_vector=create_sample_preference_vector(),
            current_focus_id=selected_occ.uuid,
            current_recommendation_type="occupation",
            action_commitment=commitment
        )

        print_info(f"Wrapping up session with commitment to: {selected_occ.occupation}")

        # Create context
        conversation_history = ConversationHistory()
        context = ConversationContext(
            all_history=conversation_history,
            history=conversation_history,
            summary=""
        )

        # Execute WRAPUP handler
        print_info("Executing WRAPUP handler...")
        handler = handlers[ConversationPhase.WRAPUP]

        with console.status("[bold green]Generating session summary...", spinner="dots"):
            response, llm_stats = await handler.handle("Yes, I'm ready to do this", state, context)

        # Display response
        print_agent(response.message)

        # Show stats
        print_section("LLM Stats")
        if llm_stats:
            stats_table = Table(box=box.SIMPLE)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            total_input = sum(s.prompt_token_count for s in llm_stats)
            total_output = sum(s.response_token_count for s in llm_stats)
            total_latency = sum(s.response_time_in_sec for s in llm_stats)

            stats_table.add_row("Input Tokens", f"{total_input:,}")
            stats_table.add_row("Output Tokens", f"{total_output:,}")
            stats_table.add_row("Latency", f"{total_latency:.2f}s")

            console.print(stats_table)

        # Show final phase
        print_section("Final State")
        print_info(f"Final phase: {state.conversation_phase}")
        print_success(f"Session finished: {response.finished}")

        console.input("\n[dim]Press Enter to continue...[/]")

    except Exception as e:
        print_error(f"Error testing WRAPUP phase: {e}")
        import traceback
        traceback.print_exc()


async def test_full_conversation():
    """Test a full conversation flow."""
    print_header("Interactive Conversation Test")

    print_info("This will start a conversation with the recommender agent.")
    print_info("Type 'quit' to exit, 'state' to see state, 'recs' to see recommendations, 'prefs' to see preferences.\n")

    session_stats = SessionStats()

    try:
        handlers = await initialize_handlers()

        # Create initial state
        recommendations = create_sample_recommendations()

        print_section("Sample Data Loaded")
        print_info("Skills, preferences, and recommendations loaded")

        # Show initial data
        print_section("Your Skills")
        skills = create_sample_skills_vector()
        skills_text = ", ".join([s["preferredLabel"] for s in skills["top_skills"]])
        console.print(f"[green]{skills_text}[/]")

        print_section("Your Preferences")
        preference_vector = create_sample_preference_vector()
        display_preference_vector(preference_vector)

        print_section("Available Recommendations")
        display_recommendations(recommendations)

        console.input("\n[dim]Press Enter to start conversation...[/]")

        # Initialize state
        state = RecommenderAdvisorAgentState(
            session_id="test_session_12345",
            youth_id="test_user_123",
            country_of_user=TEST_COUNTRY,
            conversation_phase=ConversationPhase.INTRO,
            recommendations=recommendations,
            skills_vector=skills,
            preference_vector=preference_vector
        )

        # Create conversation context
        conversation_history = ConversationHistory()

        # Start conversation loop
        turn_index = 0
        while True:
            # Get current handler
            current_handler = handlers.get(state.conversation_phase)
            if not current_handler:
                print_error(f"No handler for phase: {state.conversation_phase}")
                break

            # Get user input
            if turn_index == 0:
                user_message = ""  # First turn is automatic
                print_info(f"Starting conversation (Phase: {state.conversation_phase})...")
            else:
                user_input = get_user_input("You: ")

                # Handle commands
                if user_input.lower() == 'quit':
                    print_info("Ending conversation...")
                    break
                elif user_input.lower() == 'state':
                    display_state_info(state)
                    continue
                elif user_input.lower() == 'recs':
                    display_recommendations(state.recommendations)
                    continue
                elif user_input.lower() == 'prefs':
                    display_preference_vector(state.preference_vector)
                    continue

                user_message = user_input
                print_user(user_message)

            # Create context
            context = ConversationContext(
                all_history=conversation_history,
                history=conversation_history,
                summary=""
            )

            # Execute handler
            try:
                with console.status(f"[bold green]Agent thinking (Phase: {state.conversation_phase})...", spinner="dots"):
                    start_time = time.time()
                    response, llm_stats = await current_handler.handle(user_message, state, context)
                    latency = time.time() - start_time

                # Display response
                print_agent(response.message)

                # Note: Phase transitions are now handled automatically by the handlers
                # No manual keyword detection needed anymore

                # Collect stats
                if llm_stats:
                    input_tokens = sum(s.prompt_token_count for s in llm_stats)
                    output_tokens = sum(s.response_token_count for s in llm_stats)
                    session_stats.add_turn(input_tokens, output_tokens, latency)

                # Show turn stats
                stats_text = (
                    f"Phase: [bold]{state.conversation_phase}[/] | "
                    f"Turn: [bold]{turn_index}[/] | "
                    f"Latency: [bold]{latency:.2f}s[/]"
                )
                if llm_stats:
                    input_tokens = sum(s.prompt_token_count for s in llm_stats)
                    output_tokens = sum(s.response_token_count for s in llm_stats)
                    stats_text += f" | Tokens: [green]{input_tokens}[/] in / [yellow]{output_tokens}[/] out"

                console.print(Panel(stats_text, style="dim", box=box.ROUNDED))

                # Show state data collected so far
                print_section("Data Collected This Turn")
                state_data = Table(box=box.SIMPLE, show_header=False)
                state_data.add_column("Field", style="cyan")
                state_data.add_column("Value", style="white")

                state_data.add_row("Current Phase", str(state.conversation_phase))
                state_data.add_row("Current Focus", f"{state.current_focus_id or 'None'} ({state.current_recommendation_type or 'N/A'})")
                state_data.add_row("Explored Items", str(len(state.explored_items)))
                state_data.add_row("Presented Occupations", str(len(state.presented_occupations)))
                state_data.add_row("Rejected Occupations", str(state.rejected_occupations))
                state_data.add_row("User Interest Signals", str(len(state.user_interest_signals)))
                state_data.add_row("Concerns Raised", str(len(state.concerns_raised)))

                if state.concerns_raised:
                    latest_concern = state.concerns_raised[-1]
                    state_data.add_row("Latest Concern", f"{latest_concern.resistance_type}: {latest_concern.concern[:50]}...")

                console.print(state_data)
                console.print()  # Blank line

                # Update conversation history
                conversation_turn = ConversationTurn(
                    index=turn_index,
                    input=AgentInput(message=user_message, is_artificial=(turn_index == 0)),
                    output=AgentOutput(
                        message_for_user=response.message,
                        finished=response.finished,
                        llm_stats=llm_stats,
                        agent_response_time_in_sec=latency
                    )
                )
                conversation_history.turns.append(conversation_turn)

                turn_index += 1

                # Check if finished
                if response.finished:
                    print_success("Conversation complete!")
                    print_section("Final State")
                    display_state_info(state)
                    break

            except Exception as e:
                print_error(f"Error during conversation: {e}")
                import traceback
                traceback.print_exc()
                break

        # End of session summary
        print_header("Session Completed")
        console.print(session_stats.get_summary_table())

    except Exception as e:
        print_error(f"Error setting up conversation test: {e}")
        import traceback
        traceback.print_exc()


async def main_menu():
    """Main menu for interactive testing."""
    print_header("Recommender/Advisor Agent - Interactive Test")

    console.print(
        Panel(
            "This interactive test allows you to test all phases of the "
            "recommender advisor agent without needing full backend integration.",
            style="blue", box=box.ROUNDED
        )
    )

    # Ask user to set logging level
    print_section("Logging Configuration")
    log_choice = display_menu([
        "INFO - Standard output (recommended for normal testing)",
        "DEBUG - Detailed debug output (for investigating issues)",
        "WARNING - Only warnings and errors"
    ])

    log_levels = {
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.WARNING
    }
    setup_logging(log_levels[log_choice])
    print_success(f"Logging configured to {logging.getLevelName(log_levels[log_choice])} level")

    while True:
        print_section("Main Menu")

        choice = display_menu([
            "Test INTRO Phase",
            "Test PRESENT_RECOMMENDATIONS Phase (with LLM)",
            "Test CAREER_EXPLORATION Phase (with LLM)",
            "Test ADDRESS_CONCERNS Phase (2-step LLM)",
            "Test ACTION_PLANNING Phase (with LLM)",
            "Test DISCUSS_TRADEOFFS Phase (with LLM)",
            "Test FOLLOW_UP Phase (with LLM)",
            "Test SKILLS_UPGRADE_PIVOT Phase",
            "Test WRAPUP Phase",
            "Full Interactive Conversation (all phases)",
            "View Sample Data (recommendations, skills, preferences)",
            "Change Logging Level",
            "Exit"
        ])

        if choice == 1:
            await test_intro_phase()
        elif choice == 2:
            await test_present_phase()
        elif choice == 3:
            await test_exploration_phase()
        elif choice == 4:
            await test_concerns_phase()
        elif choice == 5:
            await test_action_phase()
        elif choice == 6:
            await test_tradeoffs_phase()
        elif choice == 7:
            await test_followup_phase()
        elif choice == 8:
            await test_skills_pivot_phase()
        elif choice == 9:
            await test_wrapup_phase()
        elif choice == 10:
            await test_full_conversation()
        elif choice == 11:
            print_section("Sample Data")

            print_info("Sample Skills:")
            skills = create_sample_skills_vector()
            skills_text = ", ".join([s["preferredLabel"] for s in skills["top_skills"]])
            console.print(f"[green]{skills_text}[/]\n")

            print_info("Sample Preferences:")
            display_preference_vector(create_sample_preference_vector())

            print_info("\nSample Recommendations:")
            display_recommendations(create_sample_recommendations())

            console.input("\n[dim]Press Enter to continue...[/]")
        elif choice == 12:
            # Change logging level
            print_section("Change Logging Level")
            new_log_choice = display_menu([
                "INFO - Standard output",
                "DEBUG - Detailed debug output",
                "WARNING - Only warnings and errors"
            ])
            setup_logging(log_levels[new_log_choice])
            print_success(f"Logging changed to {logging.getLevelName(log_levels[new_log_choice])} level")
        elif choice == 13:
            print_info("Exiting...")
            break


async def main():
    """Main entry point."""
    try:
        await main_menu()
    except KeyboardInterrupt:
        print_info("\nInterrupted by user. Exiting...")
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
