"""
Recommendation Interface for the Recommender/Advisor Agent.

Handles generating or loading recommendations from:
1. Node2Vec algorithm (Jasmin's implementation)
2. Stub recommendations for development
3. Conversion from Jasmin's format to agent format

Epic 3: Recommender Agent Implementation
"""

from typing import Any, Optional
import logging

from app.agent.recommender_advisor_agent.types import (
    Node2VecRecommendations,
    OccupationRecommendation,
    OpportunityRecommendation,
    SkillsTrainingRecommendation,
)
from app.agent.preference_elicitation_agent.types import PreferenceVector

logger = logging.getLogger(__name__)


# ========== SKILL GAP TO TRAINING MAPPING ==========
# Maps skill labels from Node2Vec to concrete training courses
# This bridges the gap between "learn skill X" and "take course Y at provider Z"

SKILL_TO_TRAINING_MAP = {
    # Electrical/Technical skills
    "supervise correctional procedures": {
        "training_title": "Law Enforcement and Corrections Training",
        "provider": "Kenya Prisons Training College",
        "cost": "KES 50,000-80,000",
        "estimated_hours": 240,
        "delivery_mode": "in_person",
        "location": "Nairobi",
    },
    "teach housekeeping skills": {
        "training_title": "Hospitality and Housekeeping Management",
        "provider": "Utalii College / Kenya Coast Hotel Training Institute",
        "cost": "KES 30,000-50,000",
        "estimated_hours": 160,
        "delivery_mode": "in_person",
        "location": "Nairobi / Mombasa",
    },
    "enterprise risk management": {
        "training_title": "Enterprise Risk Management Certification",
        "provider": "Institute of Certified Public Accountants of Kenya (ICPAK)",
        "cost": "KES 80,000-120,000",
        "estimated_hours": 120,
        "delivery_mode": "hybrid",
        "location": "Nairobi / Online",
    },
    "control compliance of railway vehicles regulations": {
        "training_title": "Transport Safety and Compliance Training",
        "provider": "Kenya School of Government / NTSA",
        "cost": "KES 25,000-40,000",
        "estimated_hours": 80,
        "delivery_mode": "in_person",
        "location": "Nairobi",
    },
    "handle equipment while suspended": {
        "training_title": "Industrial Rigging and Safety Certification",
        "provider": "Directorate of Occupational Safety and Health Services (DOSHS)",
        "cost": "KES 15,000-25,000",
        "estimated_hours": 40,
        "delivery_mode": "in_person",
        "location": "Nairobi / Mombasa / Kisumu",
    },
    "lead police investigations": {
        "training_title": "Criminal Investigation and Forensics",
        "provider": "Kenya Police College",
        "cost": "KES 60,000-100,000",
        "estimated_hours": 200,
        "delivery_mode": "in_person",
        "location": "Nairobi",
    },
    # Programming/Tech skills
    "python": {
        "training_title": "Python for Data Science and Development",
        "provider": "Moringa School / ALX Africa / Coursera",
        "cost": "Free - KES 120,000 (depending on provider)",
        "estimated_hours": 200,
        "delivery_mode": "online",
        "location": "Online / Nairobi / Mombasa",
    },
    "haskell": {
        "training_title": "Functional Programming with Haskell",
        "provider": "Online platforms (Coursera, edX)",
        "cost": "Free - KES 15,000",
        "estimated_hours": 80,
        "delivery_mode": "online",
        "location": "Online",
    },
    "git / version control": {
        "training_title": "Git and Version Control for Developers",
        "provider": "freeCodeCamp / Udemy / Moringa School",
        "cost": "Free - KES 5,000",
        "estimated_hours": 20,
        "delivery_mode": "online",
        "location": "Online",
    },
    "sql / database management": {
        "training_title": "SQL and Database Management",
        "provider": "Coursera / Udemy / Moringa School",
        "cost": "Free - KES 30,000",
        "estimated_hours": 60,
        "delivery_mode": "online",
        "location": "Online / Nairobi",
    },
    # Add more as needed...
}


def convert_skill_gaps_to_trainings(
    skill_gaps: list[dict[str, Any]]
) -> list[SkillsTrainingRecommendation]:
    """
    Convert Jasmin's skill gap recommendations to training course recommendations.

    Jasmin provides: "learn skill X" with proximity scores and job unlock counts.
    Agent wants: "take course Y at provider Z for $W".

    This function bridges the gap by mapping skills to curated training courses.
    For unmapped skills, it creates a generic training recommendation.

    Args:
        skill_gaps: List of skill gap dicts from Node2Vec output
            Format: {skill_id, skill_label, proximity_score, job_unlock_count, combined_score, reasoning}

    Returns:
        List of SkillsTrainingRecommendation objects for agent use
    """
    trainings = []

    for idx, gap in enumerate(skill_gaps):
        skill_label = gap.get("skill_label", "").lower()
        skill_id = gap.get("skill_id", "")
        reasoning = gap.get("reasoning", f"Learning this skill would help unlock {gap.get('job_unlock_count', 0)} jobs.")

        # Look up training data from map
        training_data = SKILL_TO_TRAINING_MAP.get(skill_label)

        if training_data:
            # Create training recommendation from map
            trainings.append(SkillsTrainingRecommendation(
                uuid=f"training_{skill_id}",
                originUuid=skill_id,
                rank=idx + 1,
                skill=gap.get("skill_label", "Unknown Skill"),
                training_title=training_data["training_title"],
                provider=training_data["provider"],
                cost=training_data["cost"],
                estimated_hours=training_data["estimated_hours"],
                delivery_mode=training_data["delivery_mode"],
                location=training_data.get("location"),
                justification=reasoning,
                target_occupations=[],  # Could be extracted from job_unlock_count
                fills_gap_for=[]
            ))
        else:
            # Fallback: Create generic training recommendation for unmapped skills
            logger.debug(f"No training mapping for skill: {skill_label}")
            trainings.append(SkillsTrainingRecommendation(
                uuid=f"training_{skill_id}",
                originUuid=skill_id,
                rank=idx + 1,
                skill=gap.get("skill_label", "Unknown Skill"),
                training_title=f"Training in {gap.get('skill_label', 'this skill')}",
                provider="Various training providers",
                cost="Contact training providers for pricing",
                estimated_hours=None,
                delivery_mode=None,
                location="Various locations",
                justification=reasoning,
                target_occupations=[],
                fills_gap_for=[]
            ))

    return trainings

# Node2Vec import (Jasmin's algorithm - optional)
try:
    from app.epic3.node2vec.recommender import Node2VecRecommender
    NODE2VEC_AVAILABLE = True
except ImportError:
    Node2VecRecommender = None
    NODE2VEC_AVAILABLE = False


class RecommendationInterface:
    """
    Interface for generating/loading recommendations.

    Abstracts away the source of recommendations (MatchingServiceClient or stubs)
    so the agent doesn't need to know about the implementation.
    """

    def __init__(self, matching_service_client: Optional[Any] = None, node2vec_client: Optional[Any] = None):
        """
        Initialize the recommendation interface.

        Args:
            matching_service_client: Optional MatchingServiceClient for deployed matching service.
            node2vec_client: Optional Node2Vec client (deprecated - use matching_service_client).
                            If None, uses stub recommendations.
        """
        self._matching_service_client = matching_service_client
        self._node2vec_client = node2vec_client  # Keep for backwards compatibility
    
    async def generate_recommendations(
        self,
        youth_id: str,
        city: Optional[str] = None,
        province: Optional[str] = None,
        preference_vector: Optional[PreferenceVector] = None,
        skills_vector: Optional[dict] = None,
        bws_occupation_scores: Optional[dict[str, float]] = None,
    ) -> Node2VecRecommendations:
        """
        Generate recommendations for a user.

        Tries MatchingServiceClient first, then Node2Vec, falls back to stubs if unavailable.

        Args:
            youth_id: User/youth identifier
            city: User's city (required by matching service)
            province: User's province/state (required by matching service)
            preference_vector: Preference vector from Epic 2
            skills_vector: Skills vector from Epic 4
            bws_occupation_scores: BWS occupation ranking from Epic 2

        Returns:
            Node2VecRecommendations object (in agent format)
        """
        # Try MatchingServiceClient first (deployed service)
        if self._matching_service_client:
            try:
                logger.info(f"Generating recommendations for {youth_id} via MatchingServiceClient")
                raw_output = await self._matching_service_client.generate_recommendations(
                    youth_id=youth_id,
                    city=city,
                    province=province,
                    skills_vector=skills_vector,
                    preference_vector=preference_vector
                )

                # Convert matching service format to agent format
                logger.debug("Converting MatchingService output to agent format")
                return Node2VecRecommendations.from_jasmin_output(raw_output)

            except Exception as e:
                logger.warning(f"MatchingServiceClient failed, trying fallbacks: {e}")

        # Try Node2Vec client (legacy/local)
        if self._node2vec_client and NODE2VEC_AVAILABLE:
            try:
                logger.info(f"Generating recommendations for {youth_id} via Node2Vec (legacy)")
                raw_output = await self._node2vec_client.generate_recommendations(
                    youth_id=youth_id,
                    preference_vector=preference_vector,
                    skills_vector=skills_vector,
                    bws_scores=bws_occupation_scores
                )

                # Convert Jasmin's format to agent format
                logger.debug("Converting Node2Vec output to agent format")
                return Node2VecRecommendations.from_jasmin_output(raw_output)

            except Exception as e:
                logger.warning(f"Node2Vec failed, using stubs: {e}")

        # Return stub recommendations for development
        logger.info(f"Using stub recommendations for {youth_id}")
        return self.get_stub_recommendations(youth_id)
    
    def get_stub_recommendations(self, youth_id: str) -> Node2VecRecommendations:
        """
        Get stub recommendations for development without Node2Vec.

        These are realistic sample recommendations for testing the agent,
        based on a Mombasa youth persona with informal sector background.

        PERSONA: Hassan, 24, Mombasa
        - Completed Form 4, some technical college
        - Has worked casual jobs at the port, helped uncle with electrical repairs
        - Good with hands, basic phone/mobile money skills
        - Wants stable income but values flexibility
        - Family expects him to contribute financially

        Args:
            youth_id: User/youth identifier

        Returns:
            Node2VecRecommendations with sample data
        """
        return Node2VecRecommendations(
            youth_id=youth_id,
            occupation_recommendations=[
                OccupationRecommendation(
                    uuid="occ_001_uuid",
                    originUuid="kesco_7411_origin",
                    rank=1,
                    occupation_id="KESCO_7411",
                    occupation_code="7411",
                    occupation="Fundi wa Stima (Electrician)",
                    confidence_score=0.88,
                    justification="Your hands-on experience helping your uncle with electrical work gives you a strong foundation. High demand in Mombasa's growing construction and hotel sector.",
                    skills_match_score=0.82,
                    preference_match_score=0.85,
                    labor_demand_score=0.92,
                    graph_proximity_score=0.88,
                    essential_skills=[
                        "Basic wiring and installation",
                        "Reading electrical diagrams",
                        "Safety procedures",
                        "Using multimeter and tools",
                        "Customer communication"
                    ],
                    user_skill_coverage=0.55,
                    skill_gaps=["Formal certification (Grade Test)", "Industrial wiring"],
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
                    labor_demand_category="high",
                    salary_range="KES 800-2,000/day (job-based) or KES 25,000-45,000/month (employed)"
                ),
                OccupationRecommendation(
                    uuid="occ_002_uuid",
                    originUuid="kesco_8322_origin",
                    rank=2,
                    occupation_id="KESCO_8322",
                    occupation_code="8322",
                    occupation="Boda-Boda Rider / Delivery Driver",
                    confidence_score=0.79,
                    justification="Offers immediate income and flexibility you value. Your knowledge of Mombasa streets is an asset. Can start quickly while building other skills.",
                    skills_match_score=0.70,
                    preference_match_score=0.88,
                    labor_demand_score=0.85,
                    graph_proximity_score=0.72,
                    essential_skills=[
                        "Motorcycle riding (valid license)",
                        "Knowledge of local routes",
                        "Basic phone/M-Pesa skills",
                        "Customer service",
                        "Time management"
                    ],
                    user_skill_coverage=0.75,
                    skill_gaps=["Motorcycle license (if not yet obtained)"],
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
                    labor_demand_category="high",
                    salary_range="KES 500-1,500/day depending on hustle"
                ),
                OccupationRecommendation(
                    uuid="occ_003_uuid",
                    originUuid="kesco_9329_origin",
                    rank=3,
                    occupation_id="KESCO_9329",
                    occupation_code="9329",
                    occupation="Port Cargo Handler / Stevedore",
                    confidence_score=0.74,
                    justification="Your experience with casual port work is valuable. More organized positions offer better pay and some job security.",
                    skills_match_score=0.78,
                    preference_match_score=0.65,
                    labor_demand_score=0.80,
                    graph_proximity_score=0.75,
                    essential_skills=[
                        "Physical fitness and stamina",
                        "Following safety protocols",
                        "Basic cargo handling",
                        "Teamwork",
                        "Punctuality and reliability"
                    ],
                    user_skill_coverage=0.70,
                    skill_gaps=["Forklift certification", "Container handling training"],
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
                    labor_demand_category="medium",
                    salary_range="KES 600-1,200/day (casual) or KES 20,000-35,000/month (registered)"
                ),
                OccupationRecommendation(
                    uuid="occ_004_uuid",
                    originUuid="kesco_7233_origin",
                    rank=4,
                    occupation_id="KESCO_7233",
                    occupation_code="7233",
                    occupation="Boat/Marine Equipment Fundi",
                    confidence_score=0.71,
                    justification="Mombasa's fishing and tourism boat industry needs repair skills. Combines your electrical knowledge with marine work.",
                    skills_match_score=0.68,
                    preference_match_score=0.75,
                    labor_demand_score=0.70,
                    graph_proximity_score=0.72,
                    essential_skills=[
                        "Outboard motor repair",
                        "Basic electrical troubleshooting",
                        "Fiberglass patching",
                        "Engine maintenance",
                        "Customer negotiation"
                    ],
                    user_skill_coverage=0.45,
                    skill_gaps=["Marine engine training", "Fiberglass work"],
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
                    labor_demand_category="medium",
                    salary_range="KES 1,000-3,000/job or KES 20,000-40,000/month (busy season)"
                ),
                OccupationRecommendation(
                    uuid="occ_005_uuid",
                    originUuid="kesco_5221_origin",
                    rank=5,
                    occupation_id="KESCO_5221",
                    occupation_code="5221",
                    occupation="Market Vendor / Trader",
                    confidence_score=0.68,
                    justification="Low startup cost, flexible hours, and potential to grow. Your M-Pesa skills help with transactions.",
                    skills_match_score=0.60,
                    preference_match_score=0.80,
                    labor_demand_score=0.75,
                    graph_proximity_score=0.65,
                    essential_skills=[
                        "Basic math and pricing",
                        "Customer service",
                        "M-Pesa transactions",
                        "Negotiation",
                        "Stock management"
                    ],
                    user_skill_coverage=0.65,
                    skill_gaps=["Sourcing goods at good prices", "Business record-keeping"],
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
                    labor_demand_category="medium",
                    salary_range="KES 300-1,000/day profit (depends on product and location)"
                )
            ],
            opportunity_recommendations=[
                OpportunityRecommendation(
                    uuid="opp_001_uuid",
                    originUuid="job_001_origin",
                    rank=1,
                    opportunity_title="Electrical Apprenticeship - Nyali Construction Site",
                    location="Nyali, Mombasa",
                    justification="Learn from certified electricians while earning. The foreman is known to train serious workers.",
                    essential_skills=["Basic wiring", "Willingness to learn", "Physical work"],
                    employer="Nyali Heights Development",
                    salary_range="KES 500-800/day + skills training",
                    contract_type="contract",
                    related_occupation_id="occ_001_uuid"
                ),
                OpportunityRecommendation(
                    uuid="opp_002_uuid",
                    originUuid="job_002_origin",
                    rank=2,
                    opportunity_title="Glovo Delivery Partner",
                    location="Mombasa (various zones)",
                    justification="Flexible hours, paid per delivery. Good way to earn while exploring other opportunities.",
                    essential_skills=["Motorcycle + license", "Smartphone", "M-Pesa"],
                    employer="Glovo Kenya",
                    salary_range="KES 100-200 per delivery",
                    contract_type="freelance",
                    posting_url="https://glovoapp.com/ke/riders",
                    related_occupation_id="occ_002_uuid"
                ),
                OpportunityRecommendation(
                    uuid="opp_003_uuid",
                    originUuid="job_003_origin",
                    rank=3,
                    opportunity_title="Cargo Handler - Kilindini Port",
                    location="Mombasa Port",
                    justification="Regular work available. Being registered with a gang gives more consistent income than casual pickup.",
                    essential_skills=["Physical fitness", "Reliability", "Safety awareness"],
                    employer="Various shipping agents",
                    salary_range="KES 800-1,200/day",
                    contract_type="contract",
                    related_occupation_id="occ_003_uuid"
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
                    justification="The Grade Test certification opens doors to formal employment and higher-paying contracts. Many hotels and companies require certified electricians.",
                    cost="KES 15,000-20,000",
                    location="Mombasa Technical",
                    delivery_mode="in_person",
                    target_occupations=["Electrician", "Maintenance Technician"],
                    fills_gap_for=["occ_001_uuid"]
                ),
                SkillsTrainingRecommendation(
                    uuid="skill_002_uuid",
                    originUuid="training_002_origin",
                    rank=2,
                    skill="Solar Panel Installation",
                    training_title="Solar PV Installation Training",
                    provider="Kenya Power / Various NGOs",
                    estimated_hours=40,
                    justification="Solar is booming in Coast region. Adds to your electrical skills and pays very well.",
                    cost="Free - KES 10,000 (NGO programs often subsidized)",
                    location="Mombasa / Kilifi",
                    delivery_mode="hybrid",
                    target_occupations=["Solar Technician", "Electrician"],
                    fills_gap_for=["occ_001_uuid"]
                ),
                SkillsTrainingRecommendation(
                    uuid="skill_003_uuid",
                    originUuid="training_003_origin",
                    rank=3,
                    skill="Motorcycle Riding License",
                    training_title="NTSA Motorcycle License (Class A)",
                    provider="Approved Driving Schools",
                    estimated_hours=20,
                    justification="Required for legal boda-boda work and delivery apps. Protects you from police harassment and opens formal delivery opportunities.",
                    cost="KES 3,000-5,000",
                    location="Mombasa driving schools",
                    delivery_mode="in_person",
                    target_occupations=["Boda-Boda Rider", "Delivery Driver"],
                    fills_gap_for=["occ_002_uuid"]
                ),
                SkillsTrainingRecommendation(
                    uuid="skill_004_uuid",
                    originUuid="training_004_origin",
                    rank=4,
                    skill="Forklift Operation",
                    training_title="Forklift Operator Certificate",
                    provider="Industrial Training Centres",
                    estimated_hours=40,
                    justification="Certified forklift operators earn much more at the port. Opens path to supervisor roles.",
                    cost="KES 8,000-12,000",
                    location="Mombasa",
                    delivery_mode="in_person",
                    target_occupations=["Forklift Operator", "Warehouse Supervisor"],
                    fills_gap_for=["occ_003_uuid"]
                )
            ],
            confidence=0.82
        )
