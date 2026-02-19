"""
Sample data for recommender advisor agent tests.

PERSONA: Hassan, 24, Mombasa
- Completed Form 4, some technical college
- Has worked casual jobs at the port, helped uncle with electrical repairs
- Good with hands, basic phone/mobile money skills
- Wants stable income but values flexibility
- Family expects him to contribute financially
"""

from app.agent.recommender_advisor_agent.types import (
    Node2VecRecommendations,
    OccupationRecommendation,
    OpportunityRecommendation,
    SkillsTrainingRecommendation,
    ScoreBreakdown,
    SkillComponent,
)
from app.agent.preference_elicitation_agent.types import PreferenceVector


def create_sample_recommendations() -> Node2VecRecommendations:
    """Create sample recommendations for testing."""
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
                final_score=0.88,
                score_breakdown=ScoreBreakdown(
                    total_skill_utility=0.82,
                    skill_components=SkillComponent(loc=0.85, ess=0.80, opt=0.82, grp=0.83),
                    skill_penalty_applied=0.0,
                    preference_score=0.85,
                    demand_score=0.92,
                    demand_label="High Expected Demand"
                ),
                salary_range="KES 800-2,000/day (job-based) or KES 25,000-45,000/month",
                justification="Your hands-on experience helping your uncle with electrical work gives you a strong foundation. High demand in Mombasa's growing construction and hotel sector.",
                description="Electricians install, maintain, and repair electrical wiring and systems in homes, hotels, and businesses.",
                typical_tasks=[
                    "Install and repair electrical wiring in buildings",
                    "Fix faulty sockets, switches, and lighting",
                    "Install ceiling fans and water heaters",
                    "Troubleshoot electrical problems",
                    "Quote jobs and collect payment from clients"
                ],
                career_path_next_steps=[
                    "Apprentice/Helper -> Fundi (1-2 years)",
                    "Fundi -> Certified Electrician (Grade Test)",
                    "Certified -> Contractor/Own business",
                    "Specialize in solar installation (growing demand)"
                ]
            ),
            OccupationRecommendation(
                uuid="occ_002_uuid",
                originUuid="kesco_8322_origin",
                rank=2,
                occupation_id="KESCO_8322",
                occupation_code="8322",
                occupation="Boda-Boda Rider / Delivery Driver",
                final_score=0.79,
                score_breakdown=ScoreBreakdown(
                    total_skill_utility=0.70,
                    skill_components=SkillComponent(loc=0.90, ess=0.65, opt=0.68, grp=0.72),
                    skill_penalty_applied=0.0,
                    preference_score=0.88,
                    demand_score=0.85,
                    demand_label="High Expected Demand"
                ),
                salary_range="KES 500-1,500/day depending on hustle",
                justification="Offers immediate income and flexibility you value. Your knowledge of Mombasa streets is an asset. Can start quickly while building other skills.",
                description="Boda-boda riders provide passenger transport and delivery services using motorcycles.",
                typical_tasks=[
                    "Transport passengers around the city",
                    "Deliver food, packages, and goods",
                    "Navigate traffic efficiently",
                    "Manage daily earnings and fuel costs",
                    "Maintain motorcycle in good condition"
                ],
                career_path_next_steps=[
                    "Rider (employed) -> Own motorcycle",
                    "Join delivery apps (Glovo, Uber Eats)",
                    "Build regular customer base",
                    "Grow to 2-3 bikes with riders (fleet owner)"
                ]
            ),
            OccupationRecommendation(
                uuid="occ_003_uuid",
                originUuid="kesco_9329_origin",
                rank=3,
                occupation_id="KESCO_9329",
                occupation_code="9329",
                occupation="Port Cargo Handler / Stevedore",
                final_score=0.74,
                score_breakdown=ScoreBreakdown(
                    total_skill_utility=0.78,
                    skill_components=SkillComponent(loc=0.95, ess=0.75, opt=0.70, grp=0.80),
                    skill_penalty_applied=0.0,
                    preference_score=0.65,
                    demand_score=0.60,
                    demand_label="Moderate Expected Demand"
                ),
                salary_range="KES 600-1,200/day (casual) or KES 20,000-35,000/month",
                justification="Your experience with casual port work is valuable. More organized positions offer better pay and some job security.",
                description="Cargo handlers load, unload, and move goods at the port, warehouses, and shipping yards.",
                typical_tasks=[
                    "Load and unload cargo from ships/trucks",
                    "Operate basic cargo equipment",
                    "Sort and stack containers/goods",
                    "Follow safety procedures strictly",
                    "Work in shifts (day/night)"
                ],
                career_path_next_steps=[
                    "Casual laborer -> Registered handler",
                    "Get forklift/equipment certification",
                    "Handler -> Supervisor/Tally clerk",
                    "Move to logistics/clearing agent roles"
                ]
            ),
            OccupationRecommendation(
                uuid="occ_004_uuid",
                originUuid="kesco_7233_origin",
                rank=4,
                occupation_id="KESCO_7233",
                occupation_code="7233",
                occupation="Boat/Marine Equipment Fundi",
                final_score=0.71,
                score_breakdown=ScoreBreakdown(
                    total_skill_utility=0.68,
                    skill_components=SkillComponent(loc=0.80, ess=0.65, opt=0.68, grp=0.70),
                    skill_penalty_applied=0.0,
                    preference_score=0.75,
                    demand_score=0.55,
                    demand_label="Moderate Expected Demand"
                ),
                salary_range="KES 1,000-3,000/job or KES 20,000-40,000/month (busy season)",
                justification="Mombasa's fishing and tourism boat industry needs repair skills. Combines your electrical knowledge with marine work.",
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
                ]
            ),
            OccupationRecommendation(
                uuid="occ_005_uuid",
                originUuid="kesco_5221_origin",
                rank=5,
                occupation_id="KESCO_5221",
                occupation_code="5221",
                occupation="Market Vendor / Trader",
                final_score=0.68,
                score_breakdown=ScoreBreakdown(
                    total_skill_utility=0.60,
                    skill_components=SkillComponent(loc=0.88, ess=0.55, opt=0.60, grp=0.62),
                    skill_penalty_applied=0.0,
                    preference_score=0.80,
                    demand_score=0.58,
                    demand_label="Moderate Expected Demand"
                ),
                salary_range="KES 300-1,000/day profit (depends on product and location)",
                justification="Low startup cost, flexible hours, and potential to grow. Your M-Pesa skills help with transactions.",
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
    - Financial needs (family pressure) -> HIGH
    - Work-life balance (values flexibility) -> HIGH
    - Job security (wants stability) -> MODERATE-HIGH
    """
    return PreferenceVector(
        financial_importance=0.85,
        work_environment_importance=0.55,
        career_advancement_importance=0.60,
        work_life_balance_importance=0.80,
        job_security_importance=0.70,
        task_preference_importance=0.65,
        social_impact_importance=0.40
    )
