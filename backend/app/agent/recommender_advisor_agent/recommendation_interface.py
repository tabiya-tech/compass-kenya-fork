"""
Recommendation Interface for the Recommender/Advisor Agent.

Handles generating or loading recommendations from:
1. Node2Vec algorithm (Jasmin's implementation)
2. Stub recommendations for development

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
    
    Abstracts away the source of recommendations (Node2Vec or stubs)
    so the agent doesn't need to know about the implementation.
    """
    
    def __init__(self, node2vec_client: Optional[Any] = None):
        """
        Initialize the recommendation interface.
        
        Args:
            node2vec_client: Optional Node2Vec client for generating recommendations.
                            If None, uses stub recommendations.
        """
        self._node2vec_client = node2vec_client
    
    async def generate_recommendations(
        self,
        youth_id: str,
        preference_vector: Optional[PreferenceVector] = None,
        skills_vector: Optional[dict] = None,
        bws_occupation_scores: Optional[dict[str, float]] = None,
    ) -> Node2VecRecommendations:
        """
        Generate recommendations for a user.
        
        Tries Node2Vec first, falls back to stubs if unavailable.
        
        Args:
            youth_id: User/youth identifier
            preference_vector: Preference vector from Epic 2
            skills_vector: Skills vector from Epic 4
            bws_occupation_scores: BWS occupation ranking from Epic 2
            
        Returns:
            Node2VecRecommendations object
        """
        # Try Node2Vec client first
        if self._node2vec_client and NODE2VEC_AVAILABLE:
            try:
                logger.info(f"Generating recommendations for {youth_id} via Node2Vec")
                return await self._node2vec_client.generate_recommendations(
                    youth_id=youth_id,
                    preference_vector=preference_vector,
                    skills_vector=skills_vector,
                    bws_scores=bws_occupation_scores
                )
            except Exception as e:
                logger.warning(f"Node2Vec failed, using stubs: {e}")
        
        # Return stub recommendations for development
        logger.info(f"Using stub recommendations for {youth_id}")
        return self.get_stub_recommendations(youth_id)
    
    def get_stub_recommendations(self, youth_id: str) -> Node2VecRecommendations:
        """
        Get stub recommendations for development without Node2Vec.
        
        These are realistic sample recommendations for testing the agent.
        
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
                    originUuid="esco_2512_origin",
                    rank=1,
                    occupation_id="ESCO_2512",
                    occupation_code="2512",
                    occupation="Data Analyst",
                    confidence_score=0.85,
                    justification="Matches your analytical skills and preference for structured work. Growing field with many opportunities.",
                    skills_match_score=0.75,
                    preference_match_score=0.88,
                    labor_demand_score=0.90,
                    graph_proximity_score=0.82,
                    essential_skills=["Excel", "SQL", "Data Visualization", "Statistical Analysis"],
                    user_skill_coverage=0.6,
                    skill_gaps=["Python", "Machine Learning basics"],
                    description="Data Analysts collect, process, and analyze data to help organizations make informed decisions.",
                    typical_tasks=[
                        "Collect and clean data from various sources",
                        "Create reports and visualizations",
                        "Identify trends and patterns",
                        "Present findings to stakeholders"
                    ],
                    career_path_next_steps=["Junior Data Analyst", "Senior Data Analyst", "Data Scientist", "Analytics Manager"],
                    labor_demand_category="high",
                    salary_range="KES 60,000-120,000/month"
                ),
                OccupationRecommendation(
                    uuid="occ_002_uuid",
                    originUuid="esco_2513_origin",
                    rank=2,
                    occupation_id="ESCO_2513",
                    occupation_code="2513",
                    occupation="M&E Specialist",
                    confidence_score=0.78,
                    justification="Uses your fieldwork experience and evaluation skills. Common in NGO sector (aligns with your values).",
                    skills_match_score=0.70,
                    preference_match_score=0.82,
                    labor_demand_score=0.75,
                    graph_proximity_score=0.78,
                    essential_skills=["Program Evaluation", "Report Writing", "Data Collection", "Stakeholder Management"],
                    user_skill_coverage=0.5,
                    skill_gaps=["M&E Frameworks", "Impact Assessment"],
                    description="Monitoring and Evaluation Specialists design and implement systems to track program effectiveness.",
                    typical_tasks=[
                        "Design monitoring frameworks",
                        "Collect and analyze program data",
                        "Write evaluation reports",
                        "Provide recommendations for improvement"
                    ],
                    career_path_next_steps=["M&E Officer", "M&E Specialist", "M&E Manager", "Program Director"],
                    labor_demand_category="medium",
                    salary_range="KES 50,000-100,000/month"
                ),
                OccupationRecommendation(
                    uuid="occ_003_uuid",
                    originUuid="esco_2514_origin",
                    rank=3,
                    occupation_id="ESCO_2514",
                    occupation_code="2514",
                    occupation="Research Assistant",
                    confidence_score=0.72,
                    justification="Leverages your research and writing skills. Good stepping stone with flexible arrangements.",
                    skills_match_score=0.80,
                    preference_match_score=0.70,
                    labor_demand_score=0.65,
                    graph_proximity_score=0.74,
                    essential_skills=["Research Methods", "Academic Writing", "Literature Review", "Data Entry"],
                    user_skill_coverage=0.65,
                    skill_gaps=["Statistical Software (SPSS/Stata)"],
                    description="Research Assistants support academic and policy research projects.",
                    typical_tasks=[
                        "Conduct literature reviews",
                        "Collect and enter data",
                        "Assist with analysis",
                        "Help prepare publications"
                    ],
                    career_path_next_steps=["Research Assistant", "Research Associate", "Research Fellow", "Lead Researcher"],
                    labor_demand_category="medium",
                    salary_range="KES 40,000-80,000/month"
                )
            ],
            opportunity_recommendations=[
                OpportunityRecommendation(
                    uuid="opp_001_uuid",
                    originUuid="job_001_origin",
                    rank=1,
                    opportunity_title="Internship at XYZ Foundation",
                    location="Nairobi",
                    justification="Entry-level position that builds foundation in impact evaluation. 6-month program with potential for full-time hire.",
                    essential_skills=["Data Analysis", "Report Writing"],
                    employer="XYZ Foundation",
                    salary_range="KES 25,000-35,000/month",
                    contract_type="internship",
                    related_occupation_id="occ_001_uuid"
                ),
                OpportunityRecommendation(
                    uuid="opp_002_uuid",
                    originUuid="job_002_origin",
                    rank=2,
                    opportunity_title="Research Assistantship at ABC Lab",
                    location="Remote",
                    justification="Flexible work arrangement with focus on quantitative analysis.",
                    essential_skills=["Research Methods", "Statistical Analysis"],
                    employer="ABC Research Lab",
                    salary_range="KES 30,000-40,000/month",
                    contract_type="contract",
                    related_occupation_id="occ_003_uuid"
                )
            ],
            skillstraining_recommendations=[
                SkillsTrainingRecommendation(
                    uuid="skill_001_uuid",
                    originUuid="training_001_origin",
                    rank=1,
                    skill="Data Analysis with Python",
                    training_title="Google Data Analytics Professional Certificate",
                    provider="Coursera",
                    estimated_hours=200,
                    justification="Industry-recognized certification that covers key data analysis skills. Opens doors to many analyst roles.",
                    cost="Free with financial aid",
                    location="Online",
                    delivery_mode="online",
                    target_occupations=["Data Analyst", "Business Analyst", "Research Assistant"],
                    fills_gap_for=["occ_001_uuid", "occ_003_uuid"]
                ),
                SkillsTrainingRecommendation(
                    uuid="skill_002_uuid",
                    originUuid="training_002_origin",
                    rank=2,
                    skill="M&E Fundamentals",
                    training_title="Introduction to Monitoring & Evaluation",
                    provider="ALX",
                    estimated_hours=40,
                    justification="Quick course covering M&E frameworks used in NGOs and development sector.",
                    cost="KES 5,000",
                    location="Online",
                    delivery_mode="online",
                    target_occupations=["M&E Specialist", "Program Officer"],
                    fills_gap_for=["occ_002_uuid"]
                )
            ],
            confidence=0.78
        )
