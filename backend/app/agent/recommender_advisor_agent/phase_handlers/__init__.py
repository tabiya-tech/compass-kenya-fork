"""
Phase handlers for the Recommender/Advisor Agent.

Each phase of the conversation is handled by a dedicated handler class.
"""

from app.agent.recommender_advisor_agent.phase_handlers.base_handler import BasePhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.intro_handler import IntroPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.present_handler import PresentPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.exploration_handler import ExplorationPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.concerns_handler import ConcernsPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.tradeoffs_handler import TradeoffsPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.followup_handler import FollowupPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.skills_pivot_handler import SkillsPivotPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.action_handler import ActionPhaseHandler
from app.agent.recommender_advisor_agent.phase_handlers.wrapup_handler import WrapupPhaseHandler

__all__ = [
    "BasePhaseHandler",
    "IntroPhaseHandler",
    "PresentPhaseHandler",
    "ExplorationPhaseHandler",
    "ConcernsPhaseHandler",
    "TradeoffsPhaseHandler",
    "FollowupPhaseHandler",
    "SkillsPivotPhaseHandler",
    "ActionPhaseHandler",
    "WrapupPhaseHandler",
]
