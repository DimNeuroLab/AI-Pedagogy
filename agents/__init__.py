"""
Agents package for AIP (Artificial Intelligence Pedagogy).

This package contains the agent implementations for the pedagogical interaction system.
"""

from .base_agent import Agent
from .teacher_agent import TeacherAgent
from .learner_agent import LearnerAgent
from .oracle import OracleAgent
from .expert import ExpertAgent
from .parent_agent import ParentAgent

__all__ = [
    'Agent',
    'TeacherAgent',
    'LearnerAgent',
    'OracleAgent',
    'ExpertAgent',
    'ParentAgent',
]
