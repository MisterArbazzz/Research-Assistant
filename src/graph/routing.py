"""Conditional routing functions for the Research Assistant graph.

Per CLAUDE.md pattern #4: routing is pure — no LLM calls, no I/O. Iteration
caps live in src/config.py and are read once here at the single enforcement
site (route_after_validator). A grep for MAX_RESEARCH_ATTEMPTS should find it
twice: definition in config.py, enforcement here.
"""

from __future__ import annotations

from typing import Literal

from ..config import get_settings
from .state import ResearchState

# Confidence threshold above which the Research Agent's findings skip the
# Validator and head straight to Synthesis. Per the problem statement.
RESEARCH_CONFIDENCE_THRESHOLD = 6.0


def route_after_clarity(
    state: ResearchState,
) -> Literal["interrupt_node", "research_agent"]:
    """Clarity Agent → Interrupt (if unclear) OR Research Agent (if clear)."""
    if state.clarity_status == "needs_clarification":
        return "interrupt_node"
    return "research_agent"


def route_after_research(
    state: ResearchState,
) -> Literal["validator_agent", "synthesis_agent"]:
    """Research Agent → Validator (low confidence) OR Synthesis (high confidence)."""
    if (
        state.confidence_score is not None
        and state.confidence_score >= RESEARCH_CONFIDENCE_THRESHOLD
    ):
        return "synthesis_agent"
    return "validator_agent"


def route_after_validator(
    state: ResearchState,
) -> Literal["research_agent", "synthesis_agent"]:
    """Validator → Research (retry) OR Synthesis (sufficient OR cap reached).

    Single enforcement site for MAX_RESEARCH_ATTEMPTS. When the cap is hit we
    ship to synthesis with the validation notes attached so the user gets
    *something*, not a silent stall.
    """
    if state.validation_result == "sufficient":
        return "synthesis_agent"
    if state.research_attempts >= get_settings().MAX_RESEARCH_ATTEMPTS:
        return "synthesis_agent"
    return "research_agent"
