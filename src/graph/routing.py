"""Conditional routing functions for the Research Assistant graph.

Per CLAUDE.md pattern #4: routing is pure — no LLM calls, no I/O. Iteration
caps live in src/config.py and are read once here at the single enforcement
site. A grep for MAX_RESEARCH_ATTEMPTS / COST_CEILING_PER_RUN_USD should find
each of them in two places: definition in config.py and enforcement here.
"""

from __future__ import annotations

from typing import Literal

from ..config import get_settings
from .state import ResearchState

# Confidence threshold above which the Research Agent's findings skip the
# Validator and head straight to Synthesis. Per the problem statement.
RESEARCH_CONFIDENCE_THRESHOLD = 6.0


def is_cost_ceiling_reached(state: ResearchState) -> bool:
    """True when this run has spent at or above the configured per-run cap.

    Pure helper used by the routing functions to short-circuit further LLM
    calls. The cap doesn't kill the run — it routes straight to synthesis so
    the user always gets *some* answer, even if it's based on partial data.
    """
    return state.total_cost_usd >= get_settings().COST_CEILING_PER_RUN_USD


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
    """Research Agent → Validator (low confidence) OR Synthesis.

    Synthesis is preferred when (a) confidence is high enough to skip QA,
    or (b) the cost ceiling has been reached. Either way we want to put a
    final answer in front of the user instead of burning more budget.
    """
    if is_cost_ceiling_reached(state):
        return "synthesis_agent"
    if (
        state.confidence_score is not None
        and state.confidence_score >= RESEARCH_CONFIDENCE_THRESHOLD
    ):
        return "synthesis_agent"
    return "validator_agent"


def route_after_validator(
    state: ResearchState,
) -> Literal["research_agent", "synthesis_agent"]:
    """Validator → Research (retry) OR Synthesis (sufficient / cap / ceiling).

    Three short-circuit paths to synthesis:
      1. Validator says findings are sufficient.
      2. Research-attempt cap reached — ship with notes attached.
      3. Cost ceiling reached — ship rather than burn more budget.
    """
    if state.validation_result == "sufficient":
        return "synthesis_agent"
    if state.research_attempts >= get_settings().MAX_RESEARCH_ATTEMPTS:
        return "synthesis_agent"
    if is_cost_ceiling_reached(state):
        return "synthesis_agent"
    return "research_agent"
