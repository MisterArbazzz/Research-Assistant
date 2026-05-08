"""Trajectory evaluation — LLM-as-judge over the full audit log.

Where the RAGAS metrics in metrics.py judge a single answer, trajectory eval
judges the SHAPE OF THE RUN: did clarity catch ambiguity correctly, did the
validator's retry signal lead to better confidence, was the path efficient,
was the final answer grounded? One Pro call per case; ~$0.005-0.010.

Decoupling from the RAGAS metrics matters for diagnosis: a trajectory drop
points at the orchestration logic (routing, validator, retry loop), while a
faithfulness drop points at the synthesis prompt. Two different fixes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.llm.client import ainvoke_structured, get_chat_model

logger = logging.getLogger(__name__)


class TrajectoryVerdict(BaseModel):
    """Structured judgment of a single run's trajectory."""

    clarity_correct: bool = Field(
        description="Did the Clarity Agent correctly route? (clear → research, ambiguous → interrupt)"
    )
    retry_helpful: bool = Field(
        description=(
            "If the validator triggered a retry, did confidence go up? "
            "True if no retry happened (vacuously)."
        )
    )
    efficiency_score: float = Field(
        ge=0.0,
        le=10.0,
        description="0-10. 10 = minimum nodes to a good answer; 0 = wasted hops.",
    )
    grounding_score: float = Field(
        ge=0.0,
        le=10.0,
        description="0-10. 10 = answer grounded in research; 0 = hallucinated.",
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description="0-10. Holistic judgment combining all four signals.",
    )
    rationale: str = Field(default="")


_TRAJECTORY_PROMPT = (
    "You are an impartial evaluator scoring a multi-agent research run. The "
    "system has four agents (Clarity, Research, Validator, Synthesis) plus a "
    "human-in-the-loop interrupt for ambiguous queries. The validator can "
    "send research back for up to 3 attempts.\n\n"
    "You will see:\n"
    "  - the user's query (or queries, for multi-turn)\n"
    "  - the audit log (per-node breadcrumbs)\n"
    "  - the research findings\n"
    "  - the final answer\n\n"
    "Score the run on four dimensions, then give an overall 0-10 score:\n"
    "  - clarity_correct: did Clarity correctly classify the query? "
    "(ambiguous queries should interrupt; clear ones should not)\n"
    "  - retry_helpful: if Validator triggered a retry, did Research's "
    "confidence go up on the next attempt? (true if no retry happened)\n"
    "  - efficiency_score: was the trajectory efficient? Penalize unnecessary "
    "retries, repeated clarity asks, etc.\n"
    "  - grounding_score: is the final answer faithful to the research "
    "findings? Penalize fabricated specifics.\n"
    "  - overall_score: holistic 0-10. A 10 means the system did exactly "
    "what it should have.\n"
    "Provide a one-paragraph rationale."
)


async def evaluate_trajectory(
    user_queries: list[str],
    audit_log: list[dict[str, Any]],
    research_findings: dict[str, Any] | None,
    final_answer: str | None,
) -> TrajectoryVerdict:
    """Judge a full run's trajectory. Returns a structured TrajectoryVerdict."""
    llm = get_chat_model("qa")
    prompt = [
        SystemMessage(content=_TRAJECTORY_PROMPT),
        HumanMessage(
            content=(
                "User queries (in order):\n"
                + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(user_queries))
                + "\n\nAudit log:\n"
                + json.dumps(audit_log, indent=2, default=str)
                + "\n\nResearch findings:\n"
                + json.dumps(research_findings, indent=2, default=str)
                + "\n\nFinal answer:\n"
                + (final_answer or "(no answer produced)")
            )
        ),
    ]
    try:
        verdict, _ = await ainvoke_structured(llm, prompt, TrajectoryVerdict)
    except ValueError as exc:
        logger.warning("trajectory judge parse failure", extra={"error": str(exc)})
        return TrajectoryVerdict(
            clarity_correct=False,
            retry_helpful=False,
            efficiency_score=0.0,
            grounding_score=0.0,
            overall_score=0.0,
            rationale=f"judge parse failure: {exc}",
        )
    return verdict
