"""Eval metrics — RAGAS-style scoring for the Research Assistant.

Two deterministic metrics (no LLM, fast, exact):
  - context_recall:    did the retrieved hits contain the expected company?
  - context_precision: how many top-K hits are relevant to the company?

Two LLM-as-judge metrics (1 Gemini Pro call each, costlier):
  - faithfulness:      does the final_answer claim only facts present in the
                       research_findings? Hallucination detector.
  - answer_relevance:  does the final_answer address the user's actual
                       question, or talk past it?

All metrics return a float in [0.0, 1.0]. Aggregation is mean across cases.
The two LLM judges share a Pro client; per-case cost is ~$0.005-0.010.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.adapters.tavily import TavilyHit
from src.llm.client import ainvoke_structured, get_chat_model

logger = logging.getLogger(__name__)


# -- Deterministic metrics --


def context_recall(hits: list[TavilyHit], expected_company: str | None) -> float:
    """1.0 if the expected company is mentioned in any hit's title/content; else 0.0.

    `expected_company=None` means the case has no canonical answer (unknown
    company); we score it 1.0 trivially since recall isn't meaningful.
    """
    if expected_company is None:
        return 1.0
    if not hits:
        return 0.0
    needle = expected_company.lower().split()[0]  # first word — works for "Apple Inc.", "Tesla", "NVIDIA"
    for hit in hits:
        haystack = (hit.title + " " + hit.content).lower()
        if needle in haystack:
            return 1.0
    return 0.0


def context_precision(hits: list[TavilyHit], expected_company: str | None) -> float:
    """Fraction of top-K hits that mention the expected company.

    For mock-backend evals every hit is on-topic by construction, so this
    metric is mainly a sanity check that retrieval didn't return nonsense.
    """
    if expected_company is None:
        return 1.0
    if not hits:
        return 0.0
    needle = expected_company.lower().split()[0]
    on_topic = sum(1 for h in hits if needle in (h.title + " " + h.content).lower())
    return on_topic / len(hits)


# -- LLM-as-judge metrics --


class _BinaryJudgment(BaseModel):
    """0.0 to 1.0 score with a one-line rationale."""

    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="")


_FAITHFULNESS_PROMPT = (
    "You are an impartial evaluator. Your job is to detect HALLUCINATIONS — "
    "claims in the final answer that are NOT supported by the research findings.\n\n"
    "Score from 0.0 to 1.0:\n"
    "  - 1.0: every factual claim in the answer is directly supported by the "
    "research findings\n"
    "  - 0.7-0.9: minor unsupported elaborations (e.g. tone, framing) but no "
    "fabricated facts\n"
    "  - 0.4-0.6: at least one unsupported factual claim, but most content is "
    "grounded\n"
    "  - 0.0-0.3: significant fabrication; the answer invents facts not in "
    "the research\n\n"
    "Be strict about specific numbers, names, dates, and quotes. Generic "
    "framing ('this is positive', 'investors are watching') is fine even if "
    "not literally in the research."
)

_RELEVANCE_PROMPT = (
    "You are an impartial evaluator. Your job is to assess whether the final "
    "answer ACTUALLY ADDRESSES the user's question.\n\n"
    "Score from 0.0 to 1.0:\n"
    "  - 1.0: directly answers the question with appropriate specificity\n"
    "  - 0.7-0.9: answers the question but with minor tangential content\n"
    "  - 0.4-0.6: partially addresses the question; misses an obvious aspect\n"
    "  - 0.0-0.3: talks past the question; answer is about something else\n\n"
    "If the user asked about the CEO and the answer is about products, that's "
    "low relevance. If the user asked about news and the answer leads with "
    "stock data, that's medium relevance."
)


async def faithfulness(
    user_query: str,
    research_findings: dict[str, Any] | None,
    final_answer: str | None,
) -> tuple[float, str]:
    """LLM-as-judge: does the answer fabricate facts beyond the research?"""
    if not final_answer or not research_findings:
        return 0.0, "missing answer or findings"
    llm = get_chat_model("qa")
    prompt = [
        SystemMessage(content=_FAITHFULNESS_PROMPT),
        HumanMessage(
            content=(
                f"User question: {user_query}\n\n"
                f"Research findings (the only source of truth):\n{research_findings}\n\n"
                f"Final answer to evaluate:\n{final_answer}"
            )
        ),
    ]
    try:
        verdict, _ = await ainvoke_structured(llm, prompt, _BinaryJudgment)
    except ValueError as exc:
        logger.warning("faithfulness judge parse failure", extra={"error": str(exc)})
        return 0.5, f"judge parse failure: {exc}"
    return float(verdict.score), verdict.rationale


async def answer_relevance(
    user_query: str,
    final_answer: str | None,
    prior_queries: list[str] | None = None,
) -> tuple[float, str]:
    """LLM-as-judge: does the answer address the user's question?

    `prior_queries` (optional) is the conversation history before the current
    turn — required for fair scoring on follow-ups where pronouns ('it',
    'they') only resolve in context.
    """
    if not final_answer:
        return 0.0, "no answer produced"
    llm = get_chat_model("qa")
    history_block = (
        "Prior conversation turns:\n  - " + "\n  - ".join(prior_queries)
        if prior_queries
        else "Prior conversation turns: (none — this is the first turn)"
    )
    prompt = [
        SystemMessage(content=_RELEVANCE_PROMPT),
        HumanMessage(
            content=(
                f"{history_block}\n\n"
                f"Current user question: {user_query}\n\n"
                f"Final answer:\n{final_answer}"
            )
        ),
    ]
    try:
        verdict, _ = await ainvoke_structured(llm, prompt, _BinaryJudgment)
    except ValueError as exc:
        logger.warning("relevance judge parse failure", extra={"error": str(exc)})
        return 0.5, f"judge parse failure: {exc}"
    return float(verdict.score), verdict.rationale
