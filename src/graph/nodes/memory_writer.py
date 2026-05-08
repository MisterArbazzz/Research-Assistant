"""Memory Writer — selective writes into long-term memory after synthesis.

Runs after the answer has been produced. Asks Flash a single question:
"Did this conversation reveal any DURABLE user preferences worth
remembering for future sessions?" and writes whatever it returns.

Selective writes is the key idea — we don't dump full transcripts into
long-term memory (that would just bloat retrieval). We write atomic
fact-strings the model judges worth keeping ("user prefers technical
detail", "user is researching EV companies"). Empty list is a normal
answer; most turns produce nothing to save.

Skipped entirely when LONGTERM_MEMORY_ENABLED=false. Failures are logged
and swallowed — losing a fact-write is not worth crashing the run.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from opentelemetry import trace
from pydantic import BaseModel, Field

from ...config import get_settings
from ...llm.client import ainvoke_structured, estimate_cost, get_chat_model
from ...memory import store_fact
from ..prompts import format_history
from ..state import ResearchState

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class _MemoryExtraction(BaseModel):
    facts: list[str] = Field(
        default_factory=list,
        description=(
            "Atomic, durable user-facts worth remembering across sessions. "
            "Empty list when nothing in the conversation rises to that bar."
        ),
    )
    rationale: str = Field(default="")


_EXTRACTION_PROMPT = (
    "You are the Memory Writer. Inspect the latest user turn and decide if "
    "it revealed any DURABLE preference, identity, or recurring interest "
    "worth remembering for FUTURE sessions (different thread_id) about this "
    "user.\n\n"
    "WRITE when the user reveals:\n"
    "  - Communication preferences ('I prefer technical detail', 'keep "
    "answers brief')\n"
    "  - Persistent interests ('I follow EV stocks', 'I'm researching "
    "semiconductor companies')\n"
    "  - Identity / role hints ('I'm a portfolio manager', 'I'm a student')\n"
    "  - Watchlists or focal companies they mention repeatedly\n\n"
    "DO NOT WRITE:\n"
    "  - One-off questions ('tell me about Apple') — that's just a query, "
    "not a preference\n"
    "  - Restatements of facts the assistant already knows about companies\n"
    "  - Vague feedback ('thanks', 'cool')\n\n"
    "Each fact you return must be a single short sentence in third-person "
    "form ('The user prefers technical detail over high-level summaries'). "
    "Most turns produce ZERO facts — that's fine and expected. Return an "
    "empty list rather than padding."
)


def _get_user_id(config: RunnableConfig) -> str:
    """Extract user_id from RunnableConfig, defaulting to 'anonymous'."""
    configurable = config.get("configurable") or {}
    return str(configurable.get("user_id") or "anonymous")


async def memory_writer(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    settings = get_settings()
    if not settings.LONGTERM_MEMORY_ENABLED:
        return {
            "audit_log": [
                {"node": "memory_writer", "skipped": "LONGTERM_MEMORY_ENABLED=false"}
            ]
        }

    with tracer.start_as_current_span("node.memory_writer") as span:
        span.set_attribute("run_id", state.run_id)
        user_id = _get_user_id(config)
        span.set_attribute("user_id", user_id)

        history_block = format_history(state.messages, limit=8)
        prompt = [
            SystemMessage(content=_EXTRACTION_PROMPT),
            HumanMessage(
                content=(
                    f"User identifier: {user_id}\n"
                    f"Most recent conversation:\n{history_block}\n\n"
                    f"Latest user query: {state.user_query}\n"
                    f"Final assistant answer: {state.final_answer}"
                )
            ),
        ]

        llm = get_chat_model("primary")
        t0 = perf_counter()
        try:
            extracted, usage = await ainvoke_structured(
                llm, prompt, _MemoryExtraction
            )
        except ValueError as exc:
            latency_ms = int((perf_counter() - t0) * 1000)
            logger.warning(
                "memory extraction parse failure — skipping write",
                extra={"run_id": state.run_id, "error": str(exc)},
            )
            return {
                "audit_log": [
                    {
                        "node": "memory_writer",
                        "latency_ms": latency_ms,
                        "parse_error": str(exc),
                    }
                ]
            }

        latency_ms = int((perf_counter() - t0) * 1000)
        cost_usd = estimate_cost(settings.MODEL_PRIMARY, usage)
        in_tokens = int((usage or {}).get("input_tokens", 0))
        out_tokens = int((usage or {}).get("output_tokens", 0))

        # Selective write: store each extracted fact. Failures per-fact are
        # swallowed (warning logged) — one bad write shouldn't block others.
        written = 0
        for fact in extracted.facts:
            try:
                await store_fact(user_id, fact, source_thread_id=state.run_id)
                written += 1
            except Exception as exc:
                logger.warning(
                    "store_fact failed",
                    extra={"user_id": user_id, "fact": fact, "error": str(exc)},
                )

        span.set_attribute("facts_written", written)
        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("cost_usd", round(cost_usd, 6))

        return {
            "total_cost_usd": state.total_cost_usd + cost_usd,
            "audit_log": [
                {
                    "node": "memory_writer",
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "model": settings.MODEL_PRIMARY,
                    "facts_written": written,
                    "facts": extracted.facts,
                }
            ],
        }
