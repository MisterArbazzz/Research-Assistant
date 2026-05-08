"""Clarity Agent — first node in the graph.

Decides whether the user's query is specific enough to research. If not, sets
clarity_status='needs_clarification' and routing sends the graph to the
interrupt node, which pauses and surfaces a question to the human.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from opentelemetry import trace

from ...config import get_settings
from ...llm.client import ainvoke_structured, estimate_cost, get_chat_model
from ..prompts import build_clarity_system, format_history
from ..state import ParsedClarity, ResearchState

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def clarity_agent(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    with tracer.start_as_current_span("node.clarity_agent") as span:
        span.set_attribute("run_id", state.run_id)
        span.set_attribute("user_query", state.user_query[:200])

        settings = get_settings()
        history_block = format_history(state.messages[:-1] if state.messages else [])
        prompt_messages = [
            SystemMessage(content=build_clarity_system()),
            HumanMessage(
                content=(
                    f"Conversation history:\n{history_block}\n\n"
                    f"Current user query: {state.user_query}"
                )
            ),
        ]

        llm = get_chat_model("primary")
        t0 = perf_counter()
        try:
            parsed, usage = await ainvoke_structured(llm, prompt_messages, ParsedClarity)
        except ValueError as exc:
            # ainvoke_structured raises ValueError when Gemini's output won't parse.
            # CLAUDE.md pattern: don't crash the run — degrade to "clear" so the
            # research agent at least tries. Log loudly so we notice.
            latency_ms = int((perf_counter() - t0) * 1000)
            logger.error(
                "clarity_agent parse failure — defaulting to 'clear'",
                extra={"run_id": state.run_id, "error": str(exc)},
            )
            return {
                "clarity_status": "clear",
                "audit_log": [
                    {
                        "node": "clarity_agent",
                        "latency_ms": latency_ms,
                        "parse_error": str(exc),
                    }
                ],
            }

        latency_ms = int((perf_counter() - t0) * 1000)
        cost_usd = estimate_cost(settings.MODEL_PRIMARY, usage)
        in_tokens = int((usage or {}).get("input_tokens", 0))
        out_tokens = int((usage or {}).get("output_tokens", 0))

        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("cost_usd", round(cost_usd, 6))
        span.set_attribute("input_tokens", in_tokens)
        span.set_attribute("output_tokens", out_tokens)
        span.set_attribute("model", settings.MODEL_PRIMARY)
        span.set_attribute("clarity_status", parsed.clarity_status)

        return {
            "clarity_status": parsed.clarity_status,
            "company": parsed.company,
            "clarification_request": parsed.clarification_request,
            "total_cost_usd": state.total_cost_usd + cost_usd,
            "audit_log": [
                {
                    "node": "clarity_agent",
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "model": settings.MODEL_PRIMARY,
                    "clarity_status": parsed.clarity_status,
                    "company": parsed.company,
                }
            ],
        }
