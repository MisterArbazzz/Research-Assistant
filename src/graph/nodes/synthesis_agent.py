"""Synthesis Agent — produces the final user-facing answer.

Free-form generation (no structured output). Sees the conversation history,
the research findings, and (when present) the validator's notes for cap-exit
runs so the answer can flag gaps to the user instead of pretending coverage.
The AI message is appended to state.messages via the add_messages reducer,
keeping multi-turn memory intact for follow-ups.
"""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from opentelemetry import trace

from ...config import get_settings
from ...llm.client import ainvoke_with_retry, estimate_cost, get_chat_model
from ..prompts import build_synthesis_system, format_history
from ..state import ResearchState
from ._audit_helpers import safe_record_step

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def synthesis_agent(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    with tracer.start_as_current_span("node.synthesis_agent") as span:
        span.set_attribute("run_id", state.run_id)

        settings = get_settings()
        history_block = format_history(state.messages, limit=10)
        findings_block = json.dumps(state.research_findings or {}, indent=2)
        notes_block = (
            "Validator notes (gaps the answer should acknowledge):\n  - "
            + "\n  - ".join(state.validation_notes)
            if state.validation_notes
            and state.validation_result == "insufficient"
            else ""
        )

        prompt_messages = [
            SystemMessage(content=build_synthesis_system()),
            HumanMessage(
                content=(
                    f"Conversation history:\n{history_block}\n\n"
                    f"User question: {state.user_query}\n"
                    f"Company: {state.company}\n\n"
                    f"Research findings:\n{findings_block}\n\n"
                    f"{notes_block}"
                )
            ),
        ]

        llm = get_chat_model("primary")
        t0 = perf_counter()
        try:
            response = await ainvoke_with_retry(llm, prompt_messages)
        except Exception as exc:  # noqa: BLE001 — surface a graceful answer
            latency_ms = int((perf_counter() - t0) * 1000)
            logger.exception(
                "synthesis_agent LLM call failed",
                extra={"run_id": state.run_id, "error": str(exc)},
            )
            fallback_answer = (
                "I ran into a technical issue producing a polished answer, but "
                f"here's what the research found about {state.company}: "
                + json.dumps(state.research_findings or {}, indent=2)
            )
            await safe_record_step(
                config,
                state.run_id,
                "synthesis_agent",
                latency_ms,
                metadata={"error": str(exc)},
            )
            return {
                "final_answer": fallback_answer,
                "messages": [AIMessage(content=fallback_answer)],
                "audit_log": [
                    {
                        "node": "synthesis_agent",
                        "latency_ms": latency_ms,
                        "error": str(exc),
                    }
                ],
            }

        latency_ms = int((perf_counter() - t0) * 1000)
        usage = getattr(response, "usage_metadata", None)
        cost_usd = estimate_cost(settings.MODEL_PRIMARY, usage)
        in_tokens = int((usage or {}).get("input_tokens", 0))
        out_tokens = int((usage or {}).get("output_tokens", 0))

        answer_text = (
            response.content if isinstance(response.content, str) else str(response.content)
        )

        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("cost_usd", round(cost_usd, 6))
        span.set_attribute("answer_chars", len(answer_text))

        await safe_record_step(
            config,
            state.run_id,
            "synthesis_agent",
            latency_ms,
            prompt_tokens=in_tokens,
            completion_tokens=out_tokens,
            cost_usd=cost_usd,
            metadata={"answer_chars": len(answer_text)},
        )

        return {
            "final_answer": answer_text,
            "messages": [AIMessage(content=answer_text)],
            "total_cost_usd": state.total_cost_usd + cost_usd,
            "audit_log": [
                {
                    "node": "synthesis_agent",
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "answer_chars": len(answer_text),
                }
            ],
        }
