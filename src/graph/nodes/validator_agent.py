"""Validator Agent — judges whether research findings answer the user's question.

Uses MODEL_QA (gemini-2.5-pro) — judgment task per CLAUDE.md cost guidance.
The validator's `notes` get carried into the next research attempt's prompt
so the retry isn't blind: it sees exactly what the validator wanted fixed.
"""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from opentelemetry import trace

from ...config import get_settings
from ...llm.client import ainvoke_structured, estimate_cost, get_chat_model
from ..prompts import build_validator_system
from ..state import ResearchState, ValidationVerdict

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def validator_agent(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    with tracer.start_as_current_span("node.validator_agent") as span:
        span.set_attribute("run_id", state.run_id)
        span.set_attribute("research_attempts", state.research_attempts)
        span.set_attribute("confidence_score", state.confidence_score or 0.0)

        settings = get_settings()
        findings_block = json.dumps(state.research_findings or {}, indent=2)
        prompt_messages = [
            SystemMessage(content=build_validator_system(state.validation_notes)),
            HumanMessage(
                content=(
                    f"User question: {state.user_query}\n"
                    f"Company: {state.company}\n"
                    f"Confidence score from research agent: {state.confidence_score}\n"
                    f"Attempt number: {state.research_attempts}\n\n"
                    f"Research findings:\n{findings_block}"
                )
            ),
        ]

        llm = get_chat_model("qa")
        t0 = perf_counter()
        try:
            verdict, usage = await ainvoke_structured(
                llm, prompt_messages, ValidationVerdict
            )
        except ValueError as exc:
            latency_ms = int((perf_counter() - t0) * 1000)
            logger.error(
                "validator_agent parse failure — defaulting to 'sufficient' to avoid retry storm",
                extra={"run_id": state.run_id, "error": str(exc)},
            )
            return {
                "validation_result": "sufficient",
                "validation_notes": [
                    "Validator output failed parsing; shipping with current research."
                ],
                "audit_log": [
                    {
                        "node": "validator_agent",
                        "latency_ms": latency_ms,
                        "parse_error": str(exc),
                    }
                ],
            }

        latency_ms = int((perf_counter() - t0) * 1000)
        cost_usd = estimate_cost(settings.MODEL_QA, usage)
        in_tokens = int((usage or {}).get("input_tokens", 0))
        out_tokens = int((usage or {}).get("output_tokens", 0))

        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("cost_usd", round(cost_usd, 6))
        span.set_attribute("input_tokens", in_tokens)
        span.set_attribute("output_tokens", out_tokens)
        span.set_attribute("model", settings.MODEL_QA)
        span.set_attribute("validation_result", verdict.validation_result)

        return {
            "validation_result": verdict.validation_result,
            "validation_notes": verdict.notes,
            "total_cost_usd": state.total_cost_usd + cost_usd,
            "audit_log": [
                {
                    "node": "validator_agent",
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "validation_result": verdict.validation_result,
                    "notes": verdict.notes,
                }
            ],
        }
