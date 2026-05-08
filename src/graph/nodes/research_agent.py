"""Research Agent — calls Tavily live search, digests hits with the LLM.

Increments research_attempts BEFORE calling out so the validator/cap loop
sees the right counter on retries. If Tavily returns zero hits the node
returns a sentinel ResearchFindings with confidence=1 (CLAUDE.md pattern #3
— never None) and lets the validator + cap handle the stall.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from opentelemetry import trace

from ...adapters.tavily import _format_hits, tavily_search
from ...config import get_settings
from ...llm.client import ainvoke_structured, estimate_cost, get_chat_model
from ..prompts import build_research_system
from ..state import ResearchFindings, ResearchState
from ._audit_helpers import safe_record_step

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def research_agent(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    with tracer.start_as_current_span("node.research_agent") as span:
        span.set_attribute("run_id", state.run_id)
        attempts = state.research_attempts + 1
        span.set_attribute("research_attempts", attempts)

        settings = get_settings()
        company = state.company or "(unspecified)"
        span.set_attribute("company", company)

        # Build the search query. On retries, the validation notes already
        # tell us where the previous attempt fell short — bake them into the
        # query so we don't re-fetch the same gap.
        search_query = f"{company} {state.user_query}".strip()
        if state.validation_notes:
            search_query += " " + " ".join(state.validation_notes)

        t0_search = perf_counter()
        hits = await tavily_search(search_query, k=settings.TAVILY_MAX_RESULTS)
        search_latency_ms = int((perf_counter() - t0_search) * 1000)
        span.set_attribute("tavily_hits", len(hits))

        if not hits:
            # Sentinel-not-None: stub findings with confidence 1, audit logs the miss.
            findings = ResearchFindings(
                recent_news="No recent search results available.",
                stock_info="No stock or financial information found.",
                key_developments="No key developments found in search results.",
                confidence_score=1.0,
            )
            await safe_record_step(
                config,
                state.run_id,
                "research_agent",
                search_latency_ms,
                metadata={
                    "company": company,
                    "tavily_hits": 0,
                    "attempt": attempts,
                    "stub": True,
                },
            )
            return {
                "research_findings": findings.model_dump(exclude={"confidence_score"}),
                "confidence_score": findings.confidence_score,
                "research_attempts": attempts,
                "audit_log": [
                    {
                        "node": "research_agent",
                        "attempt": attempts,
                        "tavily_hits": 0,
                        "stub": True,
                    }
                ],
            }

        prior_notes_block = (
            "Prior validation notes:\n  - " + "\n  - ".join(state.validation_notes)
            if state.validation_notes
            else "Prior validation notes: (none — this is the first attempt)"
        )
        prompt_messages = [
            SystemMessage(content=build_research_system()),
            HumanMessage(
                content=(
                    f"Company: {company}\n"
                    f"User question: {state.user_query}\n\n"
                    f"{prior_notes_block}\n\n"
                    f"Search results:\n{_format_hits(hits)}"
                )
            ),
        ]

        llm = get_chat_model("primary")
        t0_llm = perf_counter()
        try:
            findings, usage = await ainvoke_structured(
                llm, prompt_messages, ResearchFindings
            )
        except ValueError as exc:
            llm_latency_ms = int((perf_counter() - t0_llm) * 1000)
            logger.error(
                "research_agent parse failure — emitting stub",
                extra={"run_id": state.run_id, "error": str(exc)},
            )
            findings = ResearchFindings(
                recent_news="Research output failed validation; retrying.",
                stock_info="Unavailable on this attempt.",
                key_developments="Unavailable on this attempt.",
                confidence_score=2.0,
            )
            await safe_record_step(
                config,
                state.run_id,
                "research_agent",
                search_latency_ms + llm_latency_ms,
                metadata={"parse_error": str(exc), "attempt": attempts},
            )
            return {
                "research_findings": findings.model_dump(exclude={"confidence_score"}),
                "confidence_score": findings.confidence_score,
                "research_attempts": attempts,
                "audit_log": [
                    {
                        "node": "research_agent",
                        "attempt": attempts,
                        "parse_error": str(exc),
                    }
                ],
            }

        llm_latency_ms = int((perf_counter() - t0_llm) * 1000)
        total_latency_ms = search_latency_ms + llm_latency_ms
        cost_usd = estimate_cost(settings.MODEL_PRIMARY, usage)
        in_tokens = int((usage or {}).get("input_tokens", 0))
        out_tokens = int((usage or {}).get("output_tokens", 0))

        span.set_attribute("latency_ms", total_latency_ms)
        span.set_attribute("cost_usd", round(cost_usd, 6))
        span.set_attribute("confidence_score", findings.confidence_score)

        await safe_record_step(
            config,
            state.run_id,
            "research_agent",
            total_latency_ms,
            prompt_tokens=in_tokens,
            completion_tokens=out_tokens,
            cost_usd=cost_usd,
            metadata={
                "company": company,
                "tavily_hits": len(hits),
                "attempt": attempts,
                "confidence_score": findings.confidence_score,
            },
        )

        return {
            "research_findings": findings.model_dump(exclude={"confidence_score"}),
            "confidence_score": findings.confidence_score,
            "research_attempts": attempts,
            "total_cost_usd": state.total_cost_usd + cost_usd,
            "audit_log": [
                {
                    "node": "research_agent",
                    "attempt": attempts,
                    "tavily_hits": len(hits),
                    "latency_ms": total_latency_ms,
                    "cost_usd": cost_usd,
                    "confidence_score": findings.confidence_score,
                }
            ],
        }
