"""Research Agent — query rewrite → Tavily → rerank → digest.

Pipeline (Tier 2):
  1. rewrite_query (flash LLM)   — clean conversational input for retrieval
  2. tavily_search               — over-fetch (k=TAVILY_MAX_RESULTS)
  3. rerank                      — cross-encoder, keep top-K (RERANK_TOP_K)
  4. ainvoke_structured digest   — flash LLM produces ResearchFindings + score

Increments research_attempts BEFORE calling out so the validator/cap loop
sees the right counter on retries. Sentinel-not-None pattern (CLAUDE.md #3)
when retrieval comes back empty: stub findings with confidence=1, validator
+ cap handle the stall.
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
from ...memory import retrieve_relevant_facts
from ...retrieval.query_rewriting import rewrite_query
from ...retrieval.rerank import rerank
from ..prompts import build_research_system
from ..state import ResearchFindings, ResearchState

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

        # 0. Long-term memory lookup. Pull any per-user facts relevant to
        # this query so the digest can adapt tone / focus to known prefs.
        # Empty list when memory is disabled OR the user has no prior facts.
        configurable = config.get("configurable") or {}
        user_id = str(configurable.get("user_id") or "anonymous")
        relevant_facts: list[str] = []
        if settings.LONGTERM_MEMORY_ENABLED:
            relevant_facts = await retrieve_relevant_facts(
                user_id, state.user_query, k=settings.LONGTERM_MEMORY_TOP_K
            )
        span.set_attribute("memory_facts_retrieved", len(relevant_facts))

        # 1. Query rewriting — turn conversational input into a search query.
        # Cost is rolled into the same total_cost_usd accumulator as the digest.
        t0_rewrite = perf_counter()
        rewritten_query, rewrite_usage, rewrite_rationale = await rewrite_query(
            state.user_query, state.company, state.messages
        )
        rewrite_latency_ms = int((perf_counter() - t0_rewrite) * 1000)
        rewrite_cost = estimate_cost(settings.MODEL_PRIMARY, rewrite_usage)
        # Append validation notes for retries — the rewriter doesn't see them.
        search_query = rewritten_query
        if state.validation_notes:
            search_query += " " + " ".join(state.validation_notes)
        span.set_attribute("rewritten_query", rewritten_query[:200])

        # 2. Tavily over-fetches — rerank trims to top_k.
        t0_search = perf_counter()
        hits_raw = await tavily_search(search_query, k=settings.TAVILY_MAX_RESULTS)
        search_latency_ms = int((perf_counter() - t0_search) * 1000)
        span.set_attribute("tavily_hits_raw", len(hits_raw))

        # 3. Rerank with cross-encoder; pass-through when disabled.
        t0_rerank = perf_counter()
        hits = await rerank(rewritten_query, hits_raw, top_k=settings.RERANK_TOP_K)
        rerank_latency_ms = int((perf_counter() - t0_rerank) * 1000)
        span.set_attribute("tavily_hits", len(hits))
        span.set_attribute("rerank_enabled", settings.RERANK_ENABLED)

        if not hits:
            findings = ResearchFindings(
                recent_news="No recent search results available.",
                stock_info="No stock or financial information found.",
                key_developments="No key developments found in search results.",
                confidence_score=1.0,
            )
            total_latency_ms = (
                rewrite_latency_ms + search_latency_ms + rerank_latency_ms
            )
            return {
                "research_findings": findings.model_dump(exclude={"confidence_score"}),
                "confidence_score": findings.confidence_score,
                "research_attempts": attempts,
                "total_cost_usd": state.total_cost_usd + rewrite_cost,
                "audit_log": [
                    {
                        "node": "research_agent",
                        "attempt": attempts,
                        "tavily_hits": 0,
                        "stub": True,
                        "rewritten_query": rewritten_query,
                    }
                ],
            }

        # 4. LLM digest of the top-K hits.
        prior_notes_block = (
            "Prior validation notes:\n  - " + "\n  - ".join(state.validation_notes)
            if state.validation_notes
            else "Prior validation notes: (none — this is the first attempt)"
        )
        memory_block = (
            "Things you know about this user (from prior sessions):\n  - "
            + "\n  - ".join(relevant_facts)
            if relevant_facts
            else "Things you know about this user: (no prior preferences on file)"
        )
        prompt_messages = [
            SystemMessage(content=build_research_system()),
            HumanMessage(
                content=(
                    f"Company: {company}\n"
                    f"User question: {state.user_query}\n"
                    f"Rewritten search query: {rewritten_query}\n\n"
                    f"{memory_block}\n\n"
                    f"{prior_notes_block}\n\n"
                    f"Top-{len(hits)} reranked search results:\n{_format_hits(hits)}"
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
            total_latency_ms = (
                rewrite_latency_ms + search_latency_ms + rerank_latency_ms + llm_latency_ms
            )
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
            return {
                "research_findings": findings.model_dump(exclude={"confidence_score"}),
                "confidence_score": findings.confidence_score,
                "research_attempts": attempts,
                "total_cost_usd": state.total_cost_usd + rewrite_cost,
                "audit_log": [
                    {
                        "node": "research_agent",
                        "attempt": attempts,
                        "parse_error": str(exc),
                    }
                ],
            }

        llm_latency_ms = int((perf_counter() - t0_llm) * 1000)
        total_latency_ms = (
            rewrite_latency_ms + search_latency_ms + rerank_latency_ms + llm_latency_ms
        )
        digest_cost = estimate_cost(settings.MODEL_PRIMARY, usage)
        node_cost = rewrite_cost + digest_cost
        in_tokens = int((usage or {}).get("input_tokens", 0))
        out_tokens = int((usage or {}).get("output_tokens", 0))

        span.set_attribute("latency_ms", total_latency_ms)
        span.set_attribute("cost_usd", round(node_cost, 6))
        span.set_attribute("rewrite_cost_usd", round(rewrite_cost, 6))
        span.set_attribute("input_tokens", in_tokens)
        span.set_attribute("output_tokens", out_tokens)
        span.set_attribute("model", settings.MODEL_PRIMARY)
        span.set_attribute("confidence_score", findings.confidence_score)

        return {
            "research_findings": findings.model_dump(exclude={"confidence_score"}),
            "confidence_score": findings.confidence_score,
            "research_attempts": attempts,
            "total_cost_usd": state.total_cost_usd + node_cost,
            "audit_log": [
                {
                    "node": "research_agent",
                    "attempt": attempts,
                    "tavily_hits": len(hits),
                    "rewritten_query": rewritten_query,
                    "rewrite_rationale": rewrite_rationale,
                    "rerank_enabled": settings.RERANK_ENABLED,
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "model": settings.MODEL_PRIMARY,
                    "memory_facts_retrieved": len(relevant_facts),
                    "user_id": user_id,
                    "latency_ms": total_latency_ms,
                    "cost_usd": node_cost,
                    "confidence_score": findings.confidence_score,
                }
            ],
        }
