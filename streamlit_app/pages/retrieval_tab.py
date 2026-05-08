"""Retrieval Lens — three-stage funnel deep-dive (Tier 2)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.adapters.tavily import TavilyHit, tavily_search
from src.config import get_settings
from src.retrieval.query_rewriting import rewrite_query
from src.retrieval.rerank import rerank
from streamlit_app.state import run_async


def _render_hits(hits: list[TavilyHit], label: str, baseline: list[TavilyHit] | None = None) -> None:
    """Render hits as a list with title, score, URL, and snippet."""
    if not hits:
        st.info(f"No hits for stage: {label}")
        return

    # Build a lookup of (url → original rank) so we can show position deltas after rerank.
    baseline_pos: dict[str, int] = {}
    if baseline:
        for i, h in enumerate(baseline):
            baseline_pos[h.url] = i

    for i, hit in enumerate(hits):
        pos_label = f"#{i + 1}"
        if baseline_pos:
            old = baseline_pos.get(hit.url)
            if old is not None and old != i:
                delta = old - i
                arrow = f"⬆️ +{delta}" if delta > 0 else f"⬇️ {delta}"
                pos_label = f"{pos_label} ({arrow})"

        with st.container(border=True):
            top = st.columns([6, 1])
            top[0].markdown(f"**{pos_label} — {hit.title}**")
            top[1].metric("score", f"{hit.score:.3f}")
            st.caption(hit.url)
            st.markdown(hit.content[:500] + ("…" if len(hit.content) > 500 else ""))


def render() -> None:
    st.subheader("Retrieval Lens — Tier 2 funnel")
    st.caption(
        "Run the three retrieval stages on any query, isolated from the "
        "graph: **rewrite → Tavily → rerank**. See exactly what the "
        "Research Agent sees before it digests."
    )

    settings = get_settings()

    cols = st.columns([4, 2, 1])
    with cols[0]:
        query = st.text_input(
            "Query",
            value=st.session_state.get("last_user_query") or "Apple iPhone earnings",
        )
    with cols[1]:
        company_hint = st.text_input(
            "Company hint (optional)",
            value="",
            help="Helps the rewriter resolve pronouns. Leave empty to test cold.",
        )
    with cols[2]:
        use_history = st.toggle(
            "Use chat history",
            value=False,
            help="Feed the rewriter the current conversation history (sets context).",
        )

    if not st.button("🔍 Run retrieval funnel", type="primary"):
        return
    if not query.strip():
        st.warning("Enter a query first.")
        return

    history = st.session_state.get("messages") if use_history else []
    # Reuse the conversation history from session_state, but coerce to BaseMessage shape.
    # The rewriter only reads .content — `format_history` accepts a list[BaseMessage].
    from langchain_core.messages import AIMessage, HumanMessage

    history_msgs: list[Any] = []
    for m in history or []:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        history_msgs.append(cls(content=m["content"]))

    # Stage 1 — query rewrite.
    with st.status("Stage 1 — Query rewrite (Gemini Flash)", state="running"):
        rewritten, usage, rationale = run_async(
            rewrite_query(query, company_hint or None, history_msgs)
        )
        st.markdown(f"**Rewritten:** `{rewritten}`")
        st.markdown(f"**Rationale:** _{rationale or '(none)'}_")
        if usage:
            st.caption(
                f"input_tokens={usage.get('input_tokens', 0)}, "
                f"output_tokens={usage.get('output_tokens', 0)}"
            )

    # Stage 2 — Tavily.
    with st.status(
        f"Stage 2 — Tavily search (k={settings.TAVILY_MAX_RESULTS})",
        state="running",
    ):
        raw_hits = run_async(tavily_search(rewritten, k=settings.TAVILY_MAX_RESULTS))
        st.caption(
            f"Backend: **{settings.RESEARCH_BACKEND}** · "
            f"Returned **{len(raw_hits)}** hits"
        )
        _render_hits(raw_hits, "Tavily raw")

    # Stage 3 — rerank.
    with st.status(
        f"Stage 3 — Cross-encoder rerank (top-{settings.RERANK_TOP_K})",
        state="running",
    ):
        if not settings.RERANK_ENABLED:
            st.warning("RERANK_ENABLED is off — pass-through trim to top-K only.")
        reranked = run_async(rerank(rewritten, raw_hits, top_k=settings.RERANK_TOP_K))
        st.caption(
            f"Reranker: **{settings.RERANK_MODEL}** · "
            f"Returned **{len(reranked)}** hits (arrows show position deltas vs Tavily)"
        )
        _render_hits(reranked, "reranked", baseline=raw_hits)
