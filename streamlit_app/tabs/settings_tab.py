"""Settings tab — live config toggles + A/B comparison widget.

Most flags are read at call-time via `get_settings()`, so toggling them
hot-swaps behavior without rebuilding the graph. The model picker and
RERANK_MODEL are captured by closures (LLM client) or singletons
(flashrank), so changing them flips a "requires reload" flag.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from streamlit_app.components import fmt_money, render_audit_entry
from streamlit_app.graph_runner import fresh_payload, stream_turn
from streamlit_app.state import (
    apply_settings_overrides,
    get_graph_and_saver,
    run_async,
    runtime_settings_summary,
)


def _ab_run(query: str) -> dict[str, Any]:
    """Re-run `query` against the current graph and return final state."""
    graph, _saver = get_graph_and_saver()
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": f"ab-{st.session_state['thread_id']}",
            "user_id": st.session_state["user_id"],
        }
    }
    payload = fresh_payload(query)
    return run_async(stream_turn(graph, payload, config, lambda *_args, **_kw: None))


def render() -> None:
    st.subheader("Settings — live config toggles")
    st.caption(
        "Most flags are read at call-time. The model picker and RERANK_MODEL "
        "require rebuilding the graph (use the **Reload graph** button)."
    )

    summary = runtime_settings_summary()

    st.markdown("### Feature toggles (live)")
    cols = st.columns(3)
    with cols[0]:
        rewrite = st.toggle(
            "Query rewriting",
            value=summary["rewrite_enabled"],
            help="Tier 2: pre-process conversational queries into search queries.",
        )
    with cols[1]:
        rerank = st.toggle(
            "Cross-encoder rerank",
            value=summary["rerank_enabled"],
            help="Tier 2: re-score Tavily hits by relevance to the rewritten query.",
        )
    with cols[2]:
        memory = st.toggle(
            "Long-term memory",
            value=summary["memory_enabled"],
            help="Tier 4: store/retrieve per-user facts across threads.",
        )

    st.markdown("### Caps & numeric (live)")
    cols2 = st.columns(4)
    with cols2[0]:
        max_attempts = st.number_input(
            "MAX_RESEARCH_ATTEMPTS",
            min_value=1,
            max_value=10,
            value=int(summary["max_attempts"]),
            help="Validator → research retry cap.",
        )
    with cols2[1]:
        ceiling = st.number_input(
            "COST_CEILING_PER_RUN_USD",
            min_value=0.0001,
            max_value=10.0,
            value=float(summary["cost_ceiling"]),
            step=0.001,
            format="%.4f",
            help="Set this to ~0.001 to demo the cost guard firing mid-run.",
        )
    with cols2[2]:
        rerank_top_k = st.number_input(
            "RERANK_TOP_K",
            min_value=1,
            max_value=20,
            value=int(summary["rerank_top_k"]),
        )
    with cols2[3]:
        tavily_max = st.number_input(
            "TAVILY_MAX_RESULTS",
            min_value=1,
            max_value=20,
            value=int(summary["tavily_max_results"]),
            help="Over-fetch — rerank trims to TOP_K.",
        )

    st.markdown("### Backend (live)")
    backend = st.radio(
        "RESEARCH_BACKEND",
        options=["tavily", "mock"],
        index=0 if summary["research_backend"] == "tavily" else 1,
        horizontal=True,
        help="Switch to **mock** to demo offline / deterministic behaviour.",
    )

    st.markdown("### Models (requires reload)")
    cols3 = st.columns(2)
    with cols3[0]:
        model_primary = st.selectbox(
            "MODEL_PRIMARY",
            options=["gemini-2.5-flash", "gemini-2.5-pro"],
            index=0 if summary["model_primary"] == "gemini-2.5-flash" else 1,
        )
    with cols3[1]:
        model_qa = st.selectbox(
            "MODEL_QA",
            options=["gemini-2.5-pro", "gemini-2.5-flash"],
            index=0 if summary["model_qa"] == "gemini-2.5-pro" else 1,
        )

    apply_cols = st.columns([1, 1, 3])
    with apply_cols[0]:
        if st.button("Apply (live)", type="primary", use_container_width=True):
            overrides = {
                "QUERY_REWRITE_ENABLED": str(rewrite).lower(),
                "RERANK_ENABLED": str(rerank).lower(),
                "LONGTERM_MEMORY_ENABLED": str(memory).lower(),
                "MAX_RESEARCH_ATTEMPTS": str(int(max_attempts)),
                "COST_CEILING_PER_RUN_USD": f"{ceiling:.4f}",
                "RERANK_TOP_K": str(int(rerank_top_k)),
                "TAVILY_MAX_RESULTS": str(int(tavily_max)),
                "RESEARCH_BACKEND": backend,
            }
            apply_settings_overrides(overrides)
            st.success("Live overrides applied — next query uses these.")
            st.rerun()
    with apply_cols[1]:
        if st.button("Reload graph", use_container_width=True):
            apply_settings_overrides(
                {
                    "MODEL_PRIMARY": model_primary,
                    "MODEL_QA": model_qa,
                }
            )
            st.cache_resource.clear()
            st.success("Graph + ranker rebuild on next query.")
            st.rerun()

    if st.session_state.get("settings_overrides"):
        with st.expander("Active environment overrides", expanded=False):
            st.json(st.session_state["settings_overrides"])

    st.divider()
    _render_ab_widget()


def _render_ab_widget() -> None:
    st.markdown("### 🆎 A/B compare — re-run the last query against current settings")
    st.caption(
        "Captures a snapshot of the last conversation answer; toggle settings "
        "above and run again to see them side by side."
    )

    last_q = st.session_state.get("last_user_query")
    last_a = st.session_state.get("last_final_answer")
    last_audit = st.session_state.get("last_run_audit") or []
    history: list[dict[str, Any]] = st.session_state.get("ab_history") or []

    if not last_q:
        st.info("Run at least one query in the Chat tab to enable A/B compare.")
        return

    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown(f"**Query:** _{last_q}_")
    with cols[1]:
        if st.button("Snapshot + re-run", type="secondary", use_container_width=True):
            # Step 1: snapshot the current result as variant A.
            history.append(
                {
                    "label": f"A · {_short_settings_label()}",
                    "query": last_q,
                    "answer": last_a,
                    "audit": last_audit,
                    "cost": sum(float(e.get("cost_usd") or 0.0) for e in last_audit),
                }
            )
            # Step 2: re-run with current settings as variant B.
            with st.spinner("Re-running with current settings…"):
                final = _ab_run(last_q)
            new_audit = final.get("audit_log") or []
            history.append(
                {
                    "label": f"B · {_short_settings_label()}",
                    "query": last_q,
                    "answer": final.get("final_answer"),
                    "audit": new_audit,
                    "cost": sum(float(e.get("cost_usd") or 0.0) for e in new_audit),
                }
            )
            # Keep only the last two for the side-by-side widget.
            st.session_state["ab_history"] = history[-2:]
            st.rerun()

    if len(history) < 2:
        st.caption("No A/B yet — click **Snapshot + re-run** above.")
        return

    a, b = history[-2], history[-1]
    st.markdown("---")
    cols = st.columns(2)
    for col, variant in zip(cols, [a, b], strict=False):
        with col:
            st.markdown(f"#### {variant['label']}")
            st.metric("total cost", fmt_money(variant.get("cost")))
            st.markdown("**Answer:**")
            st.markdown(f"> {variant.get('answer') or '_(no answer)_'}")
            with st.expander("Audit log", expanded=False):
                for e in variant.get("audit") or []:
                    render_audit_entry(e)


def _short_settings_label() -> str:
    s = runtime_settings_summary()
    return (
        f"rewrite={'on' if s['rewrite_enabled'] else 'off'}, "
        f"rerank={'on' if s['rerank_enabled'] else 'off'}, "
        f"backend={s['research_backend']}, "
        f"ceiling=${s['cost_ceiling']}"
    )
