"""Pipeline Trace tab — last run's per-node breakdown."""

from __future__ import annotations

import json

import streamlit as st

from streamlit_app.components import (
    cost_breakdown_chart,
    fmt_money,
    fmt_ms,
    render_audit_entry,
    render_audit_table,
)


def render() -> None:
    st.subheader("Pipeline Trace — last run breakdown")
    st.caption(
        "Audit log captured by the graph, in execution order. Per-node "
        "cost, latency, tokens, plus key state deltas. Use the Settings → "
        "A/B widget to compare two runs side by side."
    )

    audit = st.session_state.get("last_run_audit") or []
    findings = st.session_state.get("last_run_findings")
    last_q = st.session_state.get("last_user_query")
    last_a = st.session_state.get("last_final_answer")

    if not audit:
        st.info(
            "👈 Run a query in the Chat tab first, then come back here for the "
            "trace."
        )
        return

    if last_q:
        st.markdown(f"**Last query:** _{last_q}_")

    # Top-line metrics.
    total_cost = sum(float(e.get("cost_usd") or 0.0) for e in audit)
    total_latency = sum(int(e.get("latency_ms") or 0) for e in audit)
    nodes_fired = len(audit)
    cols = st.columns(4)
    cols[0].metric("nodes fired", nodes_fired)
    cols[1].metric("total latency", fmt_ms(total_latency))
    cols[2].metric("total cost", fmt_money(total_cost))
    cols[3].metric("turn", st.session_state["turn_count"])

    st.divider()

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Per-node breakdown")
        for entry in audit:
            render_audit_entry(entry)
            st.divider()
    with right:
        st.markdown("### Cost per node")
        cost_breakdown_chart(audit)

        if findings:
            st.markdown("### Research findings")
            with st.expander("research_findings (raw)", expanded=False):
                st.json(findings)

        if last_a:
            st.markdown("### Final answer")
            st.markdown(f"> {last_a}")

    st.divider()
    st.markdown("### Compact table")
    render_audit_table(audit)

    with st.expander("Raw audit_log JSON", expanded=False):
        st.code(json.dumps(audit, indent=2, default=str), language="json")
