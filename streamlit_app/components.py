"""Reusable UI bits shared across tabs."""

from __future__ import annotations

from typing import Any

import streamlit as st

# Friendly labels + descriptions per node, used in status blocks.
NODE_META: dict[str, dict[str, str]] = {
    "clarity_agent": {
        "label": "Clarity Agent",
        "icon": "🧭",
        "summary": "Decide if the query is specific enough to research",
    },
    "interrupt_node": {
        "label": "Interrupt — Human in the loop",
        "icon": "✋",
        "summary": "Pause and ask the user to clarify",
    },
    "research_agent": {
        "label": "Research Agent",
        "icon": "🔍",
        "summary": "Rewrite query → Tavily → rerank → digest",
    },
    "validator_agent": {
        "label": "Validator Agent",
        "icon": "🧪",
        "summary": "Judge research quality (may trigger retry)",
    },
    "synthesis_agent": {
        "label": "Synthesis Agent",
        "icon": "✍️",
        "summary": "Compose the user-facing answer",
    },
    "memory_writer": {
        "label": "Memory Writer",
        "icon": "💾",
        "summary": "Selectively save durable user facts",
    },
}


def fmt_money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:.5f}"


def fmt_ms(value: int | float | None) -> str:
    if value is None:
        return "—"
    return f"{int(value)} ms"


def fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def render_audit_entry(entry: dict[str, Any]) -> None:
    """Render a single audit_log entry as a compact metric row + details."""
    node = entry.get("node", "?")
    meta = NODE_META.get(node, {"label": node, "icon": "•", "summary": ""})
    cols = st.columns([2.5, 1, 1, 1, 1])
    cols[0].markdown(f"**{meta['icon']} {meta['label']}**")
    cols[1].metric("latency", fmt_ms(entry.get("latency_ms")))
    cols[2].metric("cost", fmt_money(entry.get("cost_usd")))
    in_t = entry.get("input_tokens") or entry.get("prompt_tokens")
    out_t = entry.get("output_tokens") or entry.get("completion_tokens")
    cols[3].metric("in tok", fmt_int(in_t))
    cols[4].metric("out tok", fmt_int(out_t))

    extras = {
        k: v
        for k, v in entry.items()
        if k
        not in {
            "node",
            "latency_ms",
            "cost_usd",
            "input_tokens",
            "output_tokens",
            "prompt_tokens",
            "completion_tokens",
        }
        and v not in (None, [], {}, "")
    }
    if extras:
        with st.expander("details", expanded=False):
            st.json(extras)


def render_audit_table(audit_log: list[dict[str, Any]]) -> None:
    """Render a compact dataframe summary of all audit entries."""
    if not audit_log:
        st.info("No audit entries yet — submit a query.")
        return
    rows: list[dict[str, Any]] = []
    for entry in audit_log:
        rows.append(
            {
                "node": entry.get("node", "?"),
                "latency_ms": entry.get("latency_ms"),
                "cost_usd": entry.get("cost_usd"),
                "in_tok": entry.get("input_tokens") or entry.get("prompt_tokens"),
                "out_tok": entry.get("output_tokens") or entry.get("completion_tokens"),
                "key_metric": _key_metric_for(entry),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _key_metric_for(entry: dict[str, Any]) -> str:
    node = entry.get("node", "")
    if node == "clarity_agent":
        cs = entry.get("clarity_status")
        comp = entry.get("company") or "—"
        return f"{cs} · {comp}" if cs else "—"
    if node == "research_agent":
        return f"conf={entry.get('confidence_score', '—')} · attempt={entry.get('attempt', '—')}"
    if node == "validator_agent":
        return str(entry.get("validation_result") or "—")
    if node == "synthesis_agent":
        chars = entry.get("answer_chars")
        return f"{chars} chars" if chars else "—"
    if node == "memory_writer":
        return f"facts_written={entry.get('facts_written', '—')}"
    if node == "interrupt_node":
        return "human reply received"
    return "—"


def cost_breakdown_chart(audit_log: list[dict[str, Any]]) -> None:
    """Bar chart of cost per node."""
    if not audit_log:
        return
    data = [
        {"node": e.get("node", "?"), "cost_usd": float(e.get("cost_usd") or 0.0)}
        for e in audit_log
        if e.get("cost_usd")
    ]
    if not data:
        st.caption("No cost recorded yet.")
        return
    st.bar_chart(data, x="node", y="cost_usd", use_container_width=True)


def runtime_settings_pills(summary: dict[str, Any]) -> None:
    """Compact horizontal display of active settings."""
    pills = [
        f"primary={summary['model_primary']}",
        f"qa={summary['model_qa']}",
        f"backend={summary['research_backend']}",
        f"rewrite={'on' if summary['rewrite_enabled'] else 'off'}",
        f"rerank={'on' if summary['rerank_enabled'] else 'off'}",
        f"memory={'on' if summary['memory_enabled'] else 'off'}",
        f"max_attempts={summary['max_attempts']}",
        f"ceiling=${summary['cost_ceiling']}",
    ]
    st.caption(" · ".join(pills))
