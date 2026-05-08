"""Eval Console — Tier 3 single-case live + last-full-run viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from streamlit_app.components import fmt_money
from streamlit_app.state import get_graph_and_saver, run_async
from tests.eval.golden_dataset import GOLDEN_CASES
from tests.eval.run_eval import run_case, score_case

LAST_RUN_JSON = Path(__file__).resolve().parents[2] / "tests" / "eval" / "last_run.json"


def _summary_metrics(scored: dict[str, Any]) -> None:
    cols = st.columns(6)
    cols[0].metric("recall", f"{scored.get('context_recall', 0):.2f}")
    cols[1].metric("precision", f"{scored.get('context_precision', 0):.2f}")
    cols[2].metric("faithfulness", f"{scored.get('faithfulness', 0):.2f}")
    cols[3].metric("relevance", f"{scored.get('relevance', 0):.2f}")
    cols[4].metric("trajectory", f"{scored.get('trajectory_overall', 0):.1f}/10")
    pass_str = "✅ PASS" if scored.get("pass") else "❌ FAIL"
    cols[5].metric("verdict", pass_str)


def _render_live() -> None:
    st.markdown("### Live single-case run")
    st.caption(
        "Picks one golden case, runs it through the graph (mock backend so "
        "it's deterministic + free), then scores it across the 4 RAGAS "
        "metrics + the trajectory judge."
    )

    case_ids = [c["id"] for c in GOLDEN_CASES]
    case_id = st.selectbox("Golden case", options=case_ids)
    case = next(c for c in GOLDEN_CASES if c["id"] == case_id)

    with st.expander("Case definition", expanded=False):
        st.json(case)

    if not st.button("🧪 Run case", type="primary"):
        return

    # The eval harness assumes mock backend for determinism. Don't permanently
    # change settings; the harness already sets these env vars in `run_eval.py`
    # at module load. Calling run_case here uses the cached graph, which was
    # built when the app first loaded — RESEARCH_BACKEND in the live config
    # determines retrieval. So we override here for the duration of the call.
    import os

    original = {
        k: os.environ.get(k)
        for k in ("RESEARCH_BACKEND", "TAVILY_API_KEY", "RERANK_ENABLED")
    }
    os.environ["RESEARCH_BACKEND"] = "mock"
    os.environ["RERANK_ENABLED"] = "false"
    from src.config import get_settings

    get_settings.cache_clear()

    try:
        graph, _saver = get_graph_and_saver()
        with st.spinner("Running the graph (clarity → research → … → synthesis)…"):
            outcome = run_async(run_case(graph, case))
        with st.spinner("Scoring (3 LLM-as-judge calls)…"):
            scored = run_async(score_case(case, outcome))
    finally:
        # Restore original environment.
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_settings.cache_clear()

    st.divider()
    _summary_metrics(scored)

    st.divider()
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Faithfulness rationale**")
        st.markdown(scored.get("faithfulness_rationale", "—") or "_(none)_")
        st.markdown("**Relevance rationale**")
        st.markdown(scored.get("relevance_rationale", "—") or "_(none)_")
    with cols[1]:
        st.markdown("**Trajectory judgment**")
        st.markdown(scored.get("trajectory_rationale", "—") or "_(none)_")
        st.caption(
            f"clarity_correct={scored.get('clarity_correct')} · "
            f"retry_helpful={scored.get('retry_helpful')} · "
            f"efficiency={scored.get('trajectory_efficiency'):.1f} · "
            f"grounding={scored.get('trajectory_grounding'):.1f}"
        )

    st.divider()
    st.markdown("**Final answer (preview)**")
    answer_full = (outcome.get("final_state") or {}).get("final_answer") or ""
    st.markdown(f"> {answer_full[:600]}{'…' if len(answer_full) > 600 else ''}")

    with st.expander("Full scored dict (raw)", expanded=False):
        st.json(scored)


def _render_last_full() -> None:
    st.markdown("### Last full eval run (from `tests/eval/last_run.json`)")
    st.caption(
        "Aggregate from the most recent `uv run python -m tests.eval.run_eval`. "
        "Persisted on disk so reviewers can compare runs across changes."
    )

    if not LAST_RUN_JSON.exists():
        st.info(
            f"No `last_run.json` found at `{LAST_RUN_JSON}`. Run "
            "`uv run python -m tests.eval.run_eval` from the project root."
        )
        return

    rows: list[dict[str, Any]] = json.loads(LAST_RUN_JSON.read_text(encoding="utf-8"))

    if not rows:
        st.info("`last_run.json` is empty.")
        return

    n = len(rows)
    pass_count = sum(1 for r in rows if r.get("pass"))
    cols = st.columns(5)
    cols[0].metric("pass rate", f"{pass_count}/{n}", f"{pass_count / n:.0%}")

    def mean(key: str) -> float:
        return sum(float(r.get(key) or 0.0) for r in rows) / n

    cols[1].metric("recall", f"{mean('context_recall'):.3f}")
    cols[2].metric("faithfulness", f"{mean('faithfulness'):.3f}")
    cols[3].metric("relevance", f"{mean('relevance'):.3f}")
    cols[4].metric("trajectory", f"{mean('trajectory_overall'):.2f}/10")

    st.divider()
    table_rows = [
        {
            "case_id": r["case_id"],
            "recall": f"{r.get('context_recall', 0):.2f}",
            "precision": f"{r.get('context_precision', 0):.2f}",
            "faith": f"{r.get('faithfulness', 0):.2f}",
            "relevance": f"{r.get('relevance', 0):.2f}",
            "trajectory": f"{r.get('trajectory_overall', 0):.1f}",
            "pass": "✅" if r.get("pass") else "❌",
        }
        for r in rows
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Per-case rationales")
    for r in rows:
        marker = "✅ PASS" if r.get("pass") else "❌ FAIL"
        with st.expander(f"{marker} — {r['case_id']}", expanded=not r.get("pass")):
            cols2 = st.columns(2)
            cols2[0].markdown("**Trajectory rationale:**")
            cols2[0].markdown(r.get("trajectory_rationale") or "_(none)_")
            cols2[1].markdown("**Faithfulness:**")
            cols2[1].markdown(r.get("faithfulness_rationale") or "_(none)_")
            cols2[1].markdown("**Relevance:**")
            cols2[1].markdown(r.get("relevance_rationale") or "_(none)_")
            st.markdown(f"**Answer preview:** _{r.get('answer_preview') or '—'}_")


def render() -> None:
    st.subheader("Eval Console — Tier 3")
    st.caption(
        f"{len(GOLDEN_CASES)} golden cases. Live mode runs one case through "
        "the real graph + judges (~30-45s). View mode loads the persisted "
        "JSON from the last full suite run."
    )
    sub = st.tabs(["⚡ Live single case", "📁 Last full run"])
    with sub[0]:
        _render_live()
    with sub[1]:
        _render_last_full()


def fmt_money_compat(value: float | None) -> str:
    return fmt_money(value)
