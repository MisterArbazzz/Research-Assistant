"""End-to-end eval orchestrator.

Runs every case in golden_dataset.GOLDEN_CASES through the compiled graph,
captures the trajectory + outputs, scores them with the four RAGAS-style
metrics + the trajectory judge, and prints a per-case + aggregate table.

Usage:
    uv run python -m tests.eval.run_eval

Forces RESEARCH_BACKEND=mock (deterministic + free) before importing config
so the eval is reproducible. Live LLM calls still happen for clarity /
research / synthesis / validator + the four judges, but no external search.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Hermetic + deterministic — must be set BEFORE src.config is imported.
os.environ["RESEARCH_BACKEND"] = "mock"
# Prevent Tavily live calls even if a key happens to be in env.
os.environ["TAVILY_API_KEY"] = ""
# Disable rerank for the eval — keeps the pipeline simple and fast since
# mock data already returns canonical-relevance hits.
os.environ["RERANK_ENABLED"] = "false"

# Windows console UTF-8 (model output uses em dashes, smart quotes, arrows)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.adapters.tavily import TavilyHit, tavily_search
from src.config import get_settings
from src.graph.builder import build_graph
from src.logging_config import configure_logging
from tests.eval.golden_dataset import GOLDEN_CASES, GoldenCase
from tests.eval.metrics import (
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
)
from tests.eval.trajectory_eval import evaluate_trajectory

# Force a clean settings cache after env mutation above.
get_settings.cache_clear()


def _new_turn_payload(user_query: str) -> dict[str, Any]:
    return {
        "run_id": str(uuid.uuid4()),
        "user_query": user_query,
        "messages": [HumanMessage(content=user_query)],
        "clarity_status": None,
        "clarification_request": None,
        "clarification_response": None,
        "research_findings": None,
        "confidence_score": None,
        "research_attempts": 0,
        "validation_result": None,
        "validation_notes": [],
        "final_answer": None,
    }


async def _drive_turn(
    graph: Any,
    payload: dict[str, Any] | Command,
    config: dict[str, Any],
    clarification_reply: str | None,
) -> dict[str, Any]:
    """Run a single turn, supplying a clarification once if the graph interrupts."""
    current: Any = payload
    for _ in range(3):
        result = await graph.ainvoke(current, config=config)
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if not interrupts:
            return result
        if clarification_reply is None:
            return result  # paused, no reply available — let the case fail
        current = Command(resume=clarification_reply)
        clarification_reply = None
    return result


async def run_case(graph: Any, case: GoldenCase) -> dict[str, Any]:
    """Run all turns of a case, capturing the final state."""
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    user_queries: list[str] = []
    interrupted = False
    final: dict[str, Any] | None = None

    for turn in case["turns"]:
        user_query = turn["user_query"]
        user_queries.append(user_query)
        clarification = turn.get("clarification_response")
        payload = _new_turn_payload(user_query)
        final = await _drive_turn(graph, payload, config, clarification)
        if final.get("__interrupt__"):
            interrupted = True
            break

    return {
        "case_id": case["id"],
        "user_queries": user_queries,
        "interrupted": interrupted,
        "final_state": final or {},
    }


async def score_case(
    case: GoldenCase, run_outcome: dict[str, Any]
) -> dict[str, Any]:
    """Compute all metrics for one case."""
    final_state: dict[str, Any] = run_outcome["final_state"] or {}
    research_findings = final_state.get("research_findings")
    final_answer = final_state.get("final_answer")
    audit_log = final_state.get("audit_log") or []

    # Re-run mock retrieval to get the hits the agent saw — we use this
    # for context_recall / context_precision. (The agent's hits aren't
    # carried in state explicitly, so we reconstruct them from the same
    # query the rewriter would have used. Mock retrieval is deterministic,
    # so this exactly reproduces what was fed to the LLM.)
    last_query = run_outcome["user_queries"][-1]
    company_hint = case.get("expected_company") or ""
    hits: list[TavilyHit] = await tavily_search(
        f"{company_hint} {last_query}", k=get_settings().TAVILY_MAX_RESULTS
    )

    expected_company = case.get("expected_company")
    recall = context_recall(hits, expected_company)
    precision = context_precision(hits, expected_company)

    faith_score, faith_rationale = await faithfulness(
        last_query, research_findings, final_answer
    )
    # If the last turn had an in-turn clarification, embed that in the
    # query the judge sees — otherwise it can't resolve pronouns ('their',
    # 'it') against context that only lived in the interrupt reply.
    prior_queries = run_outcome["user_queries"][:-1]
    last_turn = case.get("turns") or []
    last_clarification = last_turn[-1].get("clarification_response") if last_turn else None
    judge_query = (
        f"{last_query} (then clarified to refer to: '{last_clarification}')"
        if last_clarification
        else last_query
    )
    rel_score, rel_rationale = await answer_relevance(
        judge_query, final_answer, prior_queries=prior_queries or None
    )

    trajectory = await evaluate_trajectory(
        run_outcome["user_queries"],
        audit_log,
        research_findings,
        final_answer,
    )

    # Pass-criteria checks against case expectations.
    expected_company_ok = (
        expected_company is None
        or final_state.get("company") == expected_company
        or expected_company.split()[0].lower()
        in (final_state.get("company") or "").lower()
    )
    confidence_ok = (
        final_state.get("confidence_score") or 0.0
    ) >= case.get("min_confidence", 0.0)
    interrupt_ok = bool(case.get("expects_interrupt")) == any(
        e.get("node") == "interrupt_node" for e in audit_log
    )

    overall_pass = bool(
        expected_company_ok
        and confidence_ok
        and interrupt_ok
        and faith_score >= 0.6
        and rel_score >= 0.6
        and trajectory.overall_score >= 6.0
    )

    return {
        "case_id": case["id"],
        "context_recall": recall,
        "context_precision": precision,
        "faithfulness": faith_score,
        "faithfulness_rationale": faith_rationale,
        "relevance": rel_score,
        "relevance_rationale": rel_rationale,
        "trajectory_overall": trajectory.overall_score,
        "trajectory_efficiency": trajectory.efficiency_score,
        "trajectory_grounding": trajectory.grounding_score,
        "clarity_correct": trajectory.clarity_correct,
        "retry_helpful": trajectory.retry_helpful,
        "trajectory_rationale": trajectory.rationale,
        "expected_company_ok": expected_company_ok,
        "confidence_ok": confidence_ok,
        "interrupt_ok": interrupt_ok,
        "pass": overall_pass,
        "actual_company": final_state.get("company"),
        "actual_confidence": final_state.get("confidence_score"),
        "actual_attempts": final_state.get("research_attempts"),
        "answer_preview": (final_answer or "")[:140],
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "case_id",
        "recall",
        "prec",
        "faith",
        "rel",
        "traj",
        "eff",
        "ground",
        "cm",
        "cf",
        "int",
        "pass",
    ]
    widths = [22, 6, 6, 6, 5, 6, 5, 6, 4, 4, 4, 5]

    def fmt_row(values: list[str]) -> str:
        return " | ".join(v.ljust(w) for v, w in zip(values, widths, strict=False))

    print()
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        line = [
            r["case_id"],
            f"{r['context_recall']:.2f}",
            f"{r['context_precision']:.2f}",
            f"{r['faithfulness']:.2f}",
            f"{r['relevance']:.2f}",
            f"{r['trajectory_overall']:.1f}",
            f"{r['trajectory_efficiency']:.1f}",
            f"{r['trajectory_grounding']:.1f}",
            "Y" if r["expected_company_ok"] else "N",
            "Y" if r["confidence_ok"] else "N",
            "Y" if r["interrupt_ok"] else "N",
            "PASS" if r["pass"] else "FAIL",
        ]
        print(fmt_row(line))
    print()
    print(
        "Legend: recall/prec=context. faith=faithfulness, rel=answer_relevance, "
        "traj=trajectory_overall, eff=efficiency, ground=grounding."
    )
    print(
        "        cm=expected_company_ok, cf=confidence_ok, int=interrupt_ok."
    )


def _print_aggregate(rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    if n == 0:
        return

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / n

    pass_count = sum(1 for r in rows if r["pass"])
    print()
    print("=" * 60)
    print(f"Aggregate over {n} cases:")
    print(f"  pass rate            : {pass_count}/{n} ({pass_count/n:.0%})")
    print(f"  context_recall       : {mean('context_recall'):.3f}")
    print(f"  context_precision    : {mean('context_precision'):.3f}")
    print(f"  faithfulness         : {mean('faithfulness'):.3f}")
    print(f"  answer_relevance     : {mean('relevance'):.3f}")
    print(f"  trajectory_overall   : {mean('trajectory_overall'):.3f}/10")
    print(f"  trajectory_efficiency: {mean('trajectory_efficiency'):.3f}/10")
    print(f"  trajectory_grounding : {mean('trajectory_grounding'):.3f}/10")
    print("=" * 60)


async def main() -> int:
    configure_logging()
    settings = get_settings()
    print(f"backend={settings.RESEARCH_BACKEND} model_primary={settings.MODEL_PRIMARY} "
          f"model_qa={settings.MODEL_QA} rerank_enabled={settings.RERANK_ENABLED}")
    print(f"Running {len(GOLDEN_CASES)} golden cases…")
    print()

    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    graph = await build_graph(MemorySaver())

    for case in GOLDEN_CASES:
        cid = case["id"]
        t0 = time.perf_counter()
        try:
            outcome = await run_case(graph, case)
            scored = await score_case(case, outcome)
        except Exception as exc:
            scored = {
                "case_id": cid,
                "context_recall": 0.0,
                "context_precision": 0.0,
                "faithfulness": 0.0,
                "relevance": 0.0,
                "trajectory_overall": 0.0,
                "trajectory_efficiency": 0.0,
                "trajectory_grounding": 0.0,
                "clarity_correct": False,
                "retry_helpful": False,
                "trajectory_rationale": f"crashed: {exc}",
                "expected_company_ok": False,
                "confidence_ok": False,
                "interrupt_ok": False,
                "pass": False,
                "actual_company": None,
                "actual_confidence": None,
                "actual_attempts": None,
                "answer_preview": f"<exception: {exc}>",
                "faithfulness_rationale": "",
                "relevance_rationale": "",
            }
        elapsed = time.perf_counter() - t0
        rows.append(scored)
        marker = "PASS" if scored["pass"] else "FAIL"
        print(f"  [{elapsed:5.1f}s] {cid:24s} {marker}  "
              f"company={scored['actual_company']!r} "
              f"conf={scored['actual_confidence']} "
              f"attempts={scored['actual_attempts']}")

    elapsed_total = time.perf_counter() - t_start
    print()
    print(f"Total elapsed: {elapsed_total:.1f}s")

    _print_table(rows)
    _print_aggregate(rows)

    # Persist a JSON snapshot for diffing across runs.
    out_path = Path(__file__).resolve().parent / "last_run.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print()
    print(f"[wrote {out_path}]")

    fails = sum(1 for r in rows if not r["pass"])
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
