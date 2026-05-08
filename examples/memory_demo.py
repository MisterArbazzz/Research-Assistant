"""Long-term memory cross-thread demo (Tier 4).

Two CONVERSATIONS, same `user_id`, different `thread_id`. Demonstrates that
a preference dropped in conversation A surfaces in the research_agent's
prompt during conversation B — even though LangGraph's checkpointer scopes
short-term state to thread_id only.

What you should see:
  - Session A turn 1: user states a preference ("I prefer technical depth")
    → memory_writer logs `facts_written=1`
  - Session B turn 1 (different thread_id, same user_id): research_agent
    audit shows `memory_facts_retrieved=1` and the answer adapts tone

Forces RESEARCH_BACKEND=mock so the demo is offline-deterministic. Wipes the
long-term DB at start so reruns are reproducible.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import uuid
from pathlib import Path

os.environ["RESEARCH_BACKEND"] = "mock"
os.environ["TAVILY_API_KEY"] = ""
os.environ["RERANK_ENABLED"] = "false"  # rerank not needed for this demo

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.config import get_settings
from src.graph.builder import build_graph
from src.logging_config import configure_logging
from src.memory.longterm import LONGTERM_DB_PATH

get_settings.cache_clear()


def _payload(user_query: str) -> dict:
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


def _summarize_audit(audit_log: list[dict]) -> None:
    for entry in audit_log:
        node = entry.get("node", "?")
        bits = []
        for key in (
            "facts_written",
            "facts",
            "memory_facts_retrieved",
            "user_id",
            "company",
            "confidence_score",
        ):
            if key in entry and entry[key] is not None:
                bits.append(f"{key}={entry[key]}")
        if bits:
            print(f"    • {node}: " + ", ".join(bits))


async def main() -> None:
    configure_logging()
    user_id = "alice"
    print(f"long-term DB: {LONGTERM_DB_PATH}")
    if LONGTERM_DB_PATH.exists():
        LONGTERM_DB_PATH.unlink()
        print("  (wiped for reproducible demo)")
    print()

    graph = await build_graph(MemorySaver())

    # ============= SESSION A — different thread_id =============
    thread_a = f"sessA-{uuid.uuid4().hex[:6]}"
    config_a = {"configurable": {"thread_id": thread_a, "user_id": user_id}}
    print("=" * 70)
    print(f"SESSION A — thread_id={thread_a}, user_id={user_id}")
    print("=" * 70)
    query_a = (
        "Tell me about Tesla. I'm a portfolio manager and I prefer technical "
        "detail over high-level summaries."
    )
    print(f"  You: {query_a}")
    final_a = await graph.ainvoke(_payload(query_a), config=config_a)
    print(f"  Assistant: {(final_a.get('final_answer') or '')[:240]}…")
    print("  Audit:")
    _summarize_audit(final_a.get("audit_log") or [])
    print()

    # ============= SESSION B — DIFFERENT thread_id, SAME user_id =============
    thread_b = f"sessB-{uuid.uuid4().hex[:6]}"
    config_b = {"configurable": {"thread_id": thread_b, "user_id": user_id}}
    print("=" * 70)
    print(f"SESSION B — thread_id={thread_b} (NEW), user_id={user_id} (SAME)")
    print("=" * 70)
    query_b = "Tell me about Apple"
    print(f"  You: {query_b}")
    final_b = await graph.ainvoke(_payload(query_b), config=config_b)
    print(f"  Assistant: {(final_b.get('final_answer') or '')[:400]}…")
    print("  Audit:")
    _summarize_audit(final_b.get("audit_log") or [])
    print()
    print("=" * 70)
    print("If memory worked: research_agent in session B should show")
    print("  memory_facts_retrieved >= 1, and the answer should reflect")
    print("  the technical-detail preference set in session A.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
