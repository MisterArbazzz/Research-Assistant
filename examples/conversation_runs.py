"""Scripted demo runs producing the 2 required example conversations.

Run 1 — Clear query, possibly with validator retry loop fired
    Input: "Tell me about Apple's recent news"
    Expectation: clarity → research → (validator if confidence < 6) → synthesis

Run 2 — Ambiguous query → interrupt → resume → multi-turn follow-up
    Turn 1: "How is the company doing?"
    Clarification reply: "Tesla"
    Turn 2 (same thread): "What about their CEO?"
    Expectation: clarity_needs → interrupt → resume → research → synthesis;
                 then a follow-up turn that uses the conversation memory.

Outputs are written to examples/run_1.txt and examples/run_2.txt. The README
embeds these transcripts so reviewers can see the trajectory without running
the system.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import uuid
from pathlib import Path

# Windows consoles default to cp1252; force UTF-8 so Unicode in answers
# (arrows, em dashes, smart quotes) doesn't crash the demo.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from src.config import get_settings
from src.graph.builder import build_graph
from src.logging_config import configure_logging
from src.observability import configure_langsmith, configure_tracer

EXAMPLES_DIR = Path(__file__).resolve().parent
CHECKPOINT_DB = Path("./data/example_checkpoints.db")
CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)


class TranscriptWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[str] = []

    def write(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def flush(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _new_turn_payload(user_query: str) -> dict:
    """Build the initial input dict for a fresh turn.

    The checkpointer carries state across turns under the same thread_id,
    but per-turn fields (attempts, validation outcome, etc.) need to reset
    or the cap will fire prematurely on long conversations.
    """
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


def _format_audit_log(audit_log: list[dict]) -> str:
    lines: list[str] = []
    for entry in audit_log:
        node = entry.get("node", "?")
        bits: list[str] = []
        for key in (
            "attempt",
            "tavily_hits",
            "clarity_status",
            "company",
            "confidence_score",
            "validation_result",
            "latency_ms",
            "cost_usd",
        ):
            if key in entry and entry[key] is not None:
                val = entry[key]
                if isinstance(val, float):
                    bits.append(f"{key}={val:.3f}")
                else:
                    bits.append(f"{key}={val}")
        if entry.get("notes"):
            bits.append(f"notes={entry['notes']}")
        lines.append(f"  • {node}: " + ", ".join(bits))
        if "rewritten_query" in entry:
            lines.append(f"      rewritten_query: {entry['rewritten_query']!r}")
        if "rewrite_rationale" in entry and entry.get("rewrite_rationale"):
            lines.append(f"      rewrite_rationale: {entry['rewrite_rationale']}")
    return "\n".join(lines)


async def _drive_turn(
    graph: object,
    payload: object,
    config: dict,
    clarification_reply: str | None,
    tw: TranscriptWriter,
) -> dict:
    """Run a turn, handling at most one interrupt with the supplied reply."""
    current: object = payload
    for hop in range(3):
        result = await graph.ainvoke(current, config=config)  # type: ignore[attr-defined]
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if not interrupts:
            return result  # type: ignore[return-value]

        payload_dict = interrupts[0].value
        tw.write("")
        tw.write("[GRAPH PAUSED — interrupt fired]")
        tw.write(f"  Reason:  {payload_dict.get('reason')}")
        tw.write(f"  Request: {payload_dict.get('request')}")
        if clarification_reply is None:
            tw.write("  (no clarification provided — aborting turn)")
            return result  # type: ignore[return-value]
        tw.write(f"  Resuming with: {clarification_reply!r}")
        current = Command(resume=clarification_reply)
        # Only one interrupt expected per turn in the demo; subsequent hops
        # would re-trigger the loop if clarity stays unhappy.
        clarification_reply = None
    return result  # type: ignore[return-value]


async def run_one(graph: object, tw: TranscriptWriter) -> None:
    tw.write("=" * 72)
    tw.write("RUN 1 — Clear query (validator-retry-loop scenario)")
    tw.write("=" * 72)
    tw.write("")
    thread_id = f"demo-1-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    tw.write(f"thread_id: {thread_id}")
    tw.write("")

    user_query = "Tell me about Apple's recent news and how the stock is doing"
    tw.write(f"You: {user_query}")
    payload = _new_turn_payload(user_query)
    final = await _drive_turn(graph, payload, config, None, tw)
    tw.write("")
    tw.write("--- AUDIT TRAIL ---")
    tw.write(_format_audit_log(final.get("audit_log") or []))
    tw.write("")
    tw.write("--- RESEARCH FINDINGS ---")
    tw.write(json.dumps(final.get("research_findings"), indent=2))
    tw.write("")
    tw.write("--- ASSISTANT'S FINAL ANSWER ---")
    tw.write(final.get("final_answer") or "(no answer)")
    tw.write("")
    tw.write(
        f"Stats: research_attempts={final.get('research_attempts')}, "
        f"validation_result={final.get('validation_result')}, "
        f"confidence_score={final.get('confidence_score')}, "
        f"total_cost_usd={final.get('total_cost_usd', 0):.5f}"
    )
    tw.write("")


async def run_two(graph: object, tw: TranscriptWriter) -> None:
    tw.write("=" * 72)
    tw.write("RUN 2 — Ambiguous → interrupt → resume → multi-turn follow-up")
    tw.write("=" * 72)
    tw.write("")
    thread_id = f"demo-2-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    tw.write(f"thread_id: {thread_id} (same thread used for both turns)")
    tw.write("")

    # Turn 1 — ambiguous; expect clarity → interrupt → resume → research → synthesis.
    turn1_query = "How is the company doing?"
    tw.write(f"You (turn 1): {turn1_query}")
    payload1 = _new_turn_payload(turn1_query)
    final1 = await _drive_turn(graph, payload1, config, "Tesla", tw)
    tw.write("")
    tw.write("--- TURN 1 AUDIT TRAIL ---")
    tw.write(_format_audit_log(final1.get("audit_log") or []))
    tw.write("")
    tw.write("--- TURN 1 ASSISTANT ANSWER ---")
    tw.write(final1.get("final_answer") or "(no answer)")
    tw.write("")
    tw.write(
        f"Turn 1 stats: research_attempts={final1.get('research_attempts')}, "
        f"validation_result={final1.get('validation_result')}, "
        f"confidence_score={final1.get('confidence_score')}, "
        f"total_cost_usd={final1.get('total_cost_usd', 0):.5f}"
    )
    tw.write("")

    # Turn 2 — follow-up on the same thread_id; checkpointer carries messages.
    turn2_query = "What about their CEO?"
    tw.write(f"You (turn 2): {turn2_query}")
    payload2 = _new_turn_payload(turn2_query)
    final2 = await _drive_turn(graph, payload2, config, None, tw)
    tw.write("")
    tw.write("--- TURN 2 AUDIT TRAIL ---")
    tw.write(_format_audit_log(final2.get("audit_log") or []))
    tw.write("")
    tw.write("--- TURN 2 ASSISTANT ANSWER ---")
    tw.write(final2.get("final_answer") or "(no answer)")
    tw.write("")
    tw.write(
        f"Turn 2 stats: research_attempts={final2.get('research_attempts')}, "
        f"company_carried={final2.get('company')}, "
        f"messages_in_state={len(final2.get('messages') or [])}"
    )
    tw.write("")


async def main() -> None:
    configure_logging()
    configure_tracer()
    configure_langsmith()
    settings = get_settings()
    print(f"Research backend: {settings.RESEARCH_BACKEND}")
    print(f"Tavily key set:   {bool(settings.TAVILY_API_KEY)}")
    print(f"Max attempts:     {settings.MAX_RESEARCH_ATTEMPTS}")
    print(f"Cost ceiling:     ${settings.COST_CEILING_PER_RUN_USD:.4f}/run")
    print(f"LangSmith:        {'enabled' if settings.LANGSMITH_API_KEY else 'disabled'}")
    print()

    # Wipe the demo checkpoint so reruns don't conflict on existing thread state.
    if CHECKPOINT_DB.exists():
        CHECKPOINT_DB.unlink()

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        graph = await build_graph(checkpointer)

        tw1 = TranscriptWriter(EXAMPLES_DIR / "run_1.txt")
        await run_one(graph, tw1)
        tw1.flush()
        print()
        print(f"[wrote {tw1.path}]")
        print()

        tw2 = TranscriptWriter(EXAMPLES_DIR / "run_2.txt")
        await run_two(graph, tw2)
        tw2.flush()
        print()
        print(f"[wrote {tw2.path}]")


if __name__ == "__main__":
    asyncio.run(main())
