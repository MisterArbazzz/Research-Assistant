"""Interactive CLI runner for the Research Assistant.

One thread_id per session — every turn is a separate `ainvoke` against the
same compiled graph, with the AsyncSqliteSaver checkpointer persisting state
between turns. That's what makes follow-ups like "what about their CEO?"
work after an earlier "tell me about Tesla" — the messages channel survives.

The interrupt loop is handled here: when the graph pauses (via the
`interrupt()` call inside interrupt_node), `ainvoke` returns a state with a
populated `__interrupt__` channel. We surface the request to the user, read
their response, and resume with `Command(resume=<reply>)`.

Usage:
    uv run python scripts/cli.py
    > Tell me about Apple's recent news
    > How is the company doing?  (will trigger interrupt)
    > Tesla
    > What about their CEO?
    > exit
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
import uuid
from pathlib import Path

# Windows consoles default to cp1252; force UTF-8 so model output (arrows,
# em dashes, smart quotes, etc.) doesn't crash the REPL on print().
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Make `src` importable when running this file directly via `python scripts/cli.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from src.config import get_settings
from src.graph.builder import build_graph
from src.logging_config import configure_logging
from src.observability import configure_langsmith, configure_tracer

logger = logging.getLogger(__name__)

CHECKPOINT_DB = Path("./data/checkpoints.db")
CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)


def _print_divider(label: str = "") -> None:
    print()
    print(f"────── {label} ".ljust(72, "─"))


async def _run_turn_until_complete_or_interrupt(
    graph: object,
    payload: object,
    config: dict,
) -> dict:
    """Drive `ainvoke` to completion, handling interrupts in a loop.

    Returns the final state dict from the graph. While the graph pauses on
    an interrupt, we read the user's clarification and resume with
    Command(resume=...). The loop bounds at 5 hops to defend against a
    pathological clarity model that keeps asking.
    """
    current_input: object = payload
    for _ in range(5):
        # ainvoke returns a state-shaped dict; if it contains a __interrupt__
        # channel it means the graph paused before END.
        result = await graph.ainvoke(current_input, config=config)  # type: ignore[attr-defined]

        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if not interrupts:
            return result  # type: ignore[return-value]

        # interrupts is a tuple of Interrupt objects; first one carries our payload.
        payload_dict = interrupts[0].value  # type: ignore[attr-defined]
        request = payload_dict.get("request") if isinstance(payload_dict, dict) else str(payload_dict)
        _print_divider("AGENT NEEDS CLARIFICATION")
        print(f"  {request}")
        print()
        clarification = input("  Your reply: ").strip()
        if not clarification:
            clarification = "(no answer)"
        current_input = Command(resume=clarification)

    raise RuntimeError("Graph paused for clarification 5 times in one turn — aborting.")


async def main() -> None:
    configure_logging()
    configure_tracer()       # OTel — console exporter unless OTEL_EXPORTER_OTLP_ENDPOINT is set
    configure_langsmith()    # No-op unless LANGSMITH_API_KEY is set in .env
    settings = get_settings()
    print(f"Research backend: {settings.RESEARCH_BACKEND}")
    print(f"Tavily key set:   {bool(settings.TAVILY_API_KEY)}")
    print(f"Models:           primary={settings.MODEL_PRIMARY}, qa={settings.MODEL_QA}")
    print(f"Max attempts:     {settings.MAX_RESEARCH_ATTEMPTS}")
    print(f"Cost ceiling:     ${settings.COST_CEILING_PER_RUN_USD:.4f}/run")
    print(f"LangSmith:        {'enabled' if settings.LANGSMITH_API_KEY else 'disabled'}")
    print()

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        graph = await build_graph(checkpointer)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        print(f"Session thread_id: {thread_id}")
        print("Type 'exit' or 'quit' to leave. Multi-turn is enabled.")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            run_id = str(uuid.uuid4())
            # Reset per-turn fields. The checkpointer carries state across
            # turns under the same thread_id, but research_attempts /
            # validation_* / confidence_score should restart for each new
            # user query — otherwise the cap fires prematurely on long
            # conversations.
            initial = {
                "run_id": run_id,
                "user_query": user_input,
                "messages": [HumanMessage(content=user_input)],
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

            try:
                final_state = await _run_turn_until_complete_or_interrupt(
                    graph, initial, config
                )
            except Exception as exc:
                _print_divider("RUN FAILED")
                print(f"  {exc}")
                continue

            answer = (final_state or {}).get("final_answer") or "(no answer produced)"
            cost = (final_state or {}).get("total_cost_usd", 0.0)
            attempts = (final_state or {}).get("research_attempts", 0)
            _print_divider("ASSISTANT")
            print(answer)
            print()
            print(
                f"  [stats: research_attempts={attempts} | "
                f"total_cost=${cost:.5f} | run_id={run_id}]"
            )


if __name__ == "__main__":
    asyncio.run(main())
