"""Wrap LangGraph's astream loop for live UI updates.

The Streamlit UI passes in a per-node `status_callback` that gets called
every time a node completes; the callback updates the corresponding
`st.status` block in real time. The interrupt path returns the payload
to the caller so the chat tab can prompt for clarification.

`stream_turn` is the entry point for both fresh queries and resumes
(via `Command(resume=...)`). The caller is responsible for sequencing
the user's clarification reply.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

# Order of nodes as they should appear in the UI. Fixed display order, not
# necessarily the runtime order (some nodes may not fire on a given run).
DISPLAY_ORDER = [
    "clarity_agent",
    "interrupt_node",
    "research_agent",
    "validator_agent",
    "synthesis_agent",
    "memory_writer",
]


def fresh_payload(user_query: str) -> dict[str, Any]:
    """Build the per-turn input dict.

    Per-turn fields (attempts, validation, answer) are reset so the cap
    and routing don't fire prematurely on long conversations. Same shape
    as the CLI runner uses.
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


async def stream_turn(
    graph: Any,
    payload: dict[str, Any] | Command,
    config: dict[str, Any],
    on_node_update: Callable[[str, dict[str, Any]], None],
) -> dict[str, Any]:
    """Run one turn through the graph, calling `on_node_update` per node.

    Returns the final state dict (post-run, after END is reached). If the
    graph paused on an interrupt, the returned dict contains the special
    `__interrupt__` key — the caller surfaces the request to the user
    and re-invokes us with `Command(resume=<reply>)`.

    `on_node_update(node_name, state_delta)` fires once per node firing.
    The delta is whatever that node returned from its function, which is
    what gets merged into the state by LangGraph's reducer logic.
    """
    final_state: dict[str, Any] = {}
    interrupted_payload: dict[str, Any] | None = None

    async for chunk in graph.astream(
        payload,
        config=config,
        stream_mode="updates",
    ):
        # `chunk` is normally `{node_name: {field: value, ...}}`.
        # On interrupt LangGraph emits `{"__interrupt__": (Interrupt(...),)}`.
        if not isinstance(chunk, dict):
            continue
        for node, delta in chunk.items():
            if node == "__interrupt__":
                interrupted_payload = _extract_interrupt(delta)
                continue
            try:
                on_node_update(node, delta if isinstance(delta, dict) else {})
            except Exception:  # UI errors must not abort the run
                logger.exception("on_node_update callback raised", extra={"node": node})

    # After astream completes (or paused on interrupt), pull the
    # checkpointed state for this thread so we have everything to
    # display, including reducer-merged fields like audit_log.
    snapshot = await graph.aget_state(config)
    final_state = dict(snapshot.values) if snapshot else {}

    if interrupted_payload is not None:
        final_state["__interrupt__"] = interrupted_payload

    return final_state


def _extract_interrupt(delta: Any) -> dict[str, Any]:
    """Pull the request dict out of a LangGraph Interrupt object."""
    # delta can be a tuple of Interrupt objects or a single one
    seq = delta if isinstance(delta, (list, tuple)) else [delta]
    for item in seq:
        value = getattr(item, "value", None)
        if isinstance(value, dict):
            return value
    return {"request": "Could you clarify?", "reason": "needs_clarification"}
