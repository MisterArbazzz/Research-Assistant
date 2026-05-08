"""Common mock-mode behaviour for port-adapters.

Both:
1. Write a Mock node to Neo4j (durable audit — survives the run, queryable later)
2. Emit an SSE event (live UI render of the side-effect as it happens)

This is the seam that makes mock mode indistinguishable from live for the
audit story and the user-facing demo. Switching to live just bypasses the
mock branch in the adapter's `if settings.<X>_MODE == "live":` check.
"""

from __future__ import annotations

from typing import Any

from ..neo4j_client.audit import write_mock_node
from ..neo4j_client.driver import Neo4jClient
from ..sse_bus import sse_bus


async def emit_mock(
    neo4j_client: Neo4jClient,
    run_id: str,
    adapter: str,
    payload: dict[str, Any],
    sse_event: str | None = None,
) -> str:
    """Emit a mock side-effect into both Neo4j and the SSE stream.

    Args:
        neo4j_client: live Neo4j driver from app.state.neo4j
        run_id: the active run's uuid
        adapter: short name, e.g. "slack" or "clickup" — becomes the Mock.kind
        payload: arbitrary JSON-serializable dict (stored as JSON string in Neo4j)
        sse_event: optional event_type override; defaults to f"{adapter}.delivered"

    Returns:
        The Mock node id (use as a synthetic external id like mock-slack-<ts>).
    """
    mock_id = await write_mock_node(
        client=neo4j_client,
        run_id=run_id,
        kind=adapter,
        payload=payload,
    )
    event_name = sse_event or f"{adapter}.delivered"
    await sse_bus.publish(run_id, event_name, {**payload, "mock_id": mock_id})
    return mock_id
