"""Shared helpers for node audit-trail writes.

Every node opens a span, calls the LLM (or external service), and writes an
AgentStep row. This module hides the Neo4j-may-be-absent branch so each node
file stays readable.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ...neo4j_client.audit import record_step
from ...neo4j_client.driver import Neo4jClient

logger = logging.getLogger(__name__)


def get_neo4j_client(config: RunnableConfig) -> Neo4jClient | None:
    """Pull the Neo4j client off the LangGraph RunnableConfig if present.

    The CLI runner (and tests) may run without Neo4j up. Nodes call
    `safe_record_step` which no-ops when this returns None.
    """
    configurable = config.get("configurable") or {}
    return configurable.get("neo4j_client")  # type: ignore[no-any-return]


async def safe_record_step(
    config: RunnableConfig,
    run_id: str,
    node: str,
    latency_ms: int,
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write an AgentStep audit row; no-op if Neo4j isn't wired.

    Failures inside audit are logged but don't bubble up — a missing audit
    row is a degraded observability outcome, not a fatal run error.
    """
    client = get_neo4j_client(config)
    if client is None:
        return
    try:
        await record_step(
            client,
            run_id,
            node,
            latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001 — audit failure must not break the run
        logger.warning(
            "audit step write failed",
            extra={"node": node, "run_id": run_id, "error": str(exc)},
        )
