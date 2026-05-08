"""Audit-trail writers — Run, AgentStep, and Mock nodes.

Every run produces a `(:Run)` node connected to a chain of `(:AgentStep)` nodes
via `[:HAS_STEP]`. Mock-mode adapters additionally connect via `[:DELIVERED_VIA]`
to `(:Mock)` nodes — the audit story is identical whether the run delivered to
real Slack or to a mock.

All Cypher is parametrized; never f-string interpolation.
"""

from __future__ import annotations

import json
from typing import Any

from .driver import Neo4jClient

# --- Cypher constants — module-level so tests can match against them. ---

RECORD_RUN_START = """
MERGE (r:Run {id: $run_id})
SET r.started_at = datetime(),
    r.status = 'running',
    r.use_case = $use_case,
    r.input_json = $input_json
RETURN r.id AS id
"""

RECORD_RUN_END = """
MATCH (r:Run {id: $run_id})
SET r.ended_at = datetime(),
    r.status = $status,
    r.total_cost_usd = $total_cost_usd,
    r.duration_ms = duration.between(r.started_at, r.ended_at).milliseconds
RETURN r.id AS id
"""

RECORD_STEP = """
MATCH (r:Run {id: $run_id})
CREATE (s:AgentStep {
    id: randomUUID(),
    node: $node,
    started_at: datetime(),
    latency_ms: $latency_ms,
    prompt_tokens: $prompt_tokens,
    completion_tokens: $completion_tokens,
    cost_usd: $cost_usd,
    metadata_json: $metadata_json
})
CREATE (r)-[:HAS_STEP]->(s)
RETURN s.id AS id
"""

WRITE_MOCK_NODE = """
MATCH (r:Run {id: $run_id})
CREATE (m:Mock {
    id: randomUUID(),
    kind: $kind,
    payload_json: $payload_json,
    created_at: datetime()
})
CREATE (r)-[rel:DELIVERED_VIA {kind: $kind}]->(m)
RETURN m.id AS id
"""


async def record_run_start(
    client: Neo4jClient,
    run_id: str,
    use_case: str,
    input_dict: dict[str, Any],
) -> None:
    async with client.session() as session:
        await session.run(
            RECORD_RUN_START,
            run_id=run_id,
            use_case=use_case,
            input_json=json.dumps(input_dict, default=str),
        )


async def record_run_end(
    client: Neo4jClient,
    run_id: str,
    status: str,
    total_cost_usd: float = 0.0,
) -> None:
    async with client.session() as session:
        await session.run(
            RECORD_RUN_END,
            run_id=run_id,
            status=status,
            total_cost_usd=total_cost_usd,
        )


async def record_step(
    client: Neo4jClient,
    run_id: str,
    node: str,
    latency_ms: int,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    async with client.session() as session:
        await session.run(
            RECORD_STEP,
            run_id=run_id,
            node=node,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            metadata_json=json.dumps(metadata or {}, default=str),
        )


async def write_mock_node(
    client: Neo4jClient,
    run_id: str,
    kind: str,
    payload: dict[str, Any],
) -> str:
    """Write a Mock node tied to a Run. Used by port-adapters in mock mode.
    Returns the Mock node id (used as a synthetic external id, e.g. mock-slack-ts).
    """
    async with client.session() as session:
        result = await session.run(
            WRITE_MOCK_NODE,
            run_id=run_id,
            kind=kind,
            payload_json=json.dumps(payload, default=str),
        )
        record = await result.single()
        return str(record["id"]) if record else ""
