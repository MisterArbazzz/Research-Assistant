"""Run lifecycle endpoints — generic skeleton.

Use cases override the inner `event_generator` body to call `graph.astream(...)`
with their use-case-specific initial state. The SSE bus drain (for adapter
side-effects) and the boilerplate's run_start / run_end audit calls are reusable.

A minimal use-case override looks like this in `src/routes/<use_case>_runs.py`:

    @router.post("/runs")
    async def create_run(body: GreenlightInput, ...):
        run_id = str(uuid.uuid4())
        await record_run_start(neo4j, run_id, "greenlight", body.model_dump())
        # stash initial state somewhere keyed by run_id, then return the stream URL
        ...

    @router.get("/runs/{run_id}/events")
    async def run_events(...):
        # drain BOTH sse_bus (adapter events) AND graph.astream (state updates)
        ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..deps import get_graph, get_neo4j
from ..neo4j_client.audit import record_run_end, record_run_start
from ..neo4j_client.driver import Neo4jClient
from ..sse_bus import sse_bus

logger = logging.getLogger(__name__)
router = APIRouter()


class RunRequest(BaseModel):
    """Generic run input. Use cases replace this with a typed schema."""

    use_case: str
    input: dict[str, Any]


class RunResponse(BaseModel):
    run_id: str
    stream_url: str


@router.post("/runs", response_model=RunResponse)
async def create_run(
    body: RunRequest,
    neo4j: Neo4jClient = Depends(get_neo4j),
) -> RunResponse:
    """Generic run starter. Use cases override with typed input schemas."""
    run_id = str(uuid.uuid4())
    await record_run_start(neo4j, run_id, body.use_case, body.input)
    logger.info("run created", extra={"run_id": run_id, "use_case": body.use_case})
    return RunResponse(run_id=run_id, stream_url=f"/api/v1/runs/{run_id}/events")


@router.get("/runs/{run_id}/events")
async def run_events(
    request: Request,
    run_id: str,
    graph: Any = Depends(get_graph),
    neo4j: Neo4jClient = Depends(get_neo4j),
) -> EventSourceResponse:
    """SSE stream of LangGraph state updates merged with adapter side-effects.

    Use case overrides this to call `graph.astream(...)` with its initial state.
    The boilerplate version just emits sse_bus events and heartbeats — useful
    for testing the SSE path without a wired graph.
    """
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="graph not yet wired — see src/graph/builder.py",
        )

    queue = sse_bus.subscribe(run_id)

    async def event_generator() -> AsyncIterator[dict[str, Any]]:
        try:
            yield {"event": "run.start", "data": json.dumps({"run_id": run_id})}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield event
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": "{}"}
            yield {"event": "run.complete", "data": json.dumps({"run_id": run_id})}
        finally:
            sse_bus.unsubscribe(run_id)
            await record_run_end(neo4j, run_id, status="streamed")

    return EventSourceResponse(event_generator())
