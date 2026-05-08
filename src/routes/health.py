"""Liveness + dependency pings.

Returns a flat status object the orchestrator can poll. Both Neo4j and
Gemini are pinged; the LLM ping is cached for 60 seconds so a tight probe
loop doesn't burn quota. Returns HTTP 200 even when a dep is degraded —
the response body's `status` reports `healthy | degraded | unhealthy`.
The 503 case is reserved for "the service itself can't answer."
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage

from ..deps import get_neo4j
from ..llm.client import get_chat_model
from ..neo4j_client.driver import Neo4jClient

logger = logging.getLogger(__name__)
router = APIRouter()

LLM_PING_TTL_SECONDS = 60.0

# TTL cache for the LLM ping — avoids burning a Gemini call on every probe.
# Process-local on purpose: each replica probes once per minute, no Redis.
_llm_cache: dict[str, Any] = {"ok": None, "ts": 0.0, "error": None}


async def _check_neo4j(neo4j: Neo4jClient) -> tuple[bool, str | None]:
    try:
        async with neo4j.session() as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            if not record or record["ok"] != 1:
                return False, "ping returned no row"
        return True, None
    except Exception as exc:
        logger.exception("neo4j health check failed")
        return False, str(exc)


async def _check_llm() -> tuple[bool, str | None]:
    now = time.time()
    if (
        _llm_cache["ok"] is not None
        and now - _llm_cache["ts"] < LLM_PING_TTL_SECONDS
    ):
        return bool(_llm_cache["ok"]), _llm_cache["error"]

    try:
        llm = get_chat_model("primary", max_tokens=1)
        await llm.ainvoke([HumanMessage(content="ok")])
        ok, err = True, None
    except Exception as exc:
        logger.warning("llm health check failed", extra={"error": str(exc)})
        ok, err = False, str(exc)

    _llm_cache["ok"] = ok
    _llm_cache["error"] = err
    _llm_cache["ts"] = now
    return ok, err


@router.get("/health")
async def health(neo4j: Neo4jClient = Depends(get_neo4j)) -> dict[str, Any]:
    neo4j_ok, neo4j_err = await _check_neo4j(neo4j)
    llm_ok, llm_err = await _check_llm()

    if neo4j_ok and llm_ok:
        status = "healthy"
    elif neo4j_ok or llm_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "neo4j": "ok" if neo4j_ok else neo4j_err,
        "llm": "ok" if llm_ok else llm_err,
    }
