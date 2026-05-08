"""Liveness + dependency ping.

Returns a flat status object the orchestrator can poll. The Gemini ping is
cached for 60 seconds so a tight probe loop doesn't burn quota. Returns
HTTP 200 even when a dep is degraded — the response body's `status`
reports `healthy | degraded | unhealthy`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter
from langchain_core.messages import HumanMessage

from ..llm.client import get_chat_model

logger = logging.getLogger(__name__)
router = APIRouter()

LLM_PING_TTL_SECONDS = 60.0

# TTL cache for the LLM ping — avoids burning a Gemini call on every probe.
# Process-local on purpose: each replica probes once per minute, no Redis.
_llm_cache: dict[str, Any] = {"ok": None, "ts": 0.0, "error": None}


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
async def health() -> dict[str, Any]:
    llm_ok, llm_err = await _check_llm()
    return {
        "status": "healthy" if llm_ok else "unhealthy",
        "llm": "ok" if llm_ok else llm_err,
    }
