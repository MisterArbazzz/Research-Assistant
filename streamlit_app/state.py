"""Session-state bootstrap + persistent event loop + cached graph.

The Streamlit app needs to run async code (LangGraph, Tavily, sqlite-vec)
from synchronous button callbacks. The naive `asyncio.run()` per call
recreates the loop each time, which invalidates aiosqlite connections
held by AsyncSqliteSaver and the cached flashrank ranker. That breaks on
the second turn (interrupt resume) with `RuntimeError: got Future
attached to a different loop`.

Fix: ONE persistent loop per Streamlit session, owned by `session_state`,
and cached resources (graph + checkpointer) are bound to that loop.
Every async call goes through `run_async(coro)` below.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.config import get_settings
from src.graph.builder import build_graph
from src.logging_config import configure_logging
from src.observability import configure_langsmith, configure_tracer

CHECKPOINT_DB = Path("./data/streamlit_checkpoints.db")
CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)

# Marker we use to detect first-render bootstrap inside a session.
_BOOTSTRAP_KEY = "_bootstrapped"


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """Get the session's persistent loop; create one if missing."""
    if "loop" not in st.session_state:
        loop = asyncio.new_event_loop()
        st.session_state["loop"] = loop
    return st.session_state["loop"]  # type: ignore[no-any-return]


def run_async(coro: Any) -> Any:
    """Run `coro` on the session's persistent event loop.

    Use this everywhere instead of `asyncio.run()`. Resources captured by
    the coroutine (Saver, ranker, etc.) stay bound to the same loop across
    Streamlit reruns and across interrupt-resume cycles.
    """
    loop = _get_or_create_loop()
    return loop.run_until_complete(coro)


@st.cache_resource(show_spinner="Building agent graph (one-time)…")
def get_graph_and_saver() -> tuple[Any, Any]:
    """Build the compiled graph + AsyncSqliteSaver once per process.

    `st.cache_resource` survives Streamlit reruns within a session AND
    across sessions on the same server, which is what we want — the
    flashrank model load (~3s) and sqlite WAL setup happen exactly once.

    The AsyncSqliteSaver is normally an async context manager; we enter
    its context manually here and stash the AsyncExitStack on the cached
    object so the WAL gets a chance to flush on shutdown.
    """
    configure_logging()
    configure_tracer()
    configure_langsmith()

    loop = _get_or_create_loop()

    async def _build() -> tuple[Any, Any, AsyncExitStack]:
        stack = AsyncExitStack()
        saver = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB))
        )
        graph = await build_graph(saver)
        return graph, saver, stack

    graph, saver, _stack = loop.run_until_complete(_build())
    # Best-effort cleanup — Streamlit doesn't expose a clean shutdown hook
    # in cache_resource, but SQLite WAL recovers automatically on next open.
    return graph, saver


def init_session_state() -> None:
    """Initialize session_state keys on first render of the session."""
    if st.session_state.get(_BOOTSTRAP_KEY):
        return

    st.session_state[_BOOTSTRAP_KEY] = True
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("user_id", "interviewer")
    st.session_state.setdefault("messages", [])  # rendered chat history
    st.session_state.setdefault("turn_count", 0)
    st.session_state.setdefault("session_cost_usd", 0.0)
    st.session_state.setdefault("last_run_audit", [])
    st.session_state.setdefault("last_run_findings", None)
    st.session_state.setdefault("last_user_query", None)
    st.session_state.setdefault("last_final_answer", None)
    st.session_state.setdefault("awaiting_clarification", False)
    st.session_state.setdefault("clarification_payload", None)
    st.session_state.setdefault("settings_overrides", {})
    st.session_state.setdefault("ab_history", [])  # for A/B compare widget


def reset_conversation() -> None:
    """Rotate the thread_id; per-conversation state is wiped."""
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["messages"] = []
    st.session_state["turn_count"] = 0
    st.session_state["session_cost_usd"] = 0.0
    st.session_state["last_run_audit"] = []
    st.session_state["last_run_findings"] = None
    st.session_state["last_user_query"] = None
    st.session_state["last_final_answer"] = None
    st.session_state["awaiting_clarification"] = False
    st.session_state["clarification_payload"] = None


def apply_settings_overrides(overrides: dict[str, str]) -> None:
    """Write overrides to os.environ and reset the cached settings.

    Most fields are read at call-time via `get_settings()`, so this hot-
    swaps live without rebuilding the graph. Exception: `RERANK_MODEL`
    is captured by the lazy flashrank singleton on first call — changing
    it requires `st.cache_resource.clear()` to rebuild the graph and the
    ranker.
    """
    for k, v in overrides.items():
        os.environ[k] = str(v)
    get_settings.cache_clear()
    st.session_state["settings_overrides"] = dict(overrides)


def runtime_settings_summary() -> dict[str, Any]:
    """Compact dict of the values currently driving behaviour."""
    s = get_settings()
    return {
        "model_primary": s.MODEL_PRIMARY,
        "model_qa": s.MODEL_QA,
        "research_backend": s.RESEARCH_BACKEND,
        "tavily_key": bool(s.TAVILY_API_KEY),
        "langsmith_key": bool(s.LANGSMITH_API_KEY),
        "rerank_enabled": s.RERANK_ENABLED,
        "rewrite_enabled": s.QUERY_REWRITE_ENABLED,
        "memory_enabled": s.LONGTERM_MEMORY_ENABLED,
        "max_attempts": s.MAX_RESEARCH_ATTEMPTS,
        "cost_ceiling": s.COST_CEILING_PER_RUN_USD,
        "tavily_max_results": s.TAVILY_MAX_RESULTS,
        "rerank_top_k": s.RERANK_TOP_K,
    }
