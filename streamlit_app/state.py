"""Session-state bootstrap + persistent event-loop thread + cached graph.

The Streamlit app needs to run async code (LangGraph, Tavily, sqlite-vec)
from synchronous Streamlit callbacks. The naive approaches all break:

  - `asyncio.run()` per call recreates the loop and invalidates aiosqlite
    connections cached by AsyncSqliteSaver.
  - A persistent loop in `st.session_state` runs on whichever Streamlit
    ScriptRunner thread happened to be active when first created. On the
    NEXT rerun Streamlit spawns a different ScriptRunner thread; calling
    `loop.run_until_complete()` from that new thread leaves aiosqlite's
    worker thread orphaned and the connection raises "Connection closed"
    on next use.
  - `nest_asyncio` is incompatible with aiosqlite's threading model.

The fix used here: a SINGLE dedicated background thread that owns the
event loop for the entire process lifetime. Async work is scheduled
across thread boundaries via `asyncio.run_coroutine_threadsafe`. Every
resource bound to the loop (aiosqlite connections, etc.) stays valid as
long as the process is up.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
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

_BOOTSTRAP_KEY = "_bootstrapped"


class _LoopThread:
    """Dedicated daemon thread that owns one asyncio event loop forever."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run,
            name="streamlit-asyncio-loop",
            daemon=True,
        )
        self.thread.start()
        # Wait until the loop is actually running before we hand it out.
        ready = threading.Event()
        self.loop.call_soon_threadsafe(ready.set)
        ready.wait(timeout=5.0)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro: Any) -> Any:
        """Schedule `coro` on the loop's thread; block until it returns."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()


@st.cache_resource(show_spinner=False)
def _get_loop_thread() -> _LoopThread:
    """One LoopThread per process (cached across all sessions)."""
    return _LoopThread()


def run_async(coro: Any) -> Any:
    """Run `coro` on the dedicated background loop, with the calling
    Streamlit session's script-run context attached to the loop thread.

    Why the ctx attach: callbacks invoked from inside the coroutine
    (e.g. live-update callbacks for `st.status` blocks) execute on the
    LoopThread, not the Streamlit ScriptRunner thread. Streamlit element
    methods (`.update()`, `.write()`, etc.) silently no-op when called
    from a thread without an attached script-run context. Attaching the
    caller's ctx for the duration of this run makes those cross-thread
    UI updates land in the right session's render queue.

    Safe for our single-user demo. For multi-tenant deployments you'd
    want one LoopThread per session instead of one per process.
    """
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

    loop_thread = _get_loop_thread()
    ctx = get_script_run_ctx()
    if ctx is not None:
        add_script_run_ctx(loop_thread.thread, ctx)
    try:
        return loop_thread.run(coro)
    finally:
        # Detach so a later run from a different session doesn't inherit
        # this session's context.
        if ctx is not None:
            add_script_run_ctx(loop_thread.thread, None)


# Persistent reference holder — the AsyncSqliteSaver context manager must
# stay alive for the lifetime of the process so its aiosqlite connection
# isn't garbage-collected. `st.cache_resource` keeps the saver itself, but
# the *context manager* that owns the connection's __aexit__ is otherwise
# orphaned. We stash it here.
_KEEPALIVE: list[Any] = []


@st.cache_resource(show_spinner="Building agent graph (one-time, ~3s)…")
def get_graph_and_saver() -> tuple[Any, Any]:
    """Build the compiled graph + AsyncSqliteSaver once per process.

    The saver's aiosqlite connection is bound to the LoopThread's loop —
    every subsequent `run_async(coro)` schedules on that same loop, so
    the connection stays valid forever.
    """
    configure_logging()
    configure_tracer()
    configure_langsmith()

    loop_thread = _get_loop_thread()

    async def _build() -> tuple[Any, Any, Any]:
        # Enter the saver's async context manager manually and keep the cm
        # alive — see _KEEPALIVE module-level list above.
        saver_cm = AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB))
        saver = await saver_cm.__aenter__()
        graph = await build_graph(saver)
        return graph, saver, saver_cm

    graph, saver, saver_cm = loop_thread.run(_build())
    _KEEPALIVE.append(saver_cm)
    return graph, saver


def init_session_state() -> None:
    """Initialize session_state keys on first render of the session."""
    if st.session_state.get(_BOOTSTRAP_KEY):
        return

    st.session_state[_BOOTSTRAP_KEY] = True
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("user_id", "demo")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("turn_count", 0)
    st.session_state.setdefault("session_cost_usd", 0.0)
    st.session_state.setdefault("last_run_audit", [])
    st.session_state.setdefault("last_run_findings", None)
    st.session_state.setdefault("last_user_query", None)
    st.session_state.setdefault("last_final_answer", None)
    st.session_state.setdefault("awaiting_clarification", False)
    st.session_state.setdefault("clarification_payload", None)
    st.session_state.setdefault("settings_overrides", {})
    st.session_state.setdefault("ab_history", [])


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
