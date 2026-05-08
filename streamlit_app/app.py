"""Streamlit showcase entry point.

Run:
    uv run streamlit run streamlit_app/app.py --server.fileWatcherType=none

The `--server.fileWatcherType=none` flag matters during a demo: without
it, every save under `src/` reloads Streamlit and tears down the cached
graph + checkpointer, which costs ~5s of demo time.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _setup_stdout_utf8() -> None:
    """Windows console UTF-8 — only when running as main, not on test import."""
    import io

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import streamlit as st  # noqa: E402

from streamlit_app.components import runtime_settings_pills  # noqa: E402
from streamlit_app.pages import (  # noqa: E402
    architecture_tab,
    chat_tab,
    eval_tab,
    memory_tab,
    retrieval_tab,
    settings_tab,
    trace_tab,
)
from streamlit_app.state import (  # noqa: E402
    init_session_state,
    reset_conversation,
    runtime_settings_summary,
)


def _sidebar() -> None:
    with st.sidebar:
        st.title("🔬 Research Assistant")
        st.caption("Multi-agent LangGraph demo")

        summary = runtime_settings_summary()
        st.subheader("Live config")
        runtime_settings_pills(summary)

        st.divider()
        st.subheader("Session")
        st.text(f"thread_id: {st.session_state['thread_id'][:8]}…")
        st.text(f"user_id:   {st.session_state['user_id']}")
        st.metric("turn", st.session_state["turn_count"])
        st.metric("session cost", f"${st.session_state['session_cost_usd']:.5f}")

        if summary["langsmith_key"]:
            st.success("LangSmith: enabled")
            st.caption(
                "Traces flow to your `research-assistant` project at "
                "smith.langchain.com"
            )
        else:
            st.warning("LangSmith: disabled (set LANGSMITH_API_KEY in .env)")

        st.divider()
        if st.button("🔄 New conversation", use_container_width=True):
            reset_conversation()
            st.rerun()


def main() -> None:
    _setup_stdout_utf8()
    st.set_page_config(
        page_title="Research Assistant — Demo",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    _sidebar()

    st.title("🔬 Multi-Agent Research Assistant")
    st.caption(
        "Live demo: 4 LangGraph agents + interrupt + retry loop + Tavily + "
        "rerank + LLM-as-judge eval + cross-thread memory."
    )

    tabs = st.tabs(
        [
            "💬 Chat",
            "📊 Pipeline Trace",
            "🔎 Retrieval Lens",
            "🧠 Memory",
            "🧪 Eval Console",
            "⚙️ Settings",
            "🏗️ Architecture",
        ]
    )

    with tabs[0]:
        chat_tab.render()
    with tabs[1]:
        trace_tab.render()
    with tabs[2]:
        retrieval_tab.render()
    with tabs[3]:
        memory_tab.render()
    with tabs[4]:
        eval_tab.render()
    with tabs[5]:
        settings_tab.render()
    with tabs[6]:
        architecture_tab.render()


if __name__ == "__main__":
    main()
