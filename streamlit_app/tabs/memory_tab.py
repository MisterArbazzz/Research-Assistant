"""Memory tab — Tier 4 cross-thread per-user fact store."""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from src.memory import list_facts, retrieve_relevant_facts, store_fact
from src.memory.longterm import LONGTERM_DB_PATH
from streamlit_app.state import run_async


def _render_facts(user_id: str) -> None:
    facts = run_async(list_facts(user_id, limit=200))
    if not facts:
        st.info(
            f"No stored facts for `{user_id}`. Run a turn that mentions a "
            "preference (e.g. *'I prefer technical detail'*) and the Memory "
            "Writer will save it."
        )
        return
    st.markdown(f"**{len(facts)}** fact(s) for `{user_id}`:")
    for f in facts:
        with st.container(border=True):
            st.markdown(f"_{f['fact']}_")
            st.caption(
                f"id={f['id']} · created={f['created_at']} · "
                f"thread={(f['source_thread_id'] or '?')[:8]}…"
            )


def render() -> None:
    st.subheader("Long-term Memory — Tier 4")
    st.caption(
        "SQLite + sqlite-vec store keyed by `user_id`. Survives across "
        "threads / sessions / app restarts. The Memory Writer node selects "
        "what to persist; this tab lets you inspect and curate the store."
    )

    col_top = st.columns([3, 2])
    with col_top[0]:
        user_id = st.text_input(
            "user_id",
            value=st.session_state["user_id"],
            help="Switch to inspect a different user's memory.",
        )
        if user_id != st.session_state["user_id"]:
            st.session_state["user_id"] = user_id
            st.rerun()
    with col_top[1]:
        st.caption(f"DB file: `{LONGTERM_DB_PATH}`")
        if st.button("🗑️ Wipe long-term memory (all users)") and Path(
            LONGTERM_DB_PATH
        ).exists():
            Path(LONGTERM_DB_PATH).unlink()
            st.success("Wiped.")
            st.rerun()

    st.divider()
    st.markdown("### Stored facts")
    _render_facts(user_id)

    st.divider()
    st.markdown("### Test retrieval")
    st.caption(
        "Embed a query and find the top-3 most-relevant facts for this "
        "user_id. This is exactly what the Research Agent does behind the "
        "scenes when LONGTERM_MEMORY_ENABLED=true."
    )
    test_query = st.text_input(
        "Query",
        value="how does the user prefer answers formatted?",
        key="memory_test_query",
    )
    if st.button("Retrieve relevant facts"):
        with st.spinner("Embedding query and searching…"):
            facts = run_async(retrieve_relevant_facts(user_id, test_query, k=3))
        if not facts:
            st.warning(
                "No facts retrieved. Either the user has none stored, or "
                "nothing was a close enough match."
            )
        else:
            st.success(f"Retrieved {len(facts)} fact(s):")
            for f in facts:
                st.markdown(f"- {f}")

    st.divider()
    st.markdown("### Manually add a fact")
    st.caption(
        "Useful to seed memory before showing a cross-thread demo. Format "
        "in third person: _'The user prefers technical detail'_."
    )
    col_add = st.columns([4, 1])
    with col_add[0]:
        new_fact = st.text_input(
            "Fact text",
            placeholder="The user is researching EV companies",
            key="memory_new_fact",
        )
    with col_add[1]:
        add_clicked = st.button("Store", type="primary", use_container_width=True)
    if add_clicked and new_fact.strip():
        with st.spinner("Embedding and writing…"):
            row_id = run_async(
                store_fact(
                    user_id,
                    new_fact.strip(),
                    source_thread_id=f"manual-{uuid.uuid4().hex[:6]}",
                )
            )
        if row_id > 0:
            st.success(f"Stored as id={row_id}")
            st.rerun()
        else:
            st.error("Empty fact — nothing stored.")
