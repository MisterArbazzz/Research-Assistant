"""Chat tab — live multi-agent execution.

Each user query renders 6 `st.status` blocks (one per node). As LangGraph
emits per-node updates via `astream`, blocks transition pending → running
→ done with their cost / latency / tokens / key state delta visible.

If the graph pauses on `interrupt()`, the chat input switches to a
"clarification reply" box; submitting resumes the run via
`Command(resume=...)`.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from langgraph.types import Command

from streamlit_app.components import NODE_META, fmt_money, fmt_ms
from streamlit_app.graph_runner import DISPLAY_ORDER, fresh_payload, stream_turn
from streamlit_app.state import get_graph_and_saver, run_async


def _build_status_blocks() -> dict[str, Any]:
    """Render an empty st.status block per node and return their handles.

    Build them BEFORE entering the async loop — once the script function
    yields back to the loop, you can't add new top-level containers.

    All blocks start with state="running" (spinner). The callback flips
    each to "complete" as that node fires. Nodes that never fire on this
    run (interrupt_node when query is clear, validator when confidence
    is high) get marked "skipped" by `mark_unfired_blocks_skipped` after
    the turn completes.
    """
    blocks: dict[str, Any] = {}
    for node in DISPLAY_ORDER:
        meta = NODE_META[node]
        blocks[node] = st.status(
            f"{meta['icon']} {meta['label']} — queued",
            state="running",
            expanded=False,
        )
        with blocks[node]:
            st.caption(meta["summary"])
    return blocks


def mark_unfired_blocks_skipped(blocks: dict[str, Any], fired: set[str]) -> None:
    """After a turn completes, downgrade any block that never fired."""
    for node, block in blocks.items():
        if node in fired:
            continue
        meta = NODE_META[node]
        block.update(
            label=f"{meta['icon']} {meta['label']} — skipped (routing)",
            state="complete",
            expanded=False,
        )


def render_static_blocks_from_audit(audit_log: list[dict[str, Any]]) -> None:
    """Re-render the per-node status blocks from a saved audit log.

    Used on reruns AFTER a turn completes — the live callbacks fired
    during the user's submit, then `st.rerun()` replays the script
    without a new query, so we lose the live blocks. This renders them
    back from `st.session_state["last_run_audit"]` for visual continuity.
    """
    audit_by_node: dict[str, dict[str, Any]] = {}
    for entry in audit_log:
        node = entry.get("node")
        if node:
            # Last entry per node wins (handles validator-retry case where
            # research_agent appears twice — show the most recent).
            audit_by_node[node] = entry

    for node in DISPLAY_ORDER:
        meta = NODE_META[node]
        entry = audit_by_node.get(node)
        if entry is None:
            block = st.status(
                f"{meta['icon']} {meta['label']} — skipped (routing)",
                state="complete",
                expanded=False,
            )
            with block:
                st.caption(meta["summary"])
            continue

        block = st.status(
            f"{meta['icon']} {meta['label']} — done",
            state="complete",
            expanded=False,
        )
        with block:
            st.caption(meta["summary"])
            cols = st.columns([1, 1, 1, 1])
            cols[0].metric("latency", fmt_ms(entry.get("latency_ms")))
            cols[1].metric("cost", fmt_money(entry.get("cost_usd")))
            cols[2].metric(
                "in tok",
                str(entry.get("input_tokens") or entry.get("prompt_tokens") or "—"),
            )
            cols[3].metric(
                "out tok",
                str(entry.get("output_tokens") or entry.get("completion_tokens") or "—"),
            )

            # Node-specific summary lines — same fields the live callback
            # surfaces, all read from the audit entry (which is now
            # complete with input_tokens etc).
            if node == "clarity_agent" and "clarity_status" in entry:
                comp = entry.get("company") or "—"
                st.markdown(
                    f"**Status:** `{entry['clarity_status']}`  ·  **Company:** `{comp}`"
                )
            elif node == "research_agent":
                if "rewritten_query" in entry:
                    st.markdown(f"**Rewritten query:** `{entry['rewritten_query']}`")
                if "tavily_hits" in entry:
                    st.markdown(
                        f"**Hits:** {entry['tavily_hits']} (reranked) · "
                        f"**Confidence:** `{entry.get('confidence_score', '—')}`"
                    )
                if entry.get("memory_facts_retrieved"):
                    st.info(
                        f"💡 Used {entry['memory_facts_retrieved']} long-term memory facts "
                        f"about user `{entry.get('user_id', '?')}`"
                    )
            elif node == "validator_agent" and "validation_result" in entry:
                result = entry["validation_result"]
                if result == "sufficient":
                    st.success(f"Validation: **{result}**")
                else:
                    st.warning(f"Validation: **{result}**")
                if entry.get("notes"):
                    st.markdown("**Notes:**")
                    for n in entry["notes"]:
                        st.markdown(f"- {n}")
            elif node == "synthesis_agent" and entry.get("answer_chars"):
                st.caption(f"Answer ready · {entry['answer_chars']} chars")
            elif node == "memory_writer":
                fw = entry.get("facts_written", 0)
                facts = entry.get("facts") or []
                if fw:
                    st.success(f"Stored {fw} fact(s) for future sessions")
                    for f in facts:
                        st.markdown(f"- _{f}_")
                else:
                    st.caption("Nothing durable to remember from this turn.")


def _make_callback(blocks: dict[str, Any]) -> tuple[Any, set[str]]:
    """Build a per-node update callback. Returns (callback, fired_set).

    The fired_set is updated as nodes report in; the caller uses it to
    mark unfired nodes as skipped after the turn completes.
    """
    seen: set[str] = set()

    def on_node_update(node: str, delta: dict[str, Any]) -> None:
        if node not in blocks:
            return
        seen.add(node)
        meta = NODE_META[node]
        block = blocks[node]
        # Update the block's title and contents.
        block.update(
            label=f"{meta['icon']} {meta['label']} — done", state="complete", expanded=True
        )
        with block:
            # All per-node telemetry lives inside the audit_log entry the
            # node returns; the top-level delta only has the state fields
            # the routing layer cares about (cost is cumulative there, etc.)
            audit_entries = delta.get("audit_log") or []
            entry = audit_entries[0] if audit_entries else {}
            cols = st.columns([1, 1, 1, 1])
            cols[0].metric("latency", fmt_ms(entry.get("latency_ms")))
            cols[1].metric("cost", fmt_money(entry.get("cost_usd")))
            cols[2].metric(
                "in tok",
                str(entry.get("input_tokens") or entry.get("prompt_tokens") or "—"),
            )
            cols[3].metric(
                "out tok",
                str(entry.get("output_tokens") or entry.get("completion_tokens") or "—"),
            )

            # Node-specific summary line
            if node == "clarity_agent" and "clarity_status" in delta:
                comp = delta.get("company") or "—"
                st.markdown(
                    f"**Status:** `{delta['clarity_status']}`  ·  **Company:** `{comp}`"
                )
                if delta.get("clarification_request"):
                    st.warning(f"Asks: _{delta['clarification_request']}_")
            elif node == "research_agent":
                if "rewritten_query" in entry:
                    st.markdown(f"**Rewritten query:** `{entry['rewritten_query']}`")
                if "tavily_hits" in entry:
                    st.markdown(
                        f"**Hits:** {entry['tavily_hits']} (reranked) · "
                        f"**Confidence:** `{delta.get('confidence_score', '—')}`"
                    )
                if entry.get("memory_facts_retrieved"):
                    st.info(
                        f"💡 Used {entry['memory_facts_retrieved']} long-term memory facts "
                        f"about user `{entry.get('user_id', '?')}`"
                    )
            elif node == "validator_agent" and "validation_result" in delta:
                result = delta["validation_result"]
                if result == "sufficient":
                    st.success(f"Validation: **{result}**")
                else:
                    st.warning(f"Validation: **{result}** — triggering retry")
                if delta.get("validation_notes"):
                    st.markdown("**Notes:**")
                    for n in delta["validation_notes"]:
                        st.markdown(f"- {n}")
            elif node == "synthesis_agent" and "final_answer" in delta:
                # Don't re-render full answer here (it streams below); show stat only
                ans = delta["final_answer"] or ""
                st.caption(f"Answer ready · {len(ans)} chars")
            elif node == "memory_writer":
                fw = entry.get("facts_written", 0)
                facts = entry.get("facts") or []
                if fw:
                    st.success(f"Stored {fw} fact(s) for future sessions")
                    for f in facts:
                        st.markdown(f"- _{f}_")
                else:
                    st.caption("Nothing durable to remember from this turn.")

    return on_node_update, seen


def _run_one_turn(user_query: str | None, resume_value: str | None) -> None:
    """Execute one turn through the graph, with live UI updates.

    Pass `user_query` to start a fresh turn; pass `resume_value` to resume
    after an interrupt. (Exactly one should be provided per call.)
    """
    graph, _saver = get_graph_and_saver()
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
            "user_id": st.session_state["user_id"],
        }
    }

    blocks = _build_status_blocks()
    callback, fired = _make_callback(blocks)

    if resume_value is not None:
        payload: dict[str, Any] | Command = Command(resume=resume_value)
    else:
        assert user_query is not None
        payload = fresh_payload(user_query)

    final_state = run_async(stream_turn(graph, payload, config, callback))

    # Mark any node that didn't fire (e.g. validator skipped on high
    # confidence, interrupt skipped on a clear query).
    mark_unfired_blocks_skipped(blocks, fired)

    interrupt_payload = final_state.get("__interrupt__")
    if interrupt_payload:
        st.session_state["awaiting_clarification"] = True
        st.session_state["clarification_payload"] = interrupt_payload
        return

    # Run completed — surface the final answer + persist for other tabs.
    audit = final_state.get("audit_log") or []
    answer = final_state.get("final_answer") or "(no answer produced)"

    st.session_state["awaiting_clarification"] = False
    st.session_state["clarification_payload"] = None
    st.session_state["last_run_audit"] = audit
    st.session_state["last_run_findings"] = final_state.get("research_findings")
    st.session_state["last_final_answer"] = answer
    st.session_state["session_cost_usd"] = final_state.get(
        "total_cost_usd", st.session_state.get("session_cost_usd", 0.0)
    )
    st.session_state["turn_count"] += 1
    st.session_state["messages"].append({"role": "assistant", "content": answer})


def render() -> None:
    st.subheader("Chat with the assistant")
    st.caption(
        "Each turn lights up the graph live. Try an ambiguous query like "
        "*'How is the company doing?'* to see the human-in-the-loop interrupt fire."
    )

    # Render conversation history.
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Re-render the per-node breakdown from the most recent run.
    # During a live submit we render fresh blocks below (replacing this);
    # on subsequent reruns this preserves the visual breakdown so the
    # user can see what the agents did even after the spinner exits.
    last_audit = st.session_state.get("last_run_audit") or []
    if last_audit and not st.session_state.get("awaiting_clarification"):
        with st.expander("📊 Last turn — per-node breakdown", expanded=True):
            render_static_blocks_from_audit(last_audit)

    # If we're awaiting clarification from the user, surface that UI.
    if st.session_state.get("awaiting_clarification"):
        payload = st.session_state["clarification_payload"] or {}
        request = payload.get("request") or "Could you clarify your question?"
        st.warning(f"🛑 Graph paused — **{request}**")
        with st.form("clarification_form", clear_on_submit=True):
            reply = st.text_input("Your clarification", placeholder="e.g. 'Tesla'")
            submitted = st.form_submit_button("Send clarification")
        if submitted and reply.strip():
            st.session_state["messages"].append(
                {"role": "user", "content": f"_(clarification)_ {reply}"}
            )
            with st.spinner("Resuming the graph…"):
                _run_one_turn(user_query=None, resume_value=reply.strip())
            st.rerun()
        return

    # First-turn UX: prefilled form so the user gets a one-click demo path.
    # `st.chat_input` doesn't support pre-filled values, so we use a form
    # with `st.text_input` only on turn 0; subsequent turns use the normal
    # bottom-pinned chat_input.
    default_first_query = "Tell me about Apple's recent stock news"

    user_query: str | None = None
    if st.session_state.get("turn_count", 0) == 0:
        with st.form("first_query_form", clear_on_submit=False):
            prefilled = st.text_input(
                "Your question",
                value=default_first_query,
                label_visibility="collapsed",
            )
            send = st.form_submit_button("Send →", type="primary", use_container_width=True)
        if send and prefilled.strip():
            user_query = prefilled.strip()
    else:
        user_query = st.chat_input("Ask about a public company…")

    if not user_query:
        return

    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.session_state["last_user_query"] = user_query
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("The agents are working…"):
        _run_one_turn(user_query=user_query, resume_value=None)

    # Always rerun after a turn finishes (interrupt OR success) so the
    # sidebar metrics (turn_count, session cost) re-render with the new
    # values, and the assistant's reply renders via the message-history
    # loop at the top of the function. Without this, the sidebar shows
    # stale values because it renders BEFORE chat_tab in main()'s top-
    # to-bottom script execution.
    st.rerun()
