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
    """
    blocks: dict[str, Any] = {}
    for node in DISPLAY_ORDER:
        meta = NODE_META[node]
        blocks[node] = st.status(
            f"{meta['icon']} {meta['label']} — pending",
            state="running" if node == "clarity_agent" else "complete",
            expanded=False,
        )
        # Reset all to "queued" visual, then we'll flip to running when the
        # callback fires for that node. Streamlit doesn't have a pending
        # state, so we use complete-collapsed as the "queued" look.
        with blocks[node]:
            st.caption(meta["summary"])
    return blocks


def _make_callback(blocks: dict[str, Any]) -> Any:
    """Build a per-node update callback that closes over the placeholders."""
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
            cols = st.columns([1, 1, 1, 1])
            cols[0].metric("latency", fmt_ms(delta.get("latency_ms")))

            # cost_usd in delta is the cumulative total_cost_usd, not per-node.
            # Per-node cost is in the audit_log entry.
            audit_entries = delta.get("audit_log") or []
            entry = audit_entries[0] if audit_entries else {}
            cols[1].metric("cost", fmt_money(entry.get("cost_usd")))
            cols[2].metric(
                "in tok", str(entry.get("input_tokens") or entry.get("prompt_tokens") or "—")
            )
            cols[3].metric(
                "out tok", str(entry.get("output_tokens") or entry.get("completion_tokens") or "—")
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

    return on_node_update


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
    callback = _make_callback(blocks)

    if resume_value is not None:
        payload: dict[str, Any] | Command = Command(resume=resume_value)
    else:
        assert user_query is not None
        payload = fresh_payload(user_query)

    final_state = run_async(stream_turn(graph, payload, config, callback))

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

    # Normal chat input.
    user_query = st.chat_input("Ask about a public company…")
    if not user_query:
        return
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.session_state["last_user_query"] = user_query
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("The agents are working…"):
        _run_one_turn(user_query=user_query, resume_value=None)

    # If still awaiting clarification, the rerun below will render the form.
    if st.session_state.get("awaiting_clarification"):
        st.rerun()

    # Show the final answer in a chat bubble.
    if st.session_state.get("last_final_answer"):
        with st.chat_message("assistant"):
            st.markdown(st.session_state["last_final_answer"])
