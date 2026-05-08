"""Interrupt node — pauses the graph for human clarification.

Uses LangGraph's `interrupt()` primitive: the graph halts at this node, the
caller's `astream()` yields a special `__interrupt__` chunk carrying the
payload, and execution resumes when the caller invokes the graph again with
`Command(resume=<user_input>)`.

The user's clarification is treated as the new query — we feed it back into
clarity_agent, which now has both the original ambiguous query and the
clarification in `state.messages` and should mark the resolved query CLEAR.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from opentelemetry import trace

from ..state import ResearchState
from ._audit_helpers import safe_record_step

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def interrupt_node(
    state: ResearchState, config: RunnableConfig
) -> dict[str, Any]:
    with tracer.start_as_current_span("node.interrupt_node") as span:
        span.set_attribute("run_id", state.run_id)

        request = (
            state.clarification_request
            or "Could you clarify which company or topic you're asking about?"
        )

        # interrupt() raises a special control-flow exception that LangGraph
        # catches; the value the caller passes via Command(resume=...) becomes
        # the return value here when execution resumes.
        human_response = interrupt(
            {
                "reason": "needs_clarification",
                "request": request,
                "context": {"original_query": state.user_query},
            }
        )

        # Resume path — only reached after the caller resumes with Command(resume=...).
        await safe_record_step(
            config,
            state.run_id,
            "interrupt_node",
            latency_ms=0,
            metadata={"human_response_received": True, "response": str(human_response)},
        )

        clarification_text = str(human_response or "").strip()

        return {
            "clarification_response": clarification_text,
            "user_query": clarification_text,
            "clarity_status": None,  # force clarity to re-evaluate
            "company": None,
            "messages": [HumanMessage(content=clarification_text)],
            "audit_log": [
                {
                    "node": "interrupt_node",
                    "human_response": clarification_text,
                }
            ],
        }
