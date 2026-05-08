"""Research Assistant graph state schema.

Pydantic v2 model that LangGraph passes between nodes. The two reducer-typed
fields (audit_log, messages) accumulate across nodes; everything else is
overwritten by each node's return dict.

Multi-turn conversation works because LangGraph's checkpointer persists this
state under a thread_id; the `messages` reducer (`add_messages`) appends each
turn's HumanMessage / AIMessage onto the running history.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class ParsedClarity(BaseModel):
    """Structured output from the Clarity Agent."""

    clarity_status: Literal["clear", "needs_clarification"]
    company: str | None = None
    clarification_request: str | None = None


class ResearchFindings(BaseModel):
    """Structured output from the Research Agent."""

    recent_news: str = Field(min_length=1)
    stock_info: str = Field(min_length=1)
    key_developments: str = Field(min_length=1)
    confidence_score: float = Field(ge=0, le=10)


class ValidationVerdict(BaseModel):
    """Structured output from the Validator Agent."""

    validation_result: Literal["sufficient", "insufficient"]
    notes: list[str] = Field(default_factory=list)


class ResearchState(BaseModel):
    """LangGraph state for the Research Assistant.

    Per CLAUDE.md pattern #1: no `max_length` on any LLM-filled string field —
    Gemini does not honor schema-level length caps. Length goes in the prompt.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    user_query: str = ""

    # Clarity agent outputs
    clarity_status: Literal["clear", "needs_clarification"] | None = None
    clarification_request: str | None = None
    clarification_response: str | None = None

    # Research agent outputs
    company: str | None = None
    research_findings: dict[str, Any] | None = None
    confidence_score: float | None = None
    research_attempts: int = 0

    # Validator agent outputs
    validation_result: Literal["sufficient", "insufficient"] | None = None
    validation_notes: list[str] = Field(default_factory=list)

    # Synthesis output
    final_answer: str | None = None

    # Bookkeeping
    total_cost_usd: float = 0.0
    audit_log: Annotated[list[dict[str, Any]], add] = Field(default_factory=list)
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
