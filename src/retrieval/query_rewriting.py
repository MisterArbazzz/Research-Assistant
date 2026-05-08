"""Query rewriting — turn a conversational query into a clean retrieval query.

Search engines work better when the input looks like a search query than when
it looks like a conversation turn. The rewriter takes:

  - the user's current message ("What about their CEO?")
  - the canonical company (resolved by Clarity)
  - the last few conversation messages

…and produces a single search-friendly string ("Tesla CEO Elon Musk
leadership compensation 2025"). Especially useful for follow-up turns where
the user's message alone is meaningless without history.

One Gemini Flash call. Costs a few hundredths of a cent. Skipped entirely
when QUERY_REWRITE_ENABLED is false in settings.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import get_settings
from ..graph.prompts import format_history
from ..llm.client import ainvoke_structured, get_chat_model

logger = logging.getLogger(__name__)


class RewrittenQuery(BaseModel):
    """Structured output: a clean search query plus a one-line rationale."""

    rewritten_query: str = Field(min_length=1)
    rationale: str = Field(default="", description="One-line note for audit trail")


_SYSTEM_PROMPT = (
    "You rewrite conversational user questions into clean search queries for a "
    "live web-search engine. The rewritten query should:\n"
    "  - mention the company by canonical name (e.g. 'Tesla', 'Apple Inc.')\n"
    "  - include any topic-specific terms the user implied (CEO, earnings, "
    "stock price, products, etc.)\n"
    "  - resolve pronouns and vague references ('their', 'the company', 'it') "
    "using the conversation history\n"
    "  - be short — 4 to 10 words is ideal\n"
    "  - NOT include question marks or filler words ('what about', 'tell me')\n\n"
    "Examples:\n"
    "  history: User: Tell me about Tesla / Assistant: Tesla makes EVs...\n"
    "  user message: What about their CEO?\n"
    "  → rewritten_query: 'Tesla CEO Elon Musk leadership'\n\n"
    "  history: (empty)\n"
    "  user message: Tell me about Apple's recent news and stock\n"
    "  → rewritten_query: 'Apple Inc recent news stock price'\n\n"
    "Return both the rewritten query AND a one-line rationale (what you "
    "changed and why)."
)


async def rewrite_query(
    user_query: str,
    company: str | None,
    history: list[BaseMessage],
) -> tuple[str, dict[str, Any] | None, str]:
    """Rewrite `user_query` into a search-friendly query.

    Returns (rewritten_query, usage_metadata, rationale). When the feature is
    disabled or the call fails, returns the original query unchanged with
    None usage and an explanatory rationale.
    """
    settings = get_settings()
    if not settings.QUERY_REWRITE_ENABLED:
        return user_query, None, "rewriting disabled"

    company_hint = company or "(not yet resolved)"
    history_block = format_history(history[-6:] if history else [])
    prompt = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Canonical company (from Clarity Agent): {company_hint}\n"
                f"Conversation history:\n{history_block}\n\n"
                f"User message to rewrite: {user_query}"
            )
        ),
    ]

    llm = get_chat_model("primary")
    try:
        parsed, usage = await ainvoke_structured(llm, prompt, RewrittenQuery)
    except ValueError as exc:
        logger.warning(
            "query rewrite failed — falling back to original query",
            extra={"error": str(exc)},
        )
        return user_query, None, f"rewrite parse failure: {exc}"

    return parsed.rewritten_query, usage, parsed.rationale
