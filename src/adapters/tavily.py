"""Tavily live-search adapter (with mock fallback).

Wraps the official `tavily-python` AsyncTavilyClient behind a stable shape so
the Research Agent doesn't need to know which backend ran (port-adapter
pattern). Live mode hits the real Tavily API; mock mode synthesizes hits
from `data/mock_companies.py` so unit tests stay hermetic.

Backend selection: `settings.RESEARCH_BACKEND`. When set to "tavily" but
`TAVILY_API_KEY` is missing or empty, we fall through to mock with a warning
log — this keeps offline dev workable without a hard crash.

The tenacity-style retry pattern from src/llm/client.py isn't replicated here;
Tavily's SDK has its own internal retries and timeouts. The wrapper does add
a wall-clock timeout via asyncio.wait_for so a hung connection doesn't block
the node forever.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from pydantic import BaseModel, Field

from ..config import get_settings

logger = logging.getLogger(__name__)


TAVILY_TIMEOUT_SECONDS = 30.0


class TavilyHit(BaseModel):
    """One search result. Same shape regardless of live/mock backend."""

    title: str
    url: str
    content: str = Field(description="Snippet or summary text")
    score: float = Field(default=0.0, ge=0.0, le=1.0)


def _format_hits(hits: list[TavilyHit]) -> str:
    """Render hits as a flat numbered list for prompt inclusion."""
    if not hits:
        return "(no search results)"
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] {h.title}\n    URL: {h.url}\n    {h.content}")
    return "\n\n".join(lines)


async def tavily_search(
    query: str,
    k: int | None = None,
    topic: Literal["general", "news", "finance"] = "news",
) -> list[TavilyHit]:
    """Search Tavily (or fall back to mock). Returns at most k hits.

    `topic="news"` biases toward recent articles — the right default for the
    Research Assistant's "what's going on with X" framing. Switch to "finance"
    for stock-heavy queries; "general" for everything else.
    """
    settings = get_settings()
    k = k or settings.TAVILY_MAX_RESULTS

    if settings.RESEARCH_BACKEND == "mock":
        return _mock_search(query, k)

    if not settings.TAVILY_API_KEY:
        logger.warning(
            "RESEARCH_BACKEND=tavily but TAVILY_API_KEY missing — falling back to mock",
        )
        return _mock_search(query, k)

    return await _live_search(query, k, topic, settings.TAVILY_API_KEY)


async def _live_search(
    query: str, k: int, topic: Literal["general", "news", "finance"], api_key: str
) -> list[TavilyHit]:
    # Imported here so unit tests that monkey-patch RESEARCH_BACKEND=mock
    # never need the SDK installed (defence-in-depth; tavily-python is in deps).
    from tavily import AsyncTavilyClient

    client = AsyncTavilyClient(api_key=api_key)
    try:
        try:
            response = await asyncio.wait_for(
                client.search(
                    query=query,
                    topic=topic,
                    max_results=k,
                    search_depth="basic",
                ),
                timeout=TAVILY_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning("tavily search timed out", extra={"query": query})
            return []
        except Exception as exc:
            logger.exception("tavily search failed", extra={"query": query, "error": str(exc)})
            return []
    finally:
        await client.close()

    raw_hits = response.get("results") or []
    return [
        TavilyHit(
            title=str(r.get("title") or "(untitled)"),
            url=str(r.get("url") or ""),
            content=str(r.get("content") or "")[:2000],
            score=float(r.get("score") or 0.0),
        )
        for r in raw_hits
    ]


def _mock_search(query: str, k: int) -> list[TavilyHit]:
    """Deterministic stub hits derived from data/mock_companies.py.

    Splits the canned facts into 'hits' so the downstream LLM digest logic
    sees the same shape it would from a live call.
    """
    from data.mock_companies import MOCK_RESEARCH, normalize_company_name

    canonical = _extract_company_from_query(query)
    if canonical is None or canonical not in MOCK_RESEARCH:
        # Try ANY token in the query
        for token in query.split():
            cn = normalize_company_name(token)
            if cn and cn in MOCK_RESEARCH:
                canonical = cn
                break

    if canonical is None or canonical not in MOCK_RESEARCH:
        return []

    facts = MOCK_RESEARCH[canonical]
    return [
        TavilyHit(
            title=f"{canonical} — recent news",
            url=f"https://mock.local/{canonical.lower().replace(' ', '-')}/news",
            content=facts["recent_news"],
            score=0.95,
        ),
        TavilyHit(
            title=f"{canonical} — stock & financials",
            url=f"https://mock.local/{canonical.lower().replace(' ', '-')}/stock",
            content=facts["stock_info"],
            score=0.90,
        ),
        TavilyHit(
            title=f"{canonical} — strategic developments",
            url=f"https://mock.local/{canonical.lower().replace(' ', '-')}/strategy",
            content=facts["key_developments"],
            score=0.85,
        ),
    ][:k]


def _extract_company_from_query(query: str) -> str | None:
    """Best-effort canonical-name extraction for the mock fallback."""
    from data.mock_companies import normalize_company_name

    cn = normalize_company_name(query)
    if cn:
        return cn
    for token in query.split():
        cn = normalize_company_name(token)
        if cn:
            return cn
    return None
