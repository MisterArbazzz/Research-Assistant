"""Tavily adapter tests — mock branch only (live call is a manual smoke).

The mock branch is deterministic and exercises the same TavilyHit shape the
Research Agent will see in production, so unit-test confidence is preserved.
"""

from __future__ import annotations

import pytest

from src.adapters.tavily import TavilyHit, _format_hits, tavily_search


@pytest.fixture(autouse=True)
def _force_mock(settings_override):
    settings_override(GOOGLE_API_KEY="x", RESEARCH_BACKEND="mock", TAVILY_API_KEY="")
    from src.config import get_settings

    get_settings.cache_clear()


async def test_mock_returns_hits_for_known_company() -> None:
    hits = await tavily_search("Apple iPhone news", k=5)
    assert len(hits) == 3  # news + stock + strategy
    assert all(isinstance(h, TavilyHit) for h in hits)
    assert any("Apple" in h.title for h in hits)


async def test_mock_returns_empty_for_unknown_company() -> None:
    hits = await tavily_search("ACME Industries widgets", k=5)
    assert hits == []


async def test_mock_alias_resolution() -> None:
    hits = await tavily_search("AAPL stock movement today", k=5)
    assert len(hits) == 3


async def test_format_hits_renders_numbered_list() -> None:
    hits = [
        TavilyHit(title="A", url="https://a", content="alpha", score=0.9),
        TavilyHit(title="B", url="https://b", content="beta", score=0.8),
    ]
    out = _format_hits(hits)
    assert "[1] A" in out
    assert "[2] B" in out
    assert "alpha" in out


async def test_format_hits_handles_empty() -> None:
    assert _format_hits([]) == "(no search results)"
