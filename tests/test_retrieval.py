"""Tests for the Tier-2 retrieval helpers (query rewriting + rerank).

The rerank tests stub flashrank's Ranker to avoid downloading the ONNX
model in CI / pre-commit. The pass-through path (RERANK_ENABLED=false)
needs no stubbing and exercises the real code.
"""

from __future__ import annotations

import pytest

from src.adapters.tavily import TavilyHit
from src.retrieval.rerank import rerank


def _make_hits(n: int = 3) -> list[TavilyHit]:
    return [
        TavilyHit(
            title=f"Hit {i}",
            url=f"https://example.com/{i}",
            content=f"content {i}",
            score=0.5,
        )
        for i in range(n)
    ]


@pytest.fixture
def _disable_rerank(settings_override):
    settings_override(GOOGLE_API_KEY="x", RERANK_ENABLED=False, RERANK_TOP_K=2)
    from src.config import get_settings

    get_settings.cache_clear()


@pytest.fixture
def _enable_rerank(settings_override):
    settings_override(GOOGLE_API_KEY="x", RERANK_ENABLED=True, RERANK_TOP_K=2)
    from src.config import get_settings

    get_settings.cache_clear()


async def test_rerank_passthrough_when_disabled(_disable_rerank) -> None:
    hits = _make_hits(5)
    out = await rerank("query", hits, top_k=2)
    assert len(out) == 2
    assert [h.title for h in out] == ["Hit 0", "Hit 1"]  # original order kept


async def test_rerank_empty_input_returns_empty(_disable_rerank) -> None:
    out = await rerank("query", [], top_k=5)
    assert out == []


async def test_rerank_falls_back_on_exception(_enable_rerank, monkeypatch) -> None:
    # Force the underlying sync rerank to blow up; we should get the
    # original order back, not raise.
    def boom(*args, **kwargs):
        raise RuntimeError("simulated flashrank failure")

    monkeypatch.setattr("src.retrieval.rerank._rerank_sync", boom)

    hits = _make_hits(4)
    out = await rerank("query", hits, top_k=2)
    assert len(out) == 2
    assert [h.title for h in out] == ["Hit 0", "Hit 1"]


async def test_rerank_uses_stubbed_scores(_enable_rerank, monkeypatch) -> None:
    # Stub the sync path to return hits in reverse order with a synthetic score.
    def fake(query, hits, top_k, model_name):
        rev = list(reversed(hits))[:top_k]
        return [
            TavilyHit(title=h.title, url=h.url, content=h.content, score=0.9 - i * 0.1)
            for i, h in enumerate(rev)
        ]

    monkeypatch.setattr("src.retrieval.rerank._rerank_sync", fake)

    hits = _make_hits(4)  # Hit 0..3
    out = await rerank("query", hits, top_k=2)
    assert [h.title for h in out] == ["Hit 3", "Hit 2"]
    assert out[0].score > out[1].score
