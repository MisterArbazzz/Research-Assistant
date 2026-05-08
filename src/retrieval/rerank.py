"""Cross-encoder reranker on top of Tavily search results.

Tavily's own scoring ranks hits by retrieval relevance, but a cross-encoder
re-scores each (query, hit_text) pair with a model trained specifically for
relevance judgment. This typically pushes the most directly-on-topic hits to
the top, which the LLM digest then cites first.

Implementation: `flashrank` (ONNX runtime, ~30MB model, ~60MB total install).
Much lighter than sentence-transformers + torch on Windows. The model loads
lazily on first call (one-time ~3-second cold start) and is cached for the
process lifetime — fine because nodes are short-lived and re-using the same
ranker across turns is exactly what we want.

Setting `RERANK_ENABLED=false` makes this a no-op pass-through that simply
trims to top_k by Tavily's own score order.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

from ..adapters.tavily import TavilyHit
from ..config import get_settings

logger = logging.getLogger(__name__)

# Lazy-loaded singleton. flashrank.Ranker is heavy to construct (ONNX session
# + tokenizer); we want exactly one per process. Locked init so two concurrent
# nodes don't race on the first call.
_ranker: Any = None
_ranker_lock = threading.Lock()
_RANKER_CACHE_DIR = Path("./data/.flashrank_cache")


def _get_ranker(model_name: str) -> Any:
    """Lazy singleton for the flashrank Ranker. Thread-safe."""
    global _ranker
    if _ranker is not None:
        return _ranker
    with _ranker_lock:
        if _ranker is not None:
            return _ranker
        from flashrank import Ranker

        _RANKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "loading flashrank model (one-time)",
            extra={"model": model_name, "cache_dir": str(_RANKER_CACHE_DIR)},
        )
        _ranker = Ranker(
            model_name=model_name,
            cache_dir=str(_RANKER_CACHE_DIR),
            log_level="WARNING",
        )
    return _ranker


def _rerank_sync(
    query: str, hits: list[TavilyHit], top_k: int, model_name: str
) -> list[TavilyHit]:
    """Blocking rerank — call inside asyncio.to_thread."""
    from flashrank import RerankRequest

    ranker = _get_ranker(model_name)
    passages = [
        {"id": i, "text": f"{h.title}\n{h.content}", "meta": {"url": h.url}}
        for i, h in enumerate(hits)
    ]
    request = RerankRequest(query=query, passages=passages)
    scored = ranker.rerank(request)
    # Each scored item: {"id": int, "score": float, "text": str, "meta": {...}}
    out: list[TavilyHit] = []
    for item in scored[:top_k]:
        original = hits[int(item["id"])]
        # Replace the Tavily-internal score with the cross-encoder score so
        # downstream telemetry sees the rerank ordering, not the Tavily one.
        out.append(
            TavilyHit(
                title=original.title,
                url=original.url,
                content=original.content,
                score=float(item["score"]),
            )
        )
    return out


async def rerank(
    query: str,
    hits: list[TavilyHit],
    top_k: int | None = None,
) -> list[TavilyHit]:
    """Rerank `hits` by cross-encoder relevance to `query`. Returns top_k.

    No-op pass-through when RERANK_ENABLED is false (still trims to top_k).
    Failures fall back to the original Tavily ordering — degraded retrieval
    is a quality issue, not a fatal one.
    """
    settings = get_settings()
    top_k = top_k or settings.RERANK_TOP_K
    if not hits:
        return []
    if not settings.RERANK_ENABLED:
        return hits[:top_k]

    try:
        return await asyncio.to_thread(
            _rerank_sync, query, hits, top_k, settings.RERANK_MODEL
        )
    except Exception as exc:
        logger.warning(
            "rerank failed — falling back to Tavily ordering",
            extra={"error": str(exc)},
        )
        return hits[:top_k]
