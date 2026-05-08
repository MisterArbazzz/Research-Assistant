"""Unit tests for the deterministic eval metrics.

LLM-as-judge metrics (faithfulness, answer_relevance, trajectory) are not
unit-tested here — they're exercised live by `run_eval.py`. Stubbing them
into determinism would just test the stub.
"""

from __future__ import annotations

from src.adapters.tavily import TavilyHit
from tests.eval.metrics import context_precision, context_recall


def _hit(title: str, content: str) -> TavilyHit:
    return TavilyHit(title=title, url="https://x", content=content, score=0.5)


def test_recall_finds_company_in_title() -> None:
    hits = [_hit("Apple Inc earnings beat", "Q3 results...")]
    assert context_recall(hits, "Apple Inc.") == 1.0


def test_recall_finds_company_in_content() -> None:
    hits = [_hit("Tech roundup", "Apple announced new products today")]
    assert context_recall(hits, "Apple Inc.") == 1.0


def test_recall_misses_when_company_absent() -> None:
    hits = [_hit("Tesla news", "Cybertruck deliveries up")]
    assert context_recall(hits, "Apple Inc.") == 0.0


def test_recall_unknown_company_scores_one() -> None:
    # Unknown-company cases shouldn't be punished on recall
    assert context_recall([], None) == 1.0
    assert context_recall([_hit("foo", "bar")], None) == 1.0


def test_recall_zero_hits_zero_recall() -> None:
    assert context_recall([], "Apple Inc.") == 0.0


def test_precision_all_relevant() -> None:
    hits = [
        _hit("Tesla Q3", "Tesla revenue up"),
        _hit("Tesla CEO", "Elon Musk comp"),
    ]
    assert context_precision(hits, "Tesla") == 1.0


def test_precision_partially_relevant() -> None:
    hits = [
        _hit("Tesla Q3", "Tesla revenue up"),
        _hit("Generic", "no company name"),
    ]
    assert context_precision(hits, "Tesla") == 0.5


def test_precision_unknown_returns_one() -> None:
    assert context_precision([_hit("a", "b")], None) == 1.0
