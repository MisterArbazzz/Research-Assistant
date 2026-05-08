"""Tests for the long-term memory store.

Stubs the embeddings client so these run hermetically — no Gemini calls,
no network. The vector math (sqlite-vec MATCH on float32 embeddings) is
exercised end-to-end against a real temp SQLite database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory import longterm


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the memory module at a per-test temp DB so tests don't collide."""
    db = tmp_path / "test_longterm.db"
    monkeypatch.setattr(longterm, "LONGTERM_DB_PATH", db)


@pytest.fixture
def _fake_embeddings(monkeypatch: pytest.MonkeyPatch):
    """Stub embed_texts to deterministic 768-dim vectors keyed off a tiny
    prefix of the input string. Enough for nearest-neighbor ordering tests
    without calling Gemini."""

    async def fake_embed(texts: list[str], batch_size: int = 100) -> list[list[float]]:
        out = []
        for t in texts:
            # Lightweight hash-to-vector: same prefix → same direction
            seed = sum(ord(c) for c in t[:10])
            base = [(seed % 17) / 17.0] * 768
            # Tiny perturbation so vectors aren't all identical
            base[seed % 768] += 0.5
            out.append(base)
        return out

    monkeypatch.setattr(longterm, "embed_texts", fake_embed)


async def test_store_then_retrieve_roundtrip(_fake_embeddings) -> None:
    rid = await longterm.store_fact(
        "alice", "User prefers technical detail in answers.", source_thread_id="t1"
    )
    assert rid > 0
    facts = await longterm.retrieve_relevant_facts(
        "alice", "How does the user like answers formatted?", k=3
    )
    assert len(facts) == 1
    assert "technical detail" in facts[0]


async def test_retrieve_scopes_to_user_id(_fake_embeddings) -> None:
    await longterm.store_fact("alice", "Alice follows EV stocks", source_thread_id="t1")
    await longterm.store_fact("bob", "Bob is a portfolio manager", source_thread_id="t2")

    alice_facts = await longterm.retrieve_relevant_facts("alice", "stocks", k=5)
    bob_facts = await longterm.retrieve_relevant_facts("bob", "stocks", k=5)

    assert any("Alice" in f for f in alice_facts)
    assert not any("Bob" in f for f in alice_facts)
    assert any("Bob" in f for f in bob_facts)


async def test_retrieve_returns_empty_for_unknown_user(_fake_embeddings) -> None:
    await longterm.store_fact("alice", "alpha", "t")
    facts = await longterm.retrieve_relevant_facts("eve", "alpha", k=3)
    assert facts == []


async def test_store_skips_empty_fact(_fake_embeddings) -> None:
    rid = await longterm.store_fact("alice", "   ", source_thread_id="t")
    assert rid == -1


async def test_retrieve_ignores_empty_query(_fake_embeddings) -> None:
    facts = await longterm.retrieve_relevant_facts("alice", "", k=3)
    assert facts == []


async def test_retrieve_respects_k_limit(_fake_embeddings) -> None:
    for i in range(5):
        await longterm.store_fact("alice", f"Fact number {i}", source_thread_id="t")
    facts = await longterm.retrieve_relevant_facts("alice", "Fact", k=2)
    assert len(facts) == 2


async def test_list_facts_empty_for_new_user(_fake_embeddings) -> None:
    facts = await longterm.list_facts("never_seen_user")
    assert facts == []


async def test_list_facts_returns_newest_first(_fake_embeddings) -> None:
    for i in range(3):
        await longterm.store_fact(
            "carol", f"fact {i}", source_thread_id=f"thread-{i}"
        )
    rows = await longterm.list_facts("carol")
    assert len(rows) == 3
    # Newest-first ordering: id descends, last-stored shows up first.
    assert rows[0]["fact"] == "fact 2"
    assert rows[2]["fact"] == "fact 0"
    # Schema check
    assert {"id", "fact", "source_thread_id", "created_at"} <= set(rows[0].keys())


async def test_list_facts_scopes_to_user(_fake_embeddings) -> None:
    await longterm.store_fact("alice", "alice's fact", source_thread_id="t")
    await longterm.store_fact("bob", "bob's fact", source_thread_id="t")
    alice_only = await longterm.list_facts("alice")
    assert len(alice_only) == 1
    assert alice_only[0]["fact"] == "alice's fact"


async def test_list_facts_respects_limit(_fake_embeddings) -> None:
    for i in range(5):
        await longterm.store_fact("dan", f"fact {i}", source_thread_id="t")
    rows = await longterm.list_facts("dan", limit=2)
    assert len(rows) == 2
