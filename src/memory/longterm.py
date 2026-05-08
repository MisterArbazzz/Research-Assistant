"""SQLite + sqlite-vec long-term memory store.

Schema (auto-created on first call):

    CREATE TABLE user_facts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        fact TEXT NOT NULL,
        source_thread_id TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE VIRTUAL TABLE user_facts_vec USING vec0(
        embedding float[768]
    );

The two tables are joined by `rowid` — sqlite-vec's `vec0` virtual table
uses an internal rowid that we mirror as `user_facts.id`. Embedding lookup
returns `(rowid, distance)` which we then JOIN against user_facts to get
the raw fact text scoped to the requesting user.

Concurrency note: SQLite is single-writer. Fine here because writes are
infrequent (one extraction call per turn at most). If write contention
becomes a problem, swap to Postgres + pgvector — same API.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any

import sqlite_vec

from ..llm.embeddings import EMBED_DIM, embed_texts

logger = logging.getLogger(__name__)

LONGTERM_DB_PATH = Path("./data/longterm_memory.db")


def _connect() -> sqlite3.Connection:
    """Open a connection with sqlite-vec loaded. New connection per call —
    SQLite connections aren't safe to share across threads, and we run
    DB calls on a worker thread via asyncio.to_thread."""
    LONGTERM_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LONGTERM_DB_PATH))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_facts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            source_thread_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS user_facts_user_idx ON user_facts(user_id)")
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS user_facts_vec USING vec0(
            embedding float[{EMBED_DIM}]
        )
        """
    )
    conn.commit()


def _store_sync(user_id: str, fact: str, source_thread_id: str, embedding: list[float]) -> int:
    conn = _connect()
    try:
        _ensure_schema(conn)
        cur = conn.execute(
            "INSERT INTO user_facts(user_id, fact, source_thread_id) VALUES (?, ?, ?)",
            (user_id, fact, source_thread_id),
        )
        new_id = int(cur.lastrowid or 0)
        conn.execute(
            "INSERT INTO user_facts_vec(rowid, embedding) VALUES (?, ?)",
            (new_id, sqlite_vec.serialize_float32(embedding)),
        )
        conn.commit()
        return new_id
    finally:
        conn.close()


def _retrieve_sync(user_id: str, query_embedding: list[float], k: int) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        _ensure_schema(conn)
        # Two-stage: vec search to get candidate rowids, then JOIN against
        # user_facts to scope to this user_id.
        rows = conn.execute(
            """
            SELECT v.rowid, v.distance, f.fact, f.source_thread_id, f.created_at
            FROM user_facts_vec v
            JOIN user_facts f ON f.id = v.rowid
            WHERE v.embedding MATCH ?
                AND f.user_id = ?
                AND k = ?
            ORDER BY v.distance
            """,
            (sqlite_vec.serialize_float32(query_embedding), user_id, k * 4),
        ).fetchall()
        return [
            {"fact": r[2], "distance": float(r[1]), "source_thread_id": r[3], "created_at": r[4]}
            for r in rows[:k]
        ]
    finally:
        conn.close()


async def store_fact(user_id: str, fact: str, source_thread_id: str) -> int:
    """Embed `fact` and write it to the user's long-term store. Returns row id."""
    if not fact.strip():
        logger.warning("store_fact called with empty fact — skipping")
        return -1
    [embedding] = await embed_texts([fact])
    return await asyncio.to_thread(
        _store_sync, user_id, fact.strip(), source_thread_id, embedding
    )


def _list_sync(user_id: str, limit: int) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT id, fact, source_thread_id, created_at
            FROM user_facts
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [
            {
                "id": int(r[0]),
                "fact": r[1],
                "source_thread_id": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]
    finally:
        conn.close()


async def list_facts(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return all facts stored for `user_id`, newest first.

    No embedding lookup, no scoring — this is a simple table dump for UI
    inspection (e.g. the Streamlit Memory tab). For relevance-scored
    retrieval use `retrieve_relevant_facts` instead.
    """
    return await asyncio.to_thread(_list_sync, user_id, limit)


async def retrieve_relevant_facts(user_id: str, query: str, k: int = 3) -> list[str]:
    """Return up to `k` facts about this user most relevant to `query`.

    Empty list when the store has nothing for the user. Failures fall back
    to empty list with a warning — long-term memory is a quality lever,
    not a hard dependency.
    """
    if not query.strip():
        return []
    try:
        [query_embedding] = await embed_texts([query])
        rows = await asyncio.to_thread(_retrieve_sync, user_id, query_embedding, k)
        return [r["fact"] for r in rows]
    except Exception as exc:
        logger.warning(
            "long-term memory retrieval failed",
            extra={"user_id": user_id, "error": str(exc)},
        )
        return []
