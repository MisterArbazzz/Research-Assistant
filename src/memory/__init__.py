"""Long-term memory across threads.

Short-term memory is the LangGraph checkpointer (per-thread). Long-term
memory is a small per-user fact store (across threads): preferences,
recurring interests, durable identifiers — anything the user said in
session A that should colour the agent's behavior in session B.

Storage: SQLite + sqlite-vec. Vector dimension matches the embeddings
client (768). Postgres + pgvector is a drop-in production swap; the API
shape (`store_fact`, `retrieve_relevant_facts`) is intentionally narrow
so the backend can be replaced without touching call sites.
"""

from __future__ import annotations

from .longterm import list_facts, retrieve_relevant_facts, store_fact

__all__ = ["list_facts", "retrieve_relevant_facts", "store_fact"]
