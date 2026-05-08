"""FastAPI dependency providers.

Resources live on `app.state` and are managed by the lifespan in `server.py`.
Routes pull them via `Depends(get_<x>)` so every handler gets the same
instance without global singletons.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request

from .config import Settings, get_settings
from .neo4j_client.driver import Neo4jClient


def get_neo4j(request: Request) -> Neo4jClient:
    return request.app.state.neo4j  # type: ignore[no-any-return]


def get_graph(request: Request) -> Any:
    """Returns the compiled LangGraph or None if not yet wired for this use case."""
    return request.app.state.graph


def get_checkpointer(request: Request) -> Any:
    return request.app.state.checkpointer


__all__ = ["get_neo4j", "get_graph", "get_checkpointer", "get_settings", "Settings"]
