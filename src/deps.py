"""FastAPI dependency providers.

Resources live on `app.state` and are managed by the lifespan in `server.py`.
Routes pull them via `Depends(get_<x>)` so every handler gets the same
instance without global singletons.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request

from .config import Settings, get_settings


def get_graph(request: Request) -> Any:
    """Returns the compiled LangGraph."""
    return request.app.state.graph


def get_checkpointer(request: Request) -> Any:
    return request.app.state.checkpointer


__all__ = ["Settings", "get_checkpointer", "get_graph", "get_settings"]
