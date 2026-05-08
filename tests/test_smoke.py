"""Smoke test — every module in src/ imports cleanly + settings load."""

from __future__ import annotations


def test_imports() -> None:
    """Every module in src/ imports without errors."""
    from src import config, deps, logging_config, observability, server, sse_bus  # noqa: F401
    from src.adapters import tavily  # noqa: F401
    from src.graph import builder, prompts, routing, state  # noqa: F401
    from src.graph.nodes import (  # noqa: F401
        clarity_agent,
        interrupt_node,
        memory_writer,
        research_agent,
        synthesis_agent,
        validator_agent,
    )
    from src.llm import client, embeddings  # noqa: F401
    from src.memory import longterm  # noqa: F401
    from src.retrieval import query_rewriting, rerank  # noqa: F401
    from src.routes import health  # noqa: F401


# NOTE: streamlit module imports have side effects that conflict with pytest's
# stdout capture (writes-to-closed-file on test cleanup). The streamlit_app
# package is import-checked separately via `uv run python -c` in CI.


def test_settings_load(settings_override) -> None:
    """Settings load with required GOOGLE_API_KEY."""
    settings_override(GOOGLE_API_KEY="test-key")
    from src.config import Settings

    s = Settings()
    assert s.GOOGLE_API_KEY == "test-key"
    assert s.RESEARCH_BACKEND == "tavily"  # default
    assert s.MAX_RESEARCH_ATTEMPTS == 3  # default
