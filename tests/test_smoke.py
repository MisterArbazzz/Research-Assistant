"""Boilerplate smoke test. Verifies the infrastructure imports cleanly.

Use cases add their own end-to-end smoke tests that run the compiled graph
with mocked LLM and Neo4j.
"""

from __future__ import annotations


def test_imports() -> None:
    """Every module in src/ imports without errors."""
    from src import config, deps, logging_config, observability, server, sse_bus  # noqa: F401
    from src.adapters import _mock_helpers  # noqa: F401
    from src.llm import client, embeddings  # noqa: F401
    from src.neo4j_client import audit, driver  # noqa: F401
    from src.routes import health, runs  # noqa: F401


def test_settings_load(settings_override) -> None:
    """Settings load with required GOOGLE_API_KEY."""
    settings_override(GOOGLE_API_KEY="test-key")
    from src.config import Settings

    s = Settings()  # type: ignore[call-arg]
    assert s.GOOGLE_API_KEY == "test-key"
    assert s.RESEARCH_BACKEND == "tavily"  # default
    assert s.MAX_RESEARCH_ATTEMPTS == 3  # default
