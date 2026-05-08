"""Shared pytest fixtures.

The boilerplate ships with minimal fixtures; use cases extend with mock LLM
clients, in-memory Neo4j alternatives, fixture states, etc.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def settings_override(monkeypatch: pytest.MonkeyPatch):
    """Override settings env vars per test. Usage:

        def test_x(settings_override):
            settings_override(SLACK_MODE="live", MAX_AGENT_ITERATIONS=5)
            # get_settings.cache_clear() if you've already called get_settings() in another test
    """

    def _override(**kwargs: object) -> None:
        for k, v in kwargs.items():
            monkeypatch.setenv(k, str(v))

    return _override
