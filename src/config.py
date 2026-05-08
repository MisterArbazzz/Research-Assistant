"""Centralized application settings.

The single place that reads environment variables. Use `get_settings()` everywhere
else. The `@lru_cache` decorator means Settings is constructed once per process —
safe because Pydantic Settings reads env vars at construction time and FastAPI
processes don't fork mid-request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    GOOGLE_API_KEY: str = Field(..., description="Gemini API key")
    MODEL_PRIMARY: str = "gemini-2.5-flash"
    MODEL_QA: str = "gemini-2.5-pro"
    LLM_RETRY_MAX_ATTEMPTS: int = 4

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "devpassword"

    # Observability — all optional
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str | None = None
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: Literal["pretty", "json"] = "json"

    # Research backend — "tavily" hits live Tavily search; "mock" reads from data/mock_companies.py.
    # Tavily is the default; mock is a hermetic fallback for tests / offline dev.
    RESEARCH_BACKEND: Literal["mock", "tavily"] = "tavily"
    TAVILY_API_KEY: str | None = None
    TAVILY_MAX_RESULTS: int = 5

    # Caps — defensive boundaries enforced at single sites in routing.py.
    # MAX_RESEARCH_ATTEMPTS caps the validator → research retry loop.
    # Default 3 per the problem statement (validator can ask for 2 retries).
    MAX_RESEARCH_ATTEMPTS: int = 3
    MAX_TOOL_CALLS_PER_NODE: int = 5
    COST_CEILING_PER_RUN_USD: float = 0.10


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Use this everywhere; never read os.environ directly."""
    return Settings()  # type: ignore[call-arg]  # pydantic-settings reads env at runtime
