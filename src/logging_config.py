"""Logging configuration via dictConfig.

Two formatters: `json` (production, parseable by log aggregators) and `pretty`
(developer-friendly with colors and short timestamps). Switched via LOG_FORMAT.
"""

from __future__ import annotations

import logging.config
from typing import Any

from .config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            },
            "pretty": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": settings.LOG_FORMAT,
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["default"],
        },
        "loggers": {
            "uvicorn": {"level": "INFO"},
            "uvicorn.access": {"level": "WARNING"},
            "neo4j": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
        },
    }
    logging.config.dictConfig(config)
