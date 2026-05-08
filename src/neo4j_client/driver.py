"""Async Neo4j driver with managed lifecycle.

Open once on app startup via `await client.connect()`. Close on shutdown
via `await client.close()`. Use `async with client.session() as session:`
for every query — sessions are scoped to a single transaction context.

Connection pool sized for moderate concurrency. Tune via `max_connection_pool_size`
once you've measured your workload.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from ..config import get_settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        settings = get_settings()
        self._driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            max_connection_pool_size=10,
            connection_acquisition_timeout=30,
        )
        await self._driver.verify_connectivity()
        logger.info("neo4j connected", extra={"uri": settings.NEO4J_URI})

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("neo4j connection closed")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._driver is None:
            raise RuntimeError("Neo4jClient.connect() not called — see lifespan in server.py")
        async with self._driver.session() as session:
            yield session
