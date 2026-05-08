"""FastAPI application factory + lifespan.

Lifespan opens the Neo4j async driver, the AsyncSqliteSaver checkpointer,
and (when wired) the compiled LangGraph. All resources are stashed on
app.state and provided to routes via `deps.get_*`. Shutdown closes them
in reverse order.

The graph itself is wired by the use-case-specific `src/graph/builder.py`
factory; the boilerplate ships with `app.state.graph = None` and the
runs route returns 503 until the use case fills it in.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .logging_config import configure_logging
from .neo4j_client.driver import Neo4jClient
from .observability import configure_langsmith, configure_tracer
from .routes import health, runs

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    configure_tracer()
    configure_langsmith()

    async with AsyncExitStack() as stack:
        # Neo4j async driver
        neo4j_client = Neo4jClient()
        await neo4j_client.connect()
        stack.push_async_callback(neo4j_client.close)

        # SQLite checkpointer (swap to AsyncPostgresSaver for prod)
        checkpointer_cm = AsyncSqliteSaver.from_conn_string("./data/checkpoints.db")
        checkpointer = await stack.enter_async_context(checkpointer_cm)

        app.state.neo4j = neo4j_client
        app.state.checkpointer = checkpointer
        app.state.graph = None  # Use case wires this in: from .graph.builder import build_graph

        logger.info("startup complete")
        try:
            yield
        finally:
            logger.info("shutdown initiated")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LangGraph Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3002"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    FastAPIInstrumentor.instrument_app(app)

    @app.exception_handler(Exception)
    async def fallback_handler(request: Request, exc: Exception) -> JSONResponse:
        # Top-level safety net — log the stack, return a structured 500 with a
        # trace id the operator can grep for. Use-case routes catch their own
        # business errors and return 4xx; this only fires for genuine bugs.
        trace_id = str(uuid.uuid4())
        logger.exception(
            "unhandled exception",
            extra={"trace_id": trace_id, "path": request.url.path},
        )
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "trace_id": trace_id},
        )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(runs.router, prefix="/api/v1")

    return app


app = create_app()
