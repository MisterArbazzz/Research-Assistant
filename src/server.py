"""FastAPI application factory + lifespan.

Lifespan opens the AsyncSqliteSaver checkpointer and the compiled LangGraph.
Everything is stashed on app.state and provided to routes via `deps.get_*`.
Observability comes from LangSmith (cloud) + OpenTelemetry; no Neo4j audit
graph in this build (the in-memory `state.audit_log` plus LangSmith traces
already cover the observability story for our use case).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .graph.builder import build_graph
from .logging_config import configure_logging
from .observability import configure_langsmith, configure_tracer
from .routes import health

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    configure_tracer()
    configure_langsmith()

    async with AsyncExitStack() as stack:
        checkpointer_cm = AsyncSqliteSaver.from_conn_string("./data/checkpoints.db")
        checkpointer = await stack.enter_async_context(checkpointer_cm)

        app.state.checkpointer = checkpointer
        app.state.graph = await build_graph(checkpointer)

        logger.info("startup complete")
        try:
            yield
        finally:
            logger.info("shutdown initiated")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Assistant",
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
        # trace id the operator can grep for.
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

    return app


app = create_app()
