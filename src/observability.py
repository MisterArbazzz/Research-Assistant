"""OpenTelemetry tracer setup + LangSmith conditional initialization.

Local dev: spans go to ConsoleSpanExporter (stdout). Production: set
OTEL_EXPORTER_OTLP_ENDPOINT and spans flow to a real OTLP collector
(Tempo, Jaeger, Cloud Trace, etc.) with no code changes.

LangSmith integration uses env-var configuration — LangChain reads
LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY at runtime so we set them here
when LANGSMITH_API_KEY is present. This is the documented LangSmith
integration path; writing to os.environ here is intentional and the
single exception to the "no os.environ outside config.py" rule.
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from .config import get_settings


def configure_tracer(service_name: str = "langgraph-agent") -> None:
    settings = get_settings()
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT))
        )
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


def configure_langsmith() -> None:
    settings = get_settings()
    if not settings.LANGSMITH_API_KEY:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    if settings.LANGSMITH_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
