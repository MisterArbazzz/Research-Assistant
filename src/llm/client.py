"""Gemini chat-model factory + tenacity retry wrapper + cost estimator.

Factory returns a fresh `ChatGoogleGenerativeAI` per call — cheap, lets each
node bind its own structured output / tool config without leaking state across
nodes. Use `.with_structured_output(SomeModel)` when the response has structure.

Retry wraps the call site (not the constructor) so the same client can be
reused if the caller chooses. Tenacity catches three classes of transient
Gemini failures (rate-limit, timeout, upstream unavailable) with exponential
backoff capped at LLM_RETRY_MAX_ATTEMPTS. Non-transient errors (auth, invalid
input, content filter) bubble up immediately — retry would only mask them.

Cost estimator reads `usage_metadata` returned by langchain-google-genai and
multiplies by published per-token pricing. Numbers are a snapshot — verify in
production before relying on them for billing alarms. The single enforcement
site for the per-run cost ceiling reads `settings.COST_CEILING_PER_RUN_USD`
and uses this estimator to decide.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, TypeVar

from google.api_core.exceptions import (
    DeadlineExceeded,
    ResourceExhausted,
    ServiceUnavailable,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_settings

# The newer google-genai SDK (used by recent langchain-google-genai) raises
# google.genai.errors.ClientError / ServerError with HTTP status codes on
# the exception. Older google-api-core types are still possible depending on
# which code path raised. Import defensively so absence doesn't break.
try:
    from google.genai.errors import APIError as _GenAIAPIError
except ImportError:  # pragma: no cover
    _GenAIAPIError = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# HTTP status codes that indicate a transient failure worth retrying.
# 408 timeout, 429 rate limit, 500/502/503/504 upstream issues.
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# Hard wall-clock cap on a single LLM call. Gemini occasionally hangs an
# open connection rather than returning a 5xx; without a timeout we'd block
# a node forever. Tenacity will catch the asyncio.TimeoutError and retry.
LLM_CALL_TIMEOUT_SECONDS = 60.0


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """True for transient Gemini failures across both old and new SDKs.

    Old google-api-core exceptions carry the semantics in the class itself.
    New google-genai exceptions carry an HTTP status code as an attribute.
    asyncio.TimeoutError fires when the call exceeds LLM_CALL_TIMEOUT_SECONDS
    (slow / stuck Gemini connection) — retry rather than abort the run.
    Anything else (auth, schema, safety filter) is a real bug that should
    fail fast — retrying just hides it.
    """
    if isinstance(exc, (ResourceExhausted, DeadlineExceeded, ServiceUnavailable)):
        return True
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if _GenAIAPIError is not None and isinstance(exc, _GenAIAPIError):
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        return code in _RETRYABLE_STATUS_CODES
    return False

# Public Gemini API pricing (USD per 1M tokens). Snapshot — verify quarterly
# against ai.google.dev/pricing. Pro pricing assumes ≤200k context window.
# Last verified: 2026-05-08. The earlier 2024-era numbers undercounted by 4-8x;
# unknown models return $0 from estimate_cost (with a warning) so a "$0 across
# the run" audit row is the signal to update this table.
PRICING_USD_PER_M_TOKENS: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
}


def get_chat_model(
    role: Literal["primary", "qa"] = "primary",
    **overrides: Any,
) -> ChatGoogleGenerativeAI:
    """Construct a Gemini chat model. `role` selects between MODEL_PRIMARY and MODEL_QA.

    `temperature=0` by default for reproducibility. Override via kwargs when
    explicit creativity is needed (e.g. ideator nodes that produce diverse
    candidates with `temperature=0.7`).
    """
    settings = get_settings()
    model = settings.MODEL_PRIMARY if role == "primary" else settings.MODEL_QA
    defaults: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_retries": 0,  # we control retries via tenacity below
        "google_api_key": settings.GOOGLE_API_KEY,
    }
    defaults.update(overrides)
    return ChatGoogleGenerativeAI(**defaults)


async def ainvoke_with_retry(llm: Any, *args: Any, **kwargs: Any) -> Any:
    """Wrap llm.ainvoke with tenacity retry on transient Gemini failures.

    Wraps the call site, not the constructor — the retry state is per-call,
    so a single client instance can be reused across nodes without leaking
    retry counters.
    """
    settings = get_settings()

    @retry(
        retry=retry_if_exception(_is_retryable_llm_error),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(settings.LLM_RETRY_MAX_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call() -> Any:
        return await asyncio.wait_for(
            llm.ainvoke(*args, **kwargs), timeout=LLM_CALL_TIMEOUT_SECONDS
        )

    return await _call()


T = TypeVar("T", bound=BaseModel)


async def ainvoke_structured(
    llm: Any,
    messages: Any,
    model: type[T],
) -> tuple[T, dict[str, Any] | None]:
    """Invoke an LLM with structured output and return `(parsed, usage_metadata)`.

    Wraps three patterns every node was duplicating:
      1. `with_structured_output(Model, include_raw=True)` to get both the
         parsed model and the raw AIMessage (for usage_metadata).
      2. Tenacity retry on transient errors via `ainvoke_with_retry`.
      3. Defensive parse-failure handling — when validation fails, langchain
         returns `parsed=None` rather than raising, which causes obscure
         `'NoneType' has no attribute X` errors at the call site. We surface
         a clear error with the raw text logged for diagnosis.

    Raises ValueError if the LLM output couldn't be parsed into `model`.
    """
    structured = llm.with_structured_output(model, include_raw=True)
    result = await ainvoke_with_retry(structured, messages)

    if not isinstance(result, dict):
        # Some integrations return the parsed model directly even with include_raw.
        return result, None

    parsed = result.get("parsed")
    raw = result.get("raw")
    if parsed is None:
        parse_err = result.get("parsing_error")
        raw_text = str(getattr(raw, "content", ""))[:500] if raw is not None else ""
        logger.error(
            "structured output failed validation",
            extra={
                "model": model.__name__,
                "parsing_error": str(parse_err),
                "raw_preview": raw_text,
            },
        )
        raise ValueError(
            f"could not parse LLM output into {model.__name__}: {parse_err}"
        )

    usage = getattr(raw, "usage_metadata", None) if raw else None
    return parsed, usage


def estimate_cost(model: str, usage: dict[str, Any] | None) -> float:
    """Estimate USD cost from a Gemini `usage_metadata` dict.

    Returns 0.0 when usage is missing or the model isn't in the price table —
    safer than guessing. Audit records the value so an unknown-model gap will
    show up as $0 across a run, which is a visible signal to update pricing.
    """
    if not usage or model not in PRICING_USD_PER_M_TOKENS:
        if usage and model not in PRICING_USD_PER_M_TOKENS:
            logger.warning(
                "no pricing for model — cost reported as 0",
                extra={"model": model},
            )
        return 0.0
    price = PRICING_USD_PER_M_TOKENS[model]
    # Different langchain versions surface token counts under slightly different
    # keys; accept both and default missing to 0.
    in_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    out_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return (in_tokens * price["input"] + out_tokens * price["output"]) / 1_000_000
