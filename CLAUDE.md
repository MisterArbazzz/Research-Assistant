# CLAUDE.md — Repo conventions for LangGraph agents

Operating contract for any AI assistant editing this repo. Read once at session start; every prompt assumes these conventions hold. Don't restate them.

## Stack

- Python 3.11+, async-first throughout
- Pydantic v2 for every boundary (state, tool I/O, API schemas, config)
- LangGraph 0.2.x for graph orchestration
- FastAPI + sse-starlette for the HTTP/streaming bridge
- Neo4j 5.x (async driver) for KG + vector index storage
- Gemini via `langchain-google-genai`
- `AsyncSqliteSaver` checkpointer (`./data/checkpoints.db`); swap to `AsyncPostgresSaver` for prod

## Discipline (non-negotiable)

- `from __future__ import annotations` at the top of every module
- `mypy --strict` passes; no bare `# type: ignore` (each ignore needs a one-line justification)
- No `os.environ.get(...)` outside `src/config.py`. Settings flow through `get_settings()` factory.
- No global mutable singletons. Resources injected via FastAPI `Depends` or constructor; lifespan-managed via `@asynccontextmanager`.
- Every node opens with `with tracer.start_as_current_span("node.<name>")` and ends with `await audit.record_step(...)`.
- LLM calls go through `await ainvoke_with_retry(llm, ...)` from `src/llm/client.py` (tenacity-wrapped, exponential backoff on `ResourceExhausted`). Use `with_structured_output(SomeModel)` whenever the response has structure; bare `.ainvoke` only for free-form text.
- Every cap (max iterations, max tool calls, retries, cost ceiling) lives at exactly one enforcement site. Constants in `src/config.py`; enforcement in routing or guard functions.
- Failures inside nodes update state and return; never raise to the graph runtime unless the failure is fatal. `tenacity.RetryError` is caught and reflected in `state.audit_log`.

## State shape pattern

Use Pydantic v2 `BaseModel`. Always include:

- `run_id: str` — uuid4, set at run start
- `audit_log: Annotated[list[dict], add]` — accumulator for per-node breadcrumbs
- `messages: Annotated[list[BaseMessage], add_messages]` — LangGraph message channel

Use-case-specific fields go between. Nested types are also Pydantic models — never plain dicts.

## Routing pattern

`route_from_supervisor(state) -> str` — pure function, returns next node name. Branches ordered by which state fields are populated. **Never use an LLM for routing.**

`route_after_<gate>(state) -> str` — pure function, returns next node name. Iteration caps enforced here, reading from a `settings.MAX_<NAME>_ITERATIONS` constant (e.g. `MAX_QA_ITERATIONS` for the Greenlight QA loop).

## File-naming

- `src/graph/nodes/<name>.py` — one node per file, function `<name>_node`
- `src/adapters/<service>.py` — one adapter per external service
- `src/graph/tools.py` — all `@tool`-decorated async functions
- `src/graph/routing.py` — pure routing functions, no LLM, no I/O
- `src/graph/state.py` — Pydantic models for state + supporting types
- `src/graph/builder.py` — `build_graph(checkpointer) -> CompiledStateGraph`
- `src/graph/prompts.py` — pure functions building `SystemMessage` / `HumanMessage`

## External integrations — port-adapter pattern

Each integration exposes one async function in `src/adapters/<service>.py`. Live mode hits the real API; mock mode calls `await emit_mock(neo4j, run_id, "<service>", payload)` from `src/adapters/_mock_helpers.py`, which:

1. Writes a `Mock` node to Neo4j tied to the current `Run` (durable audit trail).
2. Emits an SSE event via `sse_bus.publish(run_id, "<service>.delivered", payload)` so the UI renders the side-effect.

Mode selected via `settings.<X>_MODE = "live" | "mock"`. Default is `mock`. **Same return shape either way** — the caller never knows which backend ran.

## Logging

Module-level: `logger = logging.getLogger(__name__)`. Pass structured context via `extra={"run_id": state.run_id, "node": "<name>"}`, never f-string interpolation. The dictConfig in `src/logging_config.py` JSON-formats records when `LOG_FORMAT=json`, pretty-prints when `pretty`.

## Observability

`tracer = trace.get_tracer(__name__)` at module top. One span per node. Set attributes: `node_name`, `run_id`, `latency_ms`; optionally `prompt_tokens`, `completion_tokens`, `model`. On failure: `span.record_exception(e); span.set_status(Status(StatusCode.ERROR))`.

LangSmith: env-flag activated. Set `LANGSMITH_API_KEY` and traces flow automatically (LangChain global config). Disabled by default.

## Testing

- `pytest` + `pytest-asyncio`
- Mock LLM and Neo4j by default; live calls only behind `@pytest.mark.integration`
- Smoke test runs the full graph with mocked dependencies in <5s
- Routing logic tested with hand-built state instances; never run the LLM in unit tests
- Mock-mode adapters tested by asserting they call `emit_mock` with the right payload

## Forbidden patterns

- Bare `except Exception:` outside the top-level FastAPI handler
- `print(...)` in source — use the logger
- Hardcoded API keys, paths, or magic numbers (constants live in `src/config.py`)
- `requests` / `urllib3` / sync HTTP clients — use `httpx.AsyncClient`
- Synchronous Neo4j driver
- f-string interpolation into Cypher queries — always parametrize
- Commented-out code blocks — delete or move to a feature branch
- TODOs without a tracking link

## Definition of done

For any new code:

- `mypy --strict src/` passes
- `ruff check src/` passes
- `pytest -m "not integration"` green in <5s
- The new code runs end-to-end via the demo entry point
- Docstrings on public functions explain *why*, not *what*

## Working with placeholders

Files in `src/graph/` are placeholders with template docstrings showing the canonical shape. When implementing a use case:

- Replace each placeholder with the use-case-specific implementation
- Keep the conventions above
- Don't introduce frameworks or patterns not already in this repo without flagging the reason in a commit message

## Hard-won patterns (lessons from production builds)

These patterns exist because earlier builds hit each of them as a real bug. Ignore them and you'll re-discover the bug.

### LLM-output Pydantic schemas don't enforce length

Drop `max_length` from any field a structured-output LLM call will fill. Gemini does NOT reliably honor schema-level length caps; you'll get `OUTPUT_PARSING_FAILURE` whenever the model writes a few extra characters. Validate **structure** (types, required fields, value ranges, min_length≥1 to reject empty strings) — guide length via the prompt. If you absolutely need a length-capped persisted field, do the truncation explicitly when building the persistent type, not via Pydantic validation on the LLM-facing schema.

### Always handle structured-output parse failures

`llm.with_structured_output(Model, include_raw=True)` returns a dict-like:

```python
{"raw": AIMessage, "parsed": Model | None, "parsing_error": Exception | None}
```

When validation fails, `parsed` is `None` and `parsing_error` carries the details. Reading `result["parsed"].field` blows up with an obscure `NoneType` attribute error. **Use `ainvoke_structured()` from `src/llm/client.py`** — it handles the retry, parses, checks for None, logs the raw text, and raises `ValueError` with a clear message.

### Return sentinels, not None, when a lookup misses

A node that loads a resource (`brand`, `voice`, `config`) and returns `None` will trip the supervisor's `is None` check, which then routes back to the same node forever — graph hits its recursion limit, run dies. Instead, return an empty/stub instance and record `loaded: False` in the audit log. Routing advances; downstream nodes can detect "stub" via the flag if they need to alter behavior.

### LLM call retry needs status-code predicates, not class predicates

`google-genai` (the new SDK) and `google-api-core` (the old one) raise different exception classes for the same HTTP status. Build retry around a predicate that checks **status codes** (408, 429, 500, 502, 503, 504) on whichever exception class fires, plus `asyncio.TimeoutError` from your hard timeout. See `_is_retryable_llm_error` in `src/llm/client.py`. A class-based `retry_if_exception_type` will silently miss half the transient failures.

### Hard wall-clock timeout on every LLM call

`asyncio.wait_for(llm.ainvoke(...), timeout=60)`. Gemini occasionally hangs an open connection rather than returning 5xx — without a timeout the node blocks until the SSE client gives up. Tenacity catches the `TimeoutError` and retries.

### Throttle fan-out nodes with a Semaphore

A node that does `asyncio.gather(*[llm_tool(c) for c in many])` will swamp rate limits and trigger Gemini's "high demand" 503s — sometimes one call hangs forever. Wrap with `asyncio.Semaphore(N)` (start with N=2, raise if your quota allows). The semaphore lives **inside the node**, not module-level — preserves the "no global mutable singletons" rule.

### Cypher seed scripts: re-MATCH by id between statements

The `;`-splitter pattern in `src/neo4j_client/schema.py` runs each statement in its own transaction. Cypher variables (`MERGE (a:Foo {...})`) **don't carry to the next statement**. Relationships must re-MATCH the endpoints by their stable id, not by the prior statement's variable:

```cypher
// Wrong: `acme` and `brand` are undefined here.
MERGE (acme)-[:HAS_BRAND]->(brand);

// Right: re-match by id.
MATCH (acme:Client {id: "acme"})
MATCH (brand:Brand   {id: "acme-brand"})
MERGE (acme)-[:HAS_BRAND]->(brand);
```

The split-on-`;` is also quote-aware in `_split_statements` — strings containing `;` (e.g. `"never sell; always inform"`) survive intact.

### Pricing constants are a snapshot, not a constant

`PRICING_USD_PER_M_TOKENS` in `src/llm/client.py` is a hardcoded snapshot of provider pricing. Verify quarterly against the provider's public price list; the `Last verified:` line in the docstring is the source of truth. Unknown models return `$0.0` from `estimate_cost` (with a warning log) — that's intentional, but a $0 audit row across an entire run is the signal to update the table.

### Failed runs are silent budget burns

The audit graph only records successful `AgentStep` writes. LLM calls that fail validation, time out, or hit retries still consume tokens but produce no audit row. For cost discipline, **cross-reference your provider's billing console** — don't trust audit-graph cost as a complete picture. A common surprise: dev iteration at 2-3x of one happy-path run because every failed validation re-invokes the full prompt.

### Cost levers, in priority order

When a use case runs hot:

1. **Tighten the QA cap.** `MAX_QA_ITERATIONS=1` (one retry) before reaching for anything else. Each extra iteration replays the whole linear chain.
2. **Generate fewer candidates.** Over-generating to trim at QA was the easy assumption; "ideator produces N=desired_count" is cheaper at comparable quality.
3. **Use the cheap model where you can.** Reserve the QA-grade thinking model (`MODEL_QA`) for steps where the analysis IS the point (final QA verdict, complex routing decisions). Use `MODEL_PRIMARY` (flash) everywhere else.
4. **Prompt caching for repeated system prompts** — out of scope here, but worth knowing as a 50%+ saver in production.

Avoid: switching ALL judgment calls to flash. The savings rarely justify the QA quality drop, which then triggers more retries, which costs more in aggregate.
