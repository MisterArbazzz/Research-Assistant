"""Architecture tab — static reference (mermaid, state schema, routing)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

MERMAID_GRAPH = """
```mermaid
flowchart TD
    START([START]) --> Clarity[clarity_agent<br/>Gemini Flash]
    Clarity -- needs_clarification --> Interrupt[interrupt_node<br/>pause for human]
    Interrupt --> Clarity
    Clarity -- clear --> Research[research_agent<br/>rewrite → Tavily → rerank → digest]
    Research -- confidence ≥ 6 OR ceiling --> Synthesis
    Research -- confidence < 6 --> Validator[validator_agent<br/>Gemini Pro]
    Validator -- sufficient OR cap reached --> Synthesis[synthesis_agent<br/>Gemini Flash]
    Validator -- insufficient AND attempts < cap --> Research
    Synthesis --> Memory[memory_writer<br/>selective writes]
    Memory --> END([END])
```
"""

STATE_SCHEMA_TABLE = [
    {"field": "run_id", "type": "str", "reducer": "replace", "purpose": "UUID for this run"},
    {"field": "user_query", "type": "str", "reducer": "replace", "purpose": "Current turn input"},
    {"field": "clarity_status", "type": "Literal | None", "reducer": "replace", "purpose": "clear / needs_clarification"},
    {"field": "clarification_request", "type": "str | None", "reducer": "replace", "purpose": "Question to ask the human"},
    {"field": "clarification_response", "type": "str | None", "reducer": "replace", "purpose": "What the human said"},
    {"field": "company", "type": "str | None", "reducer": "replace", "purpose": "Canonical company name"},
    {"field": "research_findings", "type": "dict | None", "reducer": "replace", "purpose": "News / stock / developments"},
    {"field": "confidence_score", "type": "float | None", "reducer": "replace", "purpose": "0-10 from research"},
    {"field": "research_attempts", "type": "int", "reducer": "replace", "purpose": "Counter for the cap"},
    {"field": "validation_result", "type": "Literal | None", "reducer": "replace", "purpose": "sufficient / insufficient"},
    {"field": "validation_notes", "type": "list[str]", "reducer": "replace", "purpose": "What to fix on retry"},
    {"field": "final_answer", "type": "str | None", "reducer": "replace", "purpose": "User-facing answer"},
    {"field": "total_cost_usd", "type": "float", "reducer": "replace-on-update", "purpose": "Cumulative LLM spend"},
    {"field": "audit_log", "type": "list[dict]", "reducer": "add (append)", "purpose": "Per-node breadcrumbs"},
    {"field": "messages", "type": "list[BaseMessage]", "reducer": "add_messages", "purpose": "Multi-turn conversation"},
]

ROUTING_RULES = """
**`route_after_clarity`** (`src/graph/routing.py`)
- `interrupt_node` if `clarity_status == "needs_clarification"`
- else `research_agent`

**`route_after_research`**
- `synthesis_agent` if cost ceiling hit OR `confidence_score ≥ 6`
- else `validator_agent`

**`route_after_validator`**
- `synthesis_agent` if `validation_result == "sufficient"` OR
  `research_attempts ≥ MAX_RESEARCH_ATTEMPTS` OR cost ceiling hit
- else `research_agent` (the retry loop)

A grep for `MAX_RESEARCH_ATTEMPTS` finds it twice — once in `config.py`
(definition), once in `routing.py` (enforcement). Same for
`COST_CEILING_PER_RUN_USD`.
"""

RUN_TRANSCRIPT_PATHS = [
    Path("examples/run_1.txt"),
    Path("examples/run_2.txt"),
]


def render() -> None:
    st.subheader("Architecture — graph, state, routing")

    st.markdown("### Graph topology")
    st.markdown(MERMAID_GRAPH)

    st.markdown("### State schema (`src/graph/state.py`)")
    st.dataframe(STATE_SCHEMA_TABLE, use_container_width=True, hide_index=True)
    st.caption(
        "**Reducers:** `add` appends to a list. `add_messages` is "
        "LangGraph's message-channel reducer (handles ID-based merging). "
        "`replace-on-update` means each node returns the new cumulative "
        "value rather than a delta."
    )

    st.markdown("### Routing rules")
    st.markdown(ROUTING_RULES)

    st.markdown("### Bonus tiers in this build")
    st.markdown(
        """
- **Tier 1 — Observability**: LangSmith + OpenTelemetry spans on every
  node (input_tokens, output_tokens, model, cost_usd, latency_ms) +
  per-run cost ceiling enforced once in `routing.py`.
- **Tier 2 — Retrieval quality**: query rewriting (Flash) + Tavily live
  search + cross-encoder rerank via `flashrank` (ONNX, ~30 MB).
- **Tier 3 — Evaluation**: 8 hand-curated golden cases scored across 4
  RAGAS-style metrics (recall, precision, faithfulness, relevance) plus
  an LLM-as-judge trajectory eval. Last run: 8/8 PASS at 100%.
- **Tier 4 — Long-term memory**: SQLite + sqlite-vec store keyed by
  `user_id`. Selective writes via the `memory_writer` node after
  synthesis. Retrieved into research's prompt before the digest.
"""
    )

    st.markdown("### Saved demo transcripts")
    for path in RUN_TRANSCRIPT_PATHS:
        if path.exists():
            with st.expander(str(path), expanded=False):
                st.code(path.read_text(encoding="utf-8"), language="text")
        else:
            st.caption(f"_(missing: {path})_")
