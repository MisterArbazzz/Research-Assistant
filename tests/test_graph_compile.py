"""Verify the graph compiles cleanly and has the topology we expect.

Doesn't run the graph — that's an integration concern. This catches typos in
node names, missing edges, and routing-target mismatches at the cheapest
possible test tier.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph


@pytest.fixture(autouse=True)
def _override_settings(settings_override):
    settings_override(GOOGLE_API_KEY="x", RESEARCH_BACKEND="mock", TAVILY_API_KEY="")
    from src.config import get_settings

    get_settings.cache_clear()


async def test_graph_compiles() -> None:
    graph = await build_graph(MemorySaver())
    assert graph is not None


async def test_graph_has_all_nodes() -> None:
    graph = await build_graph(MemorySaver())
    nodes = set(graph.get_graph().nodes.keys())
    expected = {
        "__start__",
        "clarity_agent",
        "interrupt_node",
        "research_agent",
        "validator_agent",
        "synthesis_agent",
        "memory_writer",
        "__end__",
    }
    assert expected.issubset(nodes)


async def test_graph_edges_include_retry_loop() -> None:
    graph = await build_graph(MemorySaver())
    edges = graph.get_graph().edges
    pairs = {(e.source, e.target) for e in edges}
    # Validator → Research is the retry loop the spec demands.
    assert ("validator_agent", "research_agent") in pairs
    # Interrupt → Clarity is the human-in-the-loop resume path.
    assert ("interrupt_node", "clarity_agent") in pairs
    # Synthesis → memory_writer → END (Tier 4)
    assert ("synthesis_agent", "memory_writer") in pairs
    assert ("memory_writer", "__end__") in pairs
