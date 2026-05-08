"""LangGraph factory for the Research Assistant.

Wires the four agent nodes plus the interrupt node into a single graph:

    START
      v
    clarity_agent ──[needs_clarification]──> interrupt_node ──> clarity_agent
      │
      └─[clear]─> research_agent
                    │
                    ├─[confidence >= 6]─> synthesis_agent ──> END
                    │
                    └─[confidence < 6]─> validator_agent
                                           │
                                           ├─[sufficient OR cap reached]─> synthesis_agent
                                           │
                                           └─[insufficient AND attempts < cap]─> research_agent

Compile with the checkpointer (AsyncSqliteSaver in lifespan, MemorySaver in
tests). path_map on every conditional edge so a typo in routing fails at
compile time, not runtime.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    clarity_agent,
    interrupt_node,
    research_agent,
    synthesis_agent,
    validator_agent,
)
from .routing import (
    route_after_clarity,
    route_after_research,
    route_after_validator,
)
from .state import ResearchState


async def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    workflow: StateGraph = StateGraph(ResearchState)

    workflow.add_node("clarity_agent", clarity_agent)
    workflow.add_node("interrupt_node", interrupt_node)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("validator_agent", validator_agent)
    workflow.add_node("synthesis_agent", synthesis_agent)

    workflow.add_edge(START, "clarity_agent")

    workflow.add_conditional_edges(
        "clarity_agent",
        route_after_clarity,
        path_map={
            "interrupt_node": "interrupt_node",
            "research_agent": "research_agent",
        },
    )

    # After the human clarifies, route back through clarity for re-evaluation.
    workflow.add_edge("interrupt_node", "clarity_agent")

    workflow.add_conditional_edges(
        "research_agent",
        route_after_research,
        path_map={
            "validator_agent": "validator_agent",
            "synthesis_agent": "synthesis_agent",
        },
    )

    workflow.add_conditional_edges(
        "validator_agent",
        route_after_validator,
        path_map={
            "research_agent": "research_agent",
            "synthesis_agent": "synthesis_agent",
        },
    )

    workflow.add_edge("synthesis_agent", END)

    return workflow.compile(checkpointer=checkpointer)
