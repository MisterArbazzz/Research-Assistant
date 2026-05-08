"""Retrieval-quality helpers (Tier 2).

Layered on top of Tavily search inside the Research Agent:
  query_rewriting → tavily_search → rerank → top-K → LLM digest

Each helper is independently togglable via settings flags so you can A/B
the retrieval pipeline without touching agent code.
"""

from __future__ import annotations
