"""Mock research data for the Research Assistant.

Stand-in for a real search backend (e.g. Tavily MCP). Keys are the canonical
company names; values match the schema the Research Agent's structured-output
LLM call digests into a `ResearchFindings`. The first two entries are taken
verbatim from the problem statement; the rest exist so retrieval has more
than two options to disambiguate.

normalize_company_name() maps user-typed variations ("apple", "AAPL", "Tesla
Motors") to a canonical key. Returns None when no match — the Research
Agent then short-circuits to a low-confidence stub finding (CLAUDE.md
pattern #3: sentinel, not None).
"""

from __future__ import annotations

from typing import TypedDict


class _CompanyFacts(TypedDict):
    recent_news: str
    stock_info: str
    key_developments: str


MOCK_RESEARCH: dict[str, _CompanyFacts] = {
    "Apple Inc.": {
        "recent_news": "Launched Vision Pro, expanding services revenue",
        "stock_info": "Trading at $195, up 45% YTD",
        "key_developments": "AI integration across product line",
    },
    "Tesla": {
        "recent_news": "Cybertruck deliveries ramping up",
        "stock_info": "Trading at $242, volatile quarter",
        "key_developments": (
            "FSD v12 rollout, energy storage growth. CEO Elon Musk continues "
            "to drive product strategy across automotive, energy, and AI."
        ),
    },
    "Microsoft": {
        "recent_news": "Copilot integration expanding across Office and Windows",
        "stock_info": "Trading at $421, up 12% YTD",
        "key_developments": (
            "Azure AI services growth, OpenAI partnership deepening, gaming "
            "division consolidating after Activision Blizzard acquisition."
        ),
    },
    "NVIDIA": {
        "recent_news": "H200 GPU shipments, Blackwell architecture announced",
        "stock_info": "Trading at $880, up 78% YTD",
        "key_developments": (
            "AI training and inference dominance, automotive partnerships, "
            "CUDA software moat continues to widen."
        ),
    },
    "Meta": {
        "recent_news": "Llama 3 released, Threads passes 200M users",
        "stock_info": "Trading at $478, up 22% YTD",
        "key_developments": (
            "Reality Labs losses narrowing, advertising business resilient, "
            "open-source AI strategy differentiating from competitors."
        ),
    },
    "Alphabet": {
        "recent_news": "Gemini 2.5 launch, Search facing AI overview transition",
        "stock_info": "Trading at $172, up 18% YTD",
        "key_developments": (
            "Cloud profitability achieved, Waymo expanding, antitrust rulings "
            "creating uncertainty in core search business."
        ),
    },
}


# Variations users tend to type, mapped to canonical keys above. Lowercased
# during lookup so we don't have to enumerate every casing.
_ALIASES: dict[str, str] = {
    "apple": "Apple Inc.",
    "apple inc": "Apple Inc.",
    "apple inc.": "Apple Inc.",
    "aapl": "Apple Inc.",
    "tesla": "Tesla",
    "tesla motors": "Tesla",
    "tsla": "Tesla",
    "microsoft": "Microsoft",
    "msft": "Microsoft",
    "nvidia": "NVIDIA",
    "nvda": "NVIDIA",
    "meta": "Meta",
    "facebook": "Meta",
    "meta platforms": "Meta",
    "alphabet": "Alphabet",
    "google": "Alphabet",
    "googl": "Alphabet",
}


def normalize_company_name(raw: str | None) -> str | None:
    """Map user-typed variations to a canonical key in MOCK_RESEARCH.

    Returns None when no match — caller is expected to handle the miss
    rather than receive a guessed canonical name.
    """
    if not raw:
        return None
    key = raw.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # Direct hit on canonical (case-insensitive) — handles "APPLE INC."
    for canonical in MOCK_RESEARCH:
        if canonical.lower() == key:
            return canonical
    return None
