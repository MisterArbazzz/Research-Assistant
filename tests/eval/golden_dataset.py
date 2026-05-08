"""Hand-curated golden dataset for the Research Assistant eval suite.

Each case targets a specific behavior the system needs to handle:

  - clear queries that should bypass the validator (high confidence)
  - low-information queries that should trigger validator retries
  - ambiguous queries that should interrupt for clarification
  - multi-turn follow-ups that exercise conversation memory + query rewriting
  - unknown companies that should ship a degraded but coherent answer

Each case declares EXPECTATIONS the eval harness checks against the actual
trajectory + outputs. When the system regresses, the failing case identifies
which behavior broke.
"""

from __future__ import annotations

from typing import TypedDict


class Turn(TypedDict, total=False):
    user_query: str
    clarification_response: str  # supplied if an interrupt is expected


class GoldenCase(TypedDict, total=False):
    id: str
    description: str
    turns: list[Turn]
    # Expectations
    expected_company: str | None
    expected_clarity_status: str  # "clear" | "needs_clarification"
    expects_interrupt: bool
    min_confidence: float
    expects_validator_retry: bool


GOLDEN_CASES: list[GoldenCase] = [
    {
        "id": "clear_apple",
        "description": "Direct, well-formed query about Apple — should breeze through with high confidence.",
        "turns": [{"user_query": "Tell me about Apple's recent news and stock"}],
        "expected_company": "Apple Inc.",
        "expected_clarity_status": "clear",
        "expects_interrupt": False,
        "min_confidence": 6.0,
        "expects_validator_retry": False,
    },
    {
        "id": "clear_tesla",
        "description": "Direct query about Tesla — same shape as Apple but a different company.",
        "turns": [{"user_query": "What are the key recent developments at Tesla?"}],
        "expected_company": "Tesla",
        "expected_clarity_status": "clear",
        "expects_interrupt": False,
        "min_confidence": 6.0,
        "expects_validator_retry": False,
    },
    {
        "id": "ticker_lookup",
        "description": "User uses a ticker symbol — Clarity should canonicalize it.",
        "turns": [{"user_query": "Tell me about NVDA's AI strategy"}],
        "expected_company": "NVIDIA",
        "expected_clarity_status": "clear",
        "expects_interrupt": False,
        "min_confidence": 5.0,
        "expects_validator_retry": False,
    },
    {
        "id": "ambiguous_no_company",
        "description": "Query with no company — Clarity must interrupt for clarification.",
        "turns": [
            {
                "user_query": "How is the company doing?",
                "clarification_response": "Tesla",
            }
        ],
        "expected_company": "Tesla",
        "expected_clarity_status": "clear",
        "expects_interrupt": True,
        "min_confidence": 5.0,
        "expects_validator_retry": False,
    },
    {
        "id": "ambiguous_pronoun",
        "description": "Generic pronoun query — interrupt, then user supplies the company.",
        "turns": [
            {
                "user_query": "Tell me about their plans",
                "clarification_response": "Microsoft",
            }
        ],
        "expected_company": "Microsoft",
        "expected_clarity_status": "clear",
        "expects_interrupt": True,
        "min_confidence": 5.0,
        "expects_validator_retry": False,
    },
    {
        "id": "follow_up_ceo",
        "description": "Multi-turn: Tesla, then 'their CEO?' — query rewriter must resolve 'their' from history.",
        "turns": [
            {"user_query": "Tell me about Tesla"},
            {"user_query": "What about their CEO?"},
        ],
        "expected_company": "Tesla",
        "expected_clarity_status": "clear",
        "expects_interrupt": False,
        "min_confidence": 5.0,
        "expects_validator_retry": False,
    },
    {
        "id": "follow_up_competitors",
        "description": "Multi-turn: Apple, then 'how does it compare?' — vague follow-up needs history.",
        "turns": [
            {"user_query": "Tell me about Apple's stock"},
            {"user_query": "How is it doing this year?"},
        ],
        "expected_company": "Apple Inc.",
        "expected_clarity_status": "clear",
        "expects_interrupt": False,
        "min_confidence": 5.0,
        "expects_validator_retry": False,
    },
    {
        "id": "unknown_company",
        "description": "Made-up company — graph should still produce a coherent (degraded) answer, not crash.",
        "turns": [{"user_query": "Tell me about Acme Industries Inc"}],
        "expected_company": None,  # not in MOCK_RESEARCH
        "expected_clarity_status": "clear",
        "expects_interrupt": False,
        "min_confidence": 0.0,
        "expects_validator_retry": True,  # low confidence triggers validator
    },
]
