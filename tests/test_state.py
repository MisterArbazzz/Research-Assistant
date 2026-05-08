"""ResearchState schema tests.

Verifies the reducer-typed fields accumulate correctly (LangGraph relies on
this) and the structured-output sub-models reject obviously-bad data.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from src.graph.state import (
    ParsedClarity,
    ResearchFindings,
    ResearchState,
    ValidationVerdict,
)


def test_research_state_minimal_construction() -> None:
    s = ResearchState(run_id="abc")
    assert s.run_id == "abc"
    assert s.research_attempts == 0
    assert s.audit_log == []
    assert s.messages == []


def test_research_state_carries_messages() -> None:
    s = ResearchState(
        run_id="abc",
        messages=[HumanMessage(content="hi"), AIMessage(content="hey")],
    )
    assert len(s.messages) == 2


def test_parsed_clarity_accepts_clear() -> None:
    p = ParsedClarity(clarity_status="clear", company="Apple Inc.")
    assert p.clarity_status == "clear"


def test_parsed_clarity_rejects_bad_status() -> None:
    with pytest.raises(ValidationError):
        ParsedClarity(clarity_status="maybe")  # type: ignore[arg-type]


def test_research_findings_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        ResearchFindings(
            recent_news="x", stock_info="y", key_developments="z", confidence_score=11.0
        )
    with pytest.raises(ValidationError):
        ResearchFindings(
            recent_news="x", stock_info="y", key_developments="z", confidence_score=-1.0
        )


def test_research_findings_rejects_empty_strings() -> None:
    # min_length=1 catches the case where Gemini returns "" for a required field
    with pytest.raises(ValidationError):
        ResearchFindings(
            recent_news="", stock_info="y", key_developments="z", confidence_score=5.0
        )


def test_validation_verdict_default_notes() -> None:
    v = ValidationVerdict(validation_result="sufficient")
    assert v.notes == []
