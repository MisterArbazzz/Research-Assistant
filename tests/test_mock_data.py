"""Tests for the mock research data + name normalization."""

from __future__ import annotations

from data.mock_companies import MOCK_RESEARCH, normalize_company_name


def test_spec_companies_present() -> None:
    assert "Apple Inc." in MOCK_RESEARCH
    assert "Tesla" in MOCK_RESEARCH


def test_facts_have_required_keys() -> None:
    for company, facts in MOCK_RESEARCH.items():
        assert facts["recent_news"], company
        assert facts["stock_info"], company
        assert facts["key_developments"], company


def test_normalize_canonical_passthrough() -> None:
    assert normalize_company_name("Apple Inc.") == "Apple Inc."
    assert normalize_company_name("Tesla") == "Tesla"


def test_normalize_aliases() -> None:
    assert normalize_company_name("apple") == "Apple Inc."
    assert normalize_company_name("AAPL") == "Apple Inc."
    assert normalize_company_name("tesla motors") == "Tesla"
    assert normalize_company_name("facebook") == "Meta"
    assert normalize_company_name("google") == "Alphabet"


def test_normalize_case_insensitive_canonical() -> None:
    assert normalize_company_name("APPLE INC.") == "Apple Inc."


def test_normalize_unknown_returns_none() -> None:
    assert normalize_company_name("ACME Corp") is None
    assert normalize_company_name("") is None
    assert normalize_company_name(None) is None
