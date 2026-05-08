"""Pure-function tests for the conditional routing.

Routing has no LLM, no I/O — these tests build hand-crafted ResearchState
instances and assert the next-node string. They run in milliseconds and
guard against regressions when the cap or threshold gets touched.
"""

from __future__ import annotations

from src.graph.routing import (
    is_cost_ceiling_reached,
    route_after_clarity,
    route_after_research,
    route_after_validator,
)
from src.graph.state import ResearchState


def _state(**kwargs: object) -> ResearchState:
    return ResearchState(run_id="test", **kwargs)  # type: ignore[arg-type]


def test_clarity_routes_to_interrupt_when_unclear() -> None:
    s = _state(clarity_status="needs_clarification")
    assert route_after_clarity(s) == "interrupt_node"


def test_clarity_routes_to_research_when_clear() -> None:
    s = _state(clarity_status="clear", company="Apple Inc.")
    assert route_after_clarity(s) == "research_agent"


def test_research_high_confidence_skips_validator() -> None:
    s = _state(confidence_score=8.5)
    assert route_after_research(s) == "synthesis_agent"


def test_research_low_confidence_routes_to_validator() -> None:
    s = _state(confidence_score=4.0)
    assert route_after_research(s) == "validator_agent"


def test_research_at_threshold_skips_validator() -> None:
    # >= 6 is the spec — equality must skip
    s = _state(confidence_score=6.0)
    assert route_after_research(s) == "synthesis_agent"


def test_validator_sufficient_routes_to_synthesis() -> None:
    s = _state(validation_result="sufficient", research_attempts=1)
    assert route_after_validator(s) == "synthesis_agent"


def test_validator_insufficient_loops_back_to_research() -> None:
    s = _state(validation_result="insufficient", research_attempts=1)
    assert route_after_validator(s) == "research_agent"


def test_validator_cap_short_circuits_to_synthesis(settings_override) -> None:
    settings_override(GOOGLE_API_KEY="x", MAX_RESEARCH_ATTEMPTS=3)
    from src.config import get_settings

    get_settings.cache_clear()

    s = _state(validation_result="insufficient", research_attempts=3)
    assert route_after_validator(s) == "synthesis_agent"


def test_cost_ceiling_guard_below_limit(settings_override) -> None:
    settings_override(GOOGLE_API_KEY="x", COST_CEILING_PER_RUN_USD=0.10)
    from src.config import get_settings

    get_settings.cache_clear()

    assert is_cost_ceiling_reached(_state(total_cost_usd=0.05)) is False


def test_cost_ceiling_guard_at_or_above_limit(settings_override) -> None:
    settings_override(GOOGLE_API_KEY="x", COST_CEILING_PER_RUN_USD=0.10)
    from src.config import get_settings

    get_settings.cache_clear()

    # >= ceiling triggers the guard
    assert is_cost_ceiling_reached(_state(total_cost_usd=0.10)) is True
    assert is_cost_ceiling_reached(_state(total_cost_usd=0.25)) is True


def test_research_short_circuits_to_synthesis_when_ceiling_hit(settings_override) -> None:
    settings_override(GOOGLE_API_KEY="x", COST_CEILING_PER_RUN_USD=0.001)
    from src.config import get_settings

    get_settings.cache_clear()

    # Even with low confidence, ceiling forces synthesis to bound spend
    s = _state(confidence_score=2.0, total_cost_usd=0.005)
    assert route_after_research(s) == "synthesis_agent"


def test_validator_short_circuits_to_synthesis_when_ceiling_hit(settings_override) -> None:
    settings_override(
        GOOGLE_API_KEY="x", COST_CEILING_PER_RUN_USD=0.001, MAX_RESEARCH_ATTEMPTS=3
    )
    from src.config import get_settings

    get_settings.cache_clear()

    # Insufficient + attempts left would normally retry, but ceiling overrides
    s = _state(
        validation_result="insufficient",
        research_attempts=1,
        total_cost_usd=0.005,
    )
    assert route_after_validator(s) == "synthesis_agent"
