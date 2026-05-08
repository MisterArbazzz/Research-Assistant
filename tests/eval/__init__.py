"""Evaluation harness (Tier 3).

Not a pytest target — pytest tests under `tests/eval/` are the unit tests
for the metric functions themselves. The full eval is run via:

    uv run python -m tests.eval.run_eval

It exercises the full agent pipeline against a curated golden dataset and
emits per-case + aggregate scores across four metrics + a trajectory
score. Uses the mock retrieval backend so results are deterministic and
free; LLM calls (clarity / research / validator / synthesis / judges)
remain live.
"""

from __future__ import annotations
