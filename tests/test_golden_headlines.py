"""Asserts that metrics/metrics.json matches committed headline values.

Updating tests/golden_headlines.json must be an explicit, deliberate commit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FILE = ROOT / "tests" / "golden_headlines.json"
METRICS_FILE = ROOT / "metrics" / "metrics.json"


def _get_metric_val(metrics: dict[str, Any], key_path: str) -> Any:
    """Resolve dot path or special trade-off paths from metrics dictionary."""
    if key_path.startswith("replenishment_simulation.target_fill_rate_tradeoffs."):
        _, subpart = key_path.split("replenishment_simulation.target_fill_rate_tradeoffs.", 1)
        tag, field = subpart.split(".", 1)
        L = 3 if tag.startswith("L3") else 7
        target_fr = 99.0
        tradeoffs = metrics["replenishment_simulation"]["target_fill_rate_tradeoffs"]
        item = next(
            r
            for r in tradeoffs
            if r.get("lead_time_days") == L
            and r.get("target_fill_rate_pct") == target_fr
            and r.get("comparison") == "LightGBM Point vs Seasonal Naive"
        )
        return item[field]

    curr: Any = metrics
    for token in key_path.split("."):
        curr = curr[token]
    return curr


def test_metrics_json_matches_golden_headlines():
    """Assert all golden headline values match current metrics.json."""
    assert GOLDEN_FILE.exists(), f"Missing golden headlines file: {GOLDEN_FILE}"
    assert METRICS_FILE.exists(), f"Missing metrics file: {METRICS_FILE}"

    with GOLDEN_FILE.open(encoding="utf-8") as f:
        golden = json.load(f)
    with METRICS_FILE.open(encoding="utf-8") as f:
        metrics = json.load(f)

    discrepancies: list[str] = []

    for path_key, expected_val in golden.items():
        try:
            actual_val = _get_metric_val(metrics, path_key)
        except Exception as e:
            discrepancies.append(f"{path_key}: Failed to resolve from metrics.json ({e})")
            continue

        if isinstance(expected_val, float):
            if abs(actual_val - expected_val) > 1e-4:
                discrepancies.append(
                    f"{path_key}: expected {expected_val}, got {actual_val} "
                    f"(diff {actual_val - expected_val})"
                )
        else:
            if actual_val != expected_val:
                discrepancies.append(f"{path_key}: expected {expected_val!r}, got {actual_val!r}")

    if discrepancies:
        pytest.fail(
            f"Found {len(discrepancies)} golden headline discrepancy/discrepancies:\n"
            + "\n".join(f"  - {d}" for d in discrepancies)
        )
