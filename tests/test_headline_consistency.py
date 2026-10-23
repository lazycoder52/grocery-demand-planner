"""Test that headline forecasting metrics in documentation strictly match metrics.json.

This test prevents documentation drift whenever models or cross-validation pipelines
are re-run or updated.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_readme_and_product_brief_match_metrics_json():
    root = Path(__file__).resolve().parent.parent
    metrics_file = root / "metrics" / "metrics.json"
    readme_file = root / "README.md"
    brief_file = root / "docs" / "PRODUCT_BRIEF.md"

    assert metrics_file.exists(), f"Missing {metrics_file}"
    assert readme_file.exists(), f"Missing {readme_file}"

    with metrics_file.open(encoding="utf-8") as f:
        metrics = json.load(f)

    readme_text = readme_file.read_text(encoding="utf-8")
    brief_text = brief_file.read_text(encoding="utf-8") if brief_file.exists() else ""

    cv_mean = metrics["models"]["cv_5_fold"]["mean_metrics"]
    holdout = metrics["models"]["holdout_confirmation"]["models"]

    # 1. 5-Fold CV Headline metrics
    lgbm_cv_wape = f"{cv_mean['lightgbm']['mean_wape']:.2f}%"
    snaive_cv_wape = f"{cv_mean['seasonal_naive_7d']['mean_wape']:.2f}%"
    fva_metrics = cv_mean["forecast_value_added_vs_seasonal_naive"]
    fva_pts = f"{fva_metrics['mean_wape_reduction_pts']:.2f}"
    rel_gain = f"{fva_metrics['relative_error_reduction_pct']:.2f}%"

    assert lgbm_cv_wape in readme_text, f"CV LightGBM WAPE {lgbm_cv_wape} not in README"
    assert snaive_cv_wape in readme_text, f"CV S.Naive WAPE {snaive_cv_wape} not in README"
    assert f"+{fva_pts} pts" in readme_text or f"+{fva_pts} percentage points" in readme_text
    assert rel_gain in readme_text

    # 2. Holdout Confirmation metrics
    lgbm_holdout_wape = f"{holdout['lightgbm']['wape']:.2f}%"
    snaive_holdout_wape = f"{holdout['seasonal_naive_7d']['wape']:.2f}%"
    lgbm_holdout_bias = f"{holdout['lightgbm']['bias']:+.2f}%"

    assert lgbm_holdout_wape in readme_text
    assert snaive_holdout_wape in readme_text
    assert lgbm_holdout_bias in readme_text

    # 3. Product Brief consistency
    if brief_text:
        assert lgbm_cv_wape in brief_text, f"CV LightGBM WAPE {lgbm_cv_wape} not in brief"
        assert snaive_cv_wape in brief_text, f"CV S.Naive WAPE {snaive_cv_wape} not in brief"
        assert lgbm_holdout_wape in brief_text, f"Holdout WAPE {lgbm_holdout_wape} not in brief"
        assert snaive_holdout_wape in brief_text, "Holdout S.Naive mismatch"
