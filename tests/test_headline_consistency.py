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
    h_fva = metrics["models"]["holdout_confirmation"]["forecast_value_added_vs_seasonal_naive"]
    holdout_fva_pts = f"{h_fva['wape_reduction_percentage_points']:.2f}"
    holdout_fva_rel = f"{h_fva['relative_error_reduction_pct']:.2f}%"

    assert lgbm_holdout_wape in readme_text
    assert snaive_holdout_wape in readme_text
    assert lgbm_holdout_bias in readme_text
    assert (
        f"+{holdout_fva_pts} percentage points" in readme_text
        or f"+{holdout_fva_pts} pts" in readme_text
    )
    assert holdout_fva_rel in readme_text

    # 3. Product Brief consistency
    if brief_text:
        assert lgbm_cv_wape in brief_text, f"CV LightGBM WAPE {lgbm_cv_wape} not in brief"
        assert snaive_cv_wape in brief_text, f"CV S.Naive WAPE {snaive_cv_wape} not in brief"
        assert lgbm_holdout_wape in brief_text, f"Holdout WAPE {lgbm_holdout_wape} not in brief"
        assert snaive_holdout_wape in brief_text, "Holdout S.Naive mismatch"
        assert (
            f"+{holdout_fva_pts} percentage points" in brief_text
            or f"+{holdout_fva_pts} pts" in brief_text
        )
        assert holdout_fva_rel in brief_text

    # 4. Replenishment Matched Fill Rate Range Consistency
    rep = metrics.get("replenishment_simulation", {})
    tradeoffs = rep.get("target_fill_rate_tradeoffs", [])
    l3_99 = next(
        (
            t
            for t in tradeoffs
            if t["lead_time_days"] == 3
            and t["target_fill_rate_pct"] == 99.0
            and "Point" in t["forecaster"]
        ),
        None,
    )
    l7_99 = next(
        (
            t
            for t in tradeoffs
            if t["lead_time_days"] == 7
            and t["target_fill_rate_pct"] == 99.0
            and "Point" in t["forecaster"]
        ),
        None,
    )
    if l3_99 and l7_99:
        l3_diff = f"{l3_99['pct_difference']:.2f}%"
        l7_diff = f"{l7_99['pct_difference']:.2f}%"
        assert l3_diff in readme_text, f"L=3 replenishment diff {l3_diff} not in README"
        assert l7_diff in readme_text, f"L=7 replenishment diff {l7_diff} not in README"
        if brief_text:
            assert l3_diff in brief_text, f"L=3 replenishment diff {l3_diff} not in brief"
            assert l7_diff in brief_text, f"L=7 replenishment diff {l7_diff} not in brief"

        # 5. Bootstrap Confidence Intervals for L=3 and L=7
        l3_s_low = l3_99["ci_95_series_pct_diff_low"]
        l3_s_high = l3_99["ci_95_series_pct_diff_high"]
        l3_t_low = l3_99["ci_95_time_block_pct_diff_low"]
        l3_t_high = l3_99["ci_95_time_block_pct_diff_high"]

        l7_s_low = l7_99["ci_95_series_pct_diff_low"]
        l7_s_high = l7_99["ci_95_series_pct_diff_high"]
        l7_t_low = l7_99["ci_95_time_block_pct_diff_low"]
        l7_t_high = l7_99["ci_95_time_block_pct_diff_high"]

        l3_series_ci = f"[{l3_s_low:.2f}%, {l3_s_high:.2f}%]"
        l3_time_ci = f"[{l3_t_low:.2f}%, {l3_t_high:.2f}%]"
        l7_series_ci = f"[{l7_s_low:.2f}%, {l7_s_high:.2f}%]"
        l7_time_ci = f"[{l7_t_low:.2f}%, {l7_t_high:.2f}%]"

        assert l3_series_ci in readme_text, f"L=3 Series CI {l3_series_ci} not in README"
        assert l3_time_ci in readme_text, f"L=3 Time-Block CI {l3_time_ci} not in README"
        assert l7_series_ci in readme_text, f"L=7 Series CI {l7_series_ci} not in README"
        assert l7_time_ci in readme_text, f"L=7 Time-Block CI {l7_time_ci} not in README"

        if brief_text:
            assert l3_series_ci in brief_text, f"L=3 Series CI {l3_series_ci} not in brief"
            assert l3_time_ci in brief_text, f"L=3 Time-Block CI {l3_time_ci} not in brief"
            assert l7_series_ci in brief_text, f"L=7 Series CI {l7_series_ci} not in brief"
            assert l7_time_ci in brief_text, f"L=7 Time-Block CI {l7_time_ci} not in brief"


def test_simulation_window_error_ratios_and_caveats():
    """Verify window-sum error ratios and empirical caveats in documentation."""
    root = Path(__file__).resolve().parent.parent
    readme_text = (root / "README.md").read_text(encoding="utf-8")
    brief_text = (root / "docs" / "PRODUCT_BRIEF.md").read_text(encoding="utf-8")
    insights_text = (root / "docs" / "INSIGHTS.md").read_text(encoding="utf-8")

    # Ratio 1.436 at W=10 (L=3) and 1.522 at W=14 (L=7)
    doc_sources = [
        (readme_text, "README"),
        (brief_text, "PRODUCT_BRIEF"),
        (insights_text, "INSIGHTS"),
    ]
    for text, name in doc_sources:
        assert "1.436" in text, f"Window ratio 1.436 missing in {name}"
        assert "1.522" in text, f"Window ratio 1.522 missing in {name}"
        assert "DAIRY" in text, f"DAIRY caveat missing in {name}"
        assert "BABY CARE" in text, f"BABY CARE caveat missing in {name}"
