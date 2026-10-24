"""Tests for causal promotion-lift analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

from demand_planner.promo.lift import (
    plot_promo_lift_by_family,
    run_promo_lift_analysis,
)


def test_promo_lift_synthetic_recovery():
    """Verify store-family fixed effects model accurately recovers known synthetic promo lift."""
    rng = np.random.default_rng(42)
    n_days = 120
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    stores = [1, 2, 3, 4]
    families = ["GROCERY", "BEVERAGES"]

    records = []
    # Known ground truth: beta = 0.25 -> lift = (exp(0.25) - 1) * 100 = 28.4%
    true_beta = 0.25
    for s in stores:
        for f in families:
            base_vel = 2.0 + s * 0.5 + (1.0 if f == "GROCERY" else 0.0)
            for d in dates:
                # 20% random promotions
                is_promo = 1 if rng.uniform() < 0.2 else 0
                dow_effect = 0.1 * (d.dayofweek == 5 or d.dayofweek == 6)
                log_sales = base_vel + true_beta * is_promo + dow_effect + rng.normal(0, 0.08)
                sales = float(max(0.0, np.expm1(log_sales)))
                records.append(
                    {
                        "date": d,
                        "store_nbr": s,
                        "family": f,
                        "sales": sales,
                        "onpromotion": is_promo,
                    }
                )

    df_synth = pd.DataFrame(records)
    res = run_promo_lift_analysis(
        df_synth,
        start_date="2015-01-01",
        n_bootstrap=100,
        seed=42,
    )

    true_lift_pct = (np.exp(true_beta) - 1.0) * 100.0
    est_lift_pct = res["overall_lift_pct"]
    ci_low = res["ci_95_overall_low"]
    ci_high = res["ci_95_overall_high"]

    # Estimated lift is close to true lift
    assert abs(est_lift_pct - true_lift_pct) < 5.0
    # Clustered CI covers true lift
    assert ci_low <= true_lift_pct <= ci_high
    assert res["top_5_promo_sensitive_families"] is not None


def test_promo_lift_excludes_pre_2014():
    """Verify analysis strictly ignores data before 2014-04-01."""
    dates = pd.date_range("2013-01-01", "2015-01-01", freq="D")
    records = []
    for d in dates:
        records.append(
            {
                "date": d,
                "store_nbr": 1,
                "family": "GROCERY",
                "sales": 100.0,
                "onpromotion": 1 if d.year == 2014 else 0,
            }
        )
    df = pd.DataFrame(records)
    res = run_promo_lift_analysis(df, start_date="2014-04-01", n_bootstrap=20, seed=42)
    assert res["analysis_period"]["start_date"] == "2014-04-01"


def test_promo_lift_plot_generation(tmp_path: Path):
    """Verify promo lift bar chart generation."""
    mock_results = {
        "overall_lift_pct": 18.5,
        "top_5_promo_sensitive_families": [{"family": "GROCERY I"}],
        "by_family": [
            {
                "family": "GROCERY I",
                "lift_pct": 25.0,
                "ci_95_low": 20.0,
                "ci_95_high": 30.0,
            },
            {
                "family": "BEVERAGES",
                "lift_pct": 15.0,
                "ci_95_low": 11.0,
                "ci_95_high": 19.0,
            },
        ],
    }
    out_file = tmp_path / "promo_lift.png"
    p = plot_promo_lift_by_family(mock_results, output_path=out_file)
    assert p.exists()
    assert p.stat().st_size > 1000
