"""Unit tests for report charts generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from demand_planner.reports.charts import (
    plot_actual_vs_forecast_stores,
    plot_wape_by_family,
    plot_zero_rate_by_family,
)


def test_charts_generation(tmp_path: Path):
    """Verify that all report charts generate valid PNG files."""
    metrics_path = tmp_path / "metrics.json"
    dummy_metrics = {
        "audit": {
            "family_stats": {
                "GROCERY I": {"zero_share_pct": 5.0, "is_intermittent": False},
                "BOOKS": {"zero_share_pct": 95.0, "is_intermittent": True},
            }
        },
        "models": {
            "by_family": {
                "GROCERY I": {
                    "lightgbm": {"wape": 12.0, "total_actual": 1000.0},
                    "seasonal_naive_7d": {"wape": 18.0},
                },
                "BOOKS": {
                    "lightgbm": {"wape": 45.0, "total_actual": 50.0},
                    "seasonal_naive_7d": {"wape": 60.0},
                },
            }
        },
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(dummy_metrics, fh)

    # 1. WAPE chart
    wape_out = tmp_path / "wape.png"
    plot_wape_by_family(metrics_path=metrics_path, output_path=wape_out)
    assert wape_out.exists()
    assert wape_out.stat().st_size > 1000

    # 2. Zero-rate chart
    zero_out = tmp_path / "zero.png"
    plot_zero_rate_by_family(metrics_path=metrics_path, output_path=zero_out)
    assert zero_out.exists()
    assert zero_out.stat().st_size > 1000

    # 3. Store time series chart
    preds_path = tmp_path / "preds.parquet"
    dates = pd.date_range("2017-08-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "store_nbr": [1] * 5 + [2] * 5,
            "date": list(dates) * 2,
            "sales": [100.0, 120.0, 110.0, 130.0, 140.0] * 2,
            "pred_lgbm": [105.0, 115.0, 112.0, 128.0, 138.0] * 2,
            "pred_seasonal_naive": [100.0, 110.0, 115.0, 125.0, 135.0] * 2,
        }
    )
    df.to_parquet(preds_path, index=False)

    store_out = tmp_path / "stores.png"
    plot_actual_vs_forecast_stores(preds_path=preds_path, output_path=store_out, stores=(1, 2))
    assert store_out.exists()
    assert store_out.stat().st_size > 1000
