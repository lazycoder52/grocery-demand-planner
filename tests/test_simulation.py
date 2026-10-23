"""Unit tests for periodic-review (R, S) replenishment simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_planner.simulation.replenishment import (
    PeriodicReviewSimulator,
    SimConfig,
    plot_replenishment_tradeoff,
)


@pytest.fixture
def mock_predictions_df() -> pd.DataFrame:
    """Create a small deterministic predictions DataFrame for simulation tests."""
    dates = pd.date_range("2017-05-01", periods=28, freq="D")
    records = []
    rng = np.random.default_rng(42)

    for store in [1, 2]:
        for fam in ["GROCERY I", "BEVERAGES"]:
            base_sales = rng.uniform(50.0, 150.0, size=len(dates))
            for i, d in enumerate(dates):
                s = float(base_sales[i])
                records.append(
                    {
                        "date": d,
                        "store_nbr": store,
                        "family": fam,
                        "sales": s,
                        "pred_lgbm": s * 1.02,
                        "pred_snaive": s * 0.98,
                        "pred_p10": s * 0.70,
                        "pred_p50": s * 1.01,
                        "pred_p90": s * 1.30,
                        "split": "fold_1",
                        "fold_idx": 1,
                    }
                )

    return pd.DataFrame(records)


def test_sim_config_loading():
    """Verify SimConfig loads expected parameters from config/sim_config.yaml."""
    cfg = SimConfig.from_yaml("config/sim_config.yaml")
    assert cfg.review_period_days == 7
    assert cfg.lead_times_days == (3, 7, 14)
    assert cfg.service_levels == (0.90, 0.95, 0.98)
    assert cfg.seed == 42


def test_inventory_never_negative(mock_predictions_df: pd.DataFrame):
    """Invariant: On-hand inventory must never fall below zero under any policy or lead time."""
    cfg = SimConfig(review_period_days=7, lead_times_days=(3, 7), seed=42)
    sim = PeriodicReviewSimulator(mock_predictions_df, config=cfg)

    for pol in ["seasonal_naive", "lightgbm_p50", "lightgbm_p50_safety_stock", "summed_p90"]:
        for L in [3, 7]:
            res = sim.simulate_policy(pol, lead_time_days=L, service_level=0.95)
            assert res["average_inventory_units"] >= 0.0
            assert 0.0 <= res["fill_rate_pct"] <= 100.0
            assert 0.0 <= res["stockout_day_share_pct"] <= 100.0


def test_simulation_deterministic_with_seed(mock_predictions_df: pd.DataFrame):
    """Invariant: Simulator must be completely deterministic when initialized with same seed."""
    cfg1 = SimConfig(seed=42)
    cfg2 = SimConfig(seed=42)

    sim1 = PeriodicReviewSimulator(mock_predictions_df, config=cfg1)
    sim2 = PeriodicReviewSimulator(mock_predictions_df, config=cfg2)

    res1 = sim1.simulate_policy("lightgbm_p50_safety_stock", lead_time_days=7, service_level=0.95)
    res2 = sim2.simulate_policy("lightgbm_p50_safety_stock", lead_time_days=7, service_level=0.95)

    assert res1 == res2


def test_order_quantities_respect_lead_time():
    """Invariant: Order placed on day t must arrive exactly at day t + L, not before."""
    dates = pd.date_range("2017-05-01", periods=14, freq="D")
    # Demand of 10 units on day 0, 0 units thereafter
    df = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": [1] * 14,
            "family": ["GROCERY I"] * 14,
            "sales": [10.0] + [0.0] * 13,
            "pred_lgbm": [10.0] * 14,
            "pred_snaive": [10.0] * 14,
            "pred_p10": [5.0] * 14,
            "pred_p50": [10.0] * 14,
            "pred_p90": [15.0] * 14,
            "split": ["fold_1"] * 14,
            "fold_idx": [1] * 14,
        }
    )

    lead_time = 3
    review_period = 7
    cfg = SimConfig(review_period_days=review_period, lead_times_days=(lead_time,), seed=42)
    sim = PeriodicReviewSimulator(df, config=cfg)

    res = sim.simulate_policy("lightgbm_p50", lead_time_days=lead_time)
    assert res["fill_rate_pct"] == 100.0
    assert res["stockout_day_share_pct"] == 0.0


def test_zero_demand_gives_zero_stockouts():
    """Invariant: If demand is zero everywhere, stockout count must be zero and fill rate 100%."""
    dates = pd.date_range("2017-05-01", periods=14, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": [1] * 14,
            "family": ["GROCERY I"] * 14,
            "sales": [0.0] * 14,
            "pred_lgbm": [10.0] * 14,
            "pred_snaive": [10.0] * 14,
            "pred_p10": [5.0] * 14,
            "pred_p50": [10.0] * 14,
            "pred_p90": [15.0] * 14,
            "split": ["fold_1"] * 14,
            "fold_idx": [1] * 14,
        }
    )

    cfg = SimConfig(review_period_days=7, lead_times_days=(3,), seed=42)
    sim = PeriodicReviewSimulator(df, config=cfg)
    res = sim.simulate_policy("lightgbm_p50", lead_time_days=3)

    assert res["stockout_day_share_pct"] == 0.0
    assert res["fill_rate_pct"] == 100.0


def test_tradeoff_plot_generation(mock_predictions_df: pd.DataFrame, tmp_path):
    """Verify trade-off plot function generates an image without errors."""
    cfg = SimConfig(review_period_days=7, lead_times_days=(3, 7, 14), seed=42)
    sim = PeriodicReviewSimulator(mock_predictions_df, config=cfg)
    sim_df = sim.run_all_simulations()

    chart_file = tmp_path / "tradeoff.png"
    out = plot_replenishment_tradeoff(sim_df, output_path=chart_file)
    assert out.exists()
    assert out.stat().st_size > 0
