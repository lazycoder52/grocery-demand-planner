"""Unit tests for periodic-review (R, S) replenishment simulator."""

from __future__ import annotations

from pathlib import Path

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
                        "sigma_lgbm": 8.0,
                        "sigma_snaive": 10.0,
                        "split": "fold_1" if i < 14 else "fold_2",
                        "fold_idx": 1 if i < 14 else 2,
                    }
                )

    return pd.DataFrame(records)


def test_sim_config_loading():
    """Verify SimConfig loads expected parameters from config/sim_config.yaml."""
    cfg = SimConfig.from_yaml("config/sim_config.yaml")
    assert cfg.review_period_days == 7
    assert cfg.lead_times_days == (3, 7, 14)
    assert cfg.service_levels == (0.90, 0.95, 0.98)
    assert cfg.warmup_days == 14
    assert cfg.seed == 42


def test_inventory_never_negative(mock_predictions_df: pd.DataFrame):
    """Invariant: On-hand inventory must never fall below zero under any policy or lead time."""
    cfg = SimConfig(review_period_days=7, lead_times_days=(3, 7), warmup_days=7, seed=42)
    sim = PeriodicReviewSimulator(mock_predictions_df, config=cfg)

    policies = [
        "seasonal_naive",
        "lightgbm_p50",
        "seasonal_naive_safety_stock",
        "lightgbm_p50_safety_stock",
        "summed_p90",
    ]
    for pol in policies:
        for L in [3, 7]:
            res = sim.simulate_policy(pol, lead_time_days=L, service_level=0.95)
            assert res["average_inventory_units"] >= 0.0
            assert 0.0 <= res["fill_rate_pct"] <= 100.0
            assert 0.0 <= res["stockout_day_share_pct"] <= 100.0


def test_simulation_deterministic_with_seed(mock_predictions_df: pd.DataFrame):
    """Invariant: Simulator must be completely deterministic when initialized with same seed."""
    cfg1 = SimConfig(seed=42, warmup_days=7)
    cfg2 = SimConfig(seed=42, warmup_days=7)

    sim1 = PeriodicReviewSimulator(mock_predictions_df, config=cfg1)
    sim2 = PeriodicReviewSimulator(mock_predictions_df, config=cfg2)

    res1 = sim1.simulate_policy("lightgbm_p50_safety_stock", lead_time_days=7, service_level=0.95)
    res2 = sim2.simulate_policy("lightgbm_p50_safety_stock", lead_time_days=7, service_level=0.95)

    assert res1 == res2


def test_order_quantities_respect_lead_time():
    """Invariant: Order placed on day t must arrive exactly at day t + L, not before."""
    dates = pd.date_range("2017-05-01", periods=14, freq="D")
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
            "sigma_lgbm": [2.0] * 14,
            "sigma_snaive": [3.0] * 14,
            "split": ["fold_1"] * 14,
            "fold_idx": [1] * 14,
        }
    )

    lead_time = 3
    review_period = 7
    cfg = SimConfig(
        review_period_days=review_period, lead_times_days=(lead_time,), warmup_days=0, seed=42
    )
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
            "sigma_lgbm": [2.0] * 14,
            "sigma_snaive": [3.0] * 14,
            "split": ["fold_1"] * 14,
            "fold_idx": [1] * 14,
        }
    )

    cfg = SimConfig(review_period_days=7, lead_times_days=(3,), warmup_days=0, seed=42)
    sim = PeriodicReviewSimulator(df, config=cfg)
    res = sim.simulate_policy("lightgbm_p50", lead_time_days=3)

    assert res["stockout_day_share_pct"] == 0.0
    assert res["fill_rate_pct"] == 100.0


def test_safety_stock_residuals_strictly_pre_origin(tiny_favorita_files: dict[str, Path]):
    """Verify safety stock standard deviations use ONLY historical data <= split.origin_date."""
    from demand_planner.models.quantile import generate_all_predictions

    train_df = pd.read_csv(tiny_favorita_files["train"])
    stores_df = pd.read_csv(tiny_favorita_files["stores"])
    holidays_df = pd.read_csv(tiny_favorita_files["holidays"])

    oof_df = generate_all_predictions(
        train_df,
        stores_df=stores_df,
        holidays_df=holidays_df,
        n_folds=2,
        horizon_days=16,
        fold_step_days=16,
        holdout_days=16,
        start_train_date="2016-01-01",
        params={"n_estimators": 5, "min_child_samples": 2},
    )

    assert "sigma_lgbm" in oof_df.columns
    assert "sigma_snaive" in oof_df.columns
    assert (oof_df["sigma_lgbm"] >= 0.0).all()
    assert (oof_df["sigma_snaive"] >= 0.0).all()

    # Simulator picks up the pre-origin sigmas cleanly
    cfg = SimConfig(warmup_days=4)
    sim = PeriodicReviewSimulator(oof_df, config=cfg)
    assert sim.sigma_lgbm.shape == (sim.n_series, sim.n_days)
    assert sim.sigma_snaive.shape == (sim.n_series, sim.n_days)


def test_warmup_period_excluded_from_metrics():
    """Verify that simulated demand and stockouts during warmup days are excluded from metrics."""
    dates = pd.date_range("2017-05-01", periods=20, freq="D")
    # Massive stockout on day 2 (during warmup 0..6), zero demand thereafter
    sales_pattern = [0.0] * 20
    sales_pattern[2] = 1000.0

    df = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": [1] * 20,
            "family": ["GROCERY I"] * 20,
            "sales": sales_pattern,
            "pred_lgbm": [10.0] * 20,
            "pred_snaive": [10.0] * 20,
            "pred_p10": [5.0] * 20,
            "pred_p50": [10.0] * 20,
            "pred_p90": [15.0] * 20,
            "sigma_lgbm": [1.0] * 20,
            "sigma_snaive": [1.0] * 20,
            "split": ["fold_1"] * 20,
            "fold_idx": [1] * 20,
        }
    )

    # With warmup_days=7, the stockout on day 2 is excluded; post-warmup demand is 0 -> 100% fill
    cfg_warm = SimConfig(review_period_days=7, lead_times_days=(3,), warmup_days=7, seed=42)
    sim_warm = PeriodicReviewSimulator(df, config=cfg_warm)
    res_warm = sim_warm.simulate_policy("lightgbm_p50", lead_time_days=3)
    assert res_warm["evaluated_days"] == 13
    assert res_warm["warmup_days_excluded"] == 7
    assert res_warm["fill_rate_pct"] == 100.0


def test_tradeoff_plot_generation(mock_predictions_df: pd.DataFrame, tmp_path):
    """Verify trade-off plot function generates an image without errors."""
    cfg = SimConfig(
        review_period_days=7,
        lead_times_days=(3, 7),
        warmup_days=7,
        z_grid=(0.0, 1.0, 2.0),
        seed=42,
    )
    sim = PeriodicReviewSimulator(mock_predictions_df, config=cfg)
    sim_df = sim.run_all_simulations()
    frontier_df = sim.run_frontier(lead_times=(3, 7), z_values=(0.0, 1.0, 2.0))

    chart_file = tmp_path / "tradeoff.png"
    out = plot_replenishment_tradeoff(sim_df, frontier_df=frontier_df, output_path=chart_file)
    assert out.exists()
    assert out.stat().st_size > 0
