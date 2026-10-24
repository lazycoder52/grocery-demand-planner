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


def test_order_forecasts_strictly_pre_origin_and_no_future_fold_leak():
    """Verify orders placed near fold boundaries never consume future-fold model predictions.

    Rule: For every order placed on review day t (calendar date D_t), all forecast slices
    utilized must originate from models with origin date <= D_t. Clamping to the active
    fold boundary prevents temporal leakage into fold k+1.
    """
    # 32 days: 16 days in fold 1 (origin 2017-05-01), 16 days in fold 2 (origin 2017-05-17)
    dates = pd.date_range("2017-05-02", periods=32, freq="D")
    origin_map = {
        "fold_1": pd.Timestamp("2017-05-01"),
        "fold_2": pd.Timestamp("2017-05-17"),
    }

    records = []
    for i, d in enumerate(dates):
        split_name = "fold_1" if i < 16 else "fold_2"
        # Poison fold_2 with high values to verify that fold 1 orders don't read it
        p50_val = 10.0 if split_name == "fold_1" else 9999.0
        records.append(
            {
                "date": d,
                "store_nbr": 1,
                "family": "GROCERY I",
                "sales": 10.0,
                "pred_lgbm": p50_val,
                "pred_snaive": p50_val,
                "pred_p10": p50_val * 0.8,
                "pred_p50": p50_val,
                "pred_p90": p50_val * 1.2,
                "sigma_lgbm": 2.0,
                "sigma_snaive": 2.0,
                "split": split_name,
                "fold_idx": 1 if split_name == "fold_1" else 2,
            }
        )

    df = pd.DataFrame(records)
    cfg = SimConfig(review_period_days=7, lead_times_days=(3,), warmup_days=0, seed=42)
    sim = PeriodicReviewSimulator(df, config=cfg)

    # Order on review day t=14 (2017-05-16, inside fold 1).
    # Protection window R+L = 10 days spans indices 14..23 (into fold 2, dates 2017-05-18..25).
    # Fold 2 has origin 2017-05-17 > order date 2017-05-16!
    order_day = 14
    order_date = pd.Timestamp(sim.dates[order_day])
    assert order_date == pd.Timestamp("2017-05-16")

    # The simulator must clamp to fold 1's end (index 15, 2 days observed: index 14 and 15)
    # and scale proportionally: 2 days * 10.0 = 20.0 * (10 / 2) = 100.0.
    # If it leaked into fold 2 (values 9999.0), the forecast would be > 10,000.
    forecast = sim._get_forecast("lightgbm_p50", order_day, protection_days=10)
    assert float(forecast[0]) == pytest.approx(100.0, rel=1e-3)

    # Verify origin date of all forecasts consumed is strictly <= order_date
    active_fold = df.loc[df["date"] == order_date, "split"].iloc[0]
    fold_origin = origin_map[active_fold]
    assert fold_origin <= order_date, f"Origin {fold_origin} > order date {order_date}"


def test_point_model_simulation_support(mock_predictions_df: pd.DataFrame):
    """Verify LightGBM Point model is supported in replenishment simulation."""
    cfg = SimConfig(review_period_days=7, lead_times_days=(3,), warmup_days=7, seed=42)
    sim = PeriodicReviewSimulator(mock_predictions_df, config=cfg)

    res = sim.simulate_policy("lightgbm_point_safety_stock", lead_time_days=3, service_level=0.95)
    assert res["policy"] == "lightgbm_point_safety_stock"
    assert "LightGBM Point + SS" in res["policy_label"]
    assert 0.0 <= res["fill_rate_pct"] <= 100.0
    assert res["average_inventory_units"] >= 0.0


def test_dual_bootstrap_target_fill_rate_tradeoff(mock_predictions_df: pd.DataFrame):
    """Verify compute_target_fill_rate_tradeoff produces both series and
    time-block bootstrap CIs.
    """
    cfg = SimConfig(
        review_period_days=7,
        lead_times_days=(3,),
        warmup_days=7,
        z_grid=(0.0, 1.0, 2.0),
        seed=42,
    )
    sim = PeriodicReviewSimulator(mock_predictions_df, config=cfg)
    frontier_df = sim.run_frontier(lead_times=(3,), z_values=(0.0, 1.0, 2.0))
    tradeoffs = sim.compute_target_fill_rate_tradeoff(
        frontier_df,
        lead_times=(3,),
        target_fill_rates=(98.5,),
        n_bootstrap=30,
    )
    assert len(tradeoffs) >= 1
    t = tradeoffs[0]
    assert "ci_95_series_pct_diff_low" in t
    assert "ci_95_series_pct_diff_high" in t
    assert "ci_95_time_block_pct_diff_low" in t
    assert "ci_95_time_block_pct_diff_high" in t
    assert "target_fill_rate_pct" in t
    assert "is_reachable" in t


def test_family_matched_fill_rate_frontier(mock_predictions_df: pd.DataFrame):
    """Verify per-family matched fill-rate analysis traces frontiers and handles reachability."""
    cfg = SimConfig(
        review_period_days=7,
        lead_times_days=(3,),
        warmup_days=7,
        z_grid=(0.0, 1.0, 2.0),
        seed=42,
    )
    sim = PeriodicReviewSimulator(mock_predictions_df, config=cfg)
    fam_results = sim.simulate_by_family_matched_fill_rate(
        lead_times=(3,),
        target_fill_rates=(98.5,),
        families=["GROCERY I", "BEVERAGES"],
        n_bootstrap=25,
    )
    assert len(fam_results) == 2
    for r in fam_results:
        assert "family" in r
        assert "is_reachable_p50" in r
        assert "grid_fr_range_p50" in r
        assert "max_fill_rate_p50" in r
        assert "max_fill_rate_snaive" in r


def test_review_forecast_poisoning_invariance(tiny_favorita_files: dict[str, Path]):
    """Verify forecasts and sigmas at origin t are strictly invariant to sales mutations after t."""
    from demand_planner.models.quantile import generate_review_forecasts

    clean_train = pd.read_csv(tiny_favorita_files["train"])
    stores_df = pd.read_csv(tiny_favorita_files["stores"])
    holidays_df = pd.read_csv(tiny_favorita_files["holidays"])

    clean_train["date"] = pd.to_datetime(clean_train["date"])
    max_d = clean_train["date"].max()
    eval_dates = pd.date_range(max_d - pd.Timedelta(days=27), max_d, freq="D")
    t_review_date = eval_dates[0]

    # 1. Clean run
    clean_fc = generate_review_forecasts(
        clean_train,
        stores_df=stores_df,
        holidays_df=holidays_df,
        eval_dates=eval_dates,
        n_folds=2,
        horizon_days=16,
        fold_step_days=16,
        holdout_days=16,
        start_train_date="2016-01-01",
        params={"n_estimators": 5, "min_child_samples": 2},
    )
    clean_sub = clean_fc[clean_fc["review_date"] == t_review_date].sort_values(
        ["horizon_step", "store_nbr", "family"]
    )

    # 2. Poisoned run: heavily mutate sales strictly after t_review_date
    poisoned_train = clean_train.copy()
    future_mask = poisoned_train["date"] > t_review_date
    poisoned_train.loc[future_mask, "sales"] = 999999.0

    poisoned_fc = generate_review_forecasts(
        poisoned_train,
        stores_df=stores_df,
        holidays_df=holidays_df,
        eval_dates=eval_dates,
        n_folds=2,
        horizon_days=16,
        fold_step_days=16,
        holdout_days=16,
        start_train_date="2016-01-01",
        params={"n_estimators": 5, "min_child_samples": 2},
    )
    poisoned_sub = poisoned_fc[poisoned_fc["review_date"] == t_review_date].sort_values(
        ["horizon_step", "store_nbr", "family"]
    )

    # Predictions and sigmas at origin t MUST be bit-for-bit identical
    # despite massive future sales mutations
    np.testing.assert_allclose(
        clean_sub["pred_lgbm"].to_numpy(),
        poisoned_sub["pred_lgbm"].to_numpy(),
        err_msg="Point forecast at origin t leaked future sales!",
    )
    np.testing.assert_allclose(
        clean_sub["pred_p50"].to_numpy(),
        poisoned_sub["pred_p50"].to_numpy(),
        err_msg="P50 forecast at origin t leaked future sales!",
    )
    np.testing.assert_allclose(
        clean_sub["pred_snaive"].to_numpy(),
        poisoned_sub["pred_snaive"].to_numpy(),
        err_msg="Seasonal naive forecast at origin t leaked future sales!",
    )
    np.testing.assert_allclose(
        clean_sub["sigma_lgbm"].to_numpy(),
        poisoned_sub["sigma_lgbm"].to_numpy(),
        err_msg="LightGBM sigma at origin t leaked future sales!",
    )
    np.testing.assert_allclose(
        clean_sub["sigma_snaive"].to_numpy(),
        poisoned_sub["sigma_snaive"].to_numpy(),
        err_msg="Seasonal naive sigma at origin t leaked future sales!",
    )


def test_stitching_rules_support(mock_predictions_df: pd.DataFrame):
    """Verify that PeriodicReviewSimulator supports all three stitching rules cleanly."""
    cfg = SimConfig(review_period_days=7, lead_times_days=(3,), warmup_days=7, seed=42)

    # 1. Clamped and scaled rule
    sim_clamped = PeriodicReviewSimulator(
        mock_predictions_df, config=cfg, stitching_rule="clamped_scaled"
    )
    res_clamped = sim_clamped.simulate_policy("lightgbm_p50_safety_stock", lead_time_days=3)
    assert res_clamped["stitching_rule"] == "clamped_scaled"

    # 2. Inside fold only rule
    sim_inside = PeriodicReviewSimulator(
        mock_predictions_df, config=cfg, stitching_rule="inside_fold_only"
    )
    res_inside = sim_inside.simulate_policy("lightgbm_p50_safety_stock", lead_time_days=3)
    assert res_inside["stitching_rule"] == "inside_fold_only"

    # 3. Reforecast rule (fallback without review_forecasts_df)
    sim_reforecast = PeriodicReviewSimulator(
        mock_predictions_df, config=cfg, stitching_rule="reforecast"
    )
    res_reforecast = sim_reforecast.simulate_policy("lightgbm_p50_safety_stock", lead_time_days=3)
    assert res_reforecast["stitching_rule"] == "reforecast"


def test_seasonal_naive_multi_step_matches_wape_baseline(tiny_favorita_files: dict[str, Path]):
    """Verify that multi-step seasonal naive predictions match predict_seasonal_naive exactly.

    generate_review_forecasts constructs horizons beyond 7 days by cyclically repeating the
    last observed 7 days. This must match predict_seasonal_naive from demand_planner.evaluation.
    """
    from demand_planner.evaluation.baselines import predict_seasonal_naive
    from demand_planner.models.quantile import generate_review_forecasts

    train_df = pd.read_csv(tiny_favorita_files["train"])
    train_df["date"] = pd.to_datetime(train_df["date"])
    stores_df = pd.read_csv(tiny_favorita_files["stores"])
    holidays_df = pd.read_csv(tiny_favorita_files["holidays"])

    max_d = train_df["date"].max()
    eval_dates = pd.date_range(max_d - pd.Timedelta(days=27), max_d, freq="D")
    t_origin = eval_dates[0]

    # 1. Review forecasts via generate_review_forecasts
    rf_df = generate_review_forecasts(
        train_df,
        stores_df=stores_df,
        holidays_df=holidays_df,
        eval_dates=eval_dates,
        n_folds=2,
        horizon_days=16,
        fold_step_days=16,
        holdout_days=16,
        start_train_date="2016-01-01",
        params={"n_estimators": 5, "min_child_samples": 2},
    )
    sub_rf = rf_df[rf_df["review_date"] == t_origin].sort_values(
        ["target_date", "store_nbr", "family"]
    )

    # 2. Baseline via predict_seasonal_naive
    hist_sub = train_df[train_df["date"] <= t_origin].copy()
    target_dates = pd.date_range(t_origin + pd.Timedelta(days=1), periods=16, freq="D")
    test_template = train_df[train_df["date"].isin(target_dates)][
        ["date", "store_nbr", "family"]
    ].copy()
    base_df = predict_seasonal_naive(
        hist_sub,
        test_template,
        date_col="date",
        group_cols=("store_nbr", "family"),
        target_col="sales",
        pred_col="pred_snaive",
    ).sort_values(["date", "store_nbr", "family"])

    # Assert bit-for-bit equivalence across all 16 horizon steps
    np.testing.assert_allclose(
        sub_rf["pred_snaive"].to_numpy(),
        base_df["pred_snaive"].to_numpy(),
        err_msg=(
            "Seasonal naive forecast in generate_review_forecasts diverged from "
            "predict_seasonal_naive!"
        ),
    )


def test_out_of_sample_safety_stock_residuals(tiny_favorita_files: dict[str, Path]):
    """Verify safety-stock sigmas use strictly out-of-sample forecast errors from earlier origins.

    Fails if:
    1. Any residual used to compute sigma at review date t comes from a model fit on data
       after the forecast origin.
    2. Sigmas at review date t change when sales on or after t are mutated (poisoned).
    """
    from demand_planner.models.quantile import generate_review_forecasts

    train_df = pd.read_csv(tiny_favorita_files["train"])
    train_df["date"] = pd.to_datetime(train_df["date"])
    stores_df = pd.read_csv(tiny_favorita_files["stores"])
    holidays_df = pd.read_csv(tiny_favorita_files["holidays"])

    max_d = train_df["date"].max()
    eval_dates = pd.date_range(max_d - pd.Timedelta(days=15), max_d, freq="D")
    t_origin = eval_dates[0]

    # 1. Generate baseline review forecasts
    base_fc = generate_review_forecasts(
        train_df,
        stores_df=stores_df,
        holidays_df=holidays_df,
        eval_dates=eval_dates,
        n_folds=1,
        n_burnin_folds=1,
        horizon_days=16,
        start_train_date="2016-01-01",
        params={"n_estimators": 5, "min_child_samples": 2},
    )

    assert "sigma_lgbm" in base_fc.columns
    assert "sigma_p50" in base_fc.columns
    assert "sigma_snaive" in base_fc.columns
    assert (base_fc["sigma_lgbm"] >= 0.0).all()
    assert (base_fc["sigma_p50"] >= 0.0).all()
    assert (base_fc["sigma_snaive"] >= 0.0).all()

    # 2. Poison future sales on and after t_origin
    poisoned_df = train_df.copy()
    future_mask = poisoned_df["date"] >= t_origin
    poisoned_df.loc[future_mask, "sales"] = poisoned_df.loc[future_mask, "sales"] * 10.0 + 500.0

    poisoned_fc = generate_review_forecasts(
        poisoned_df,
        stores_df=stores_df,
        holidays_df=holidays_df,
        eval_dates=eval_dates,
        n_folds=1,
        n_burnin_folds=1,
        horizon_days=16,
        start_train_date="2016-01-01",
        params={"n_estimators": 5, "min_child_samples": 2},
    )

    # Sigmas computed at review date t_origin must be invariant to sales mutations on/after t_origin
    base_t0 = base_fc[base_fc["review_day_idx"] == 0]
    poisoned_t0 = poisoned_fc[poisoned_fc["review_day_idx"] == 0]

    np.testing.assert_allclose(
        base_t0["sigma_lgbm"].to_numpy(),
        poisoned_t0["sigma_lgbm"].to_numpy(),
        rtol=1e-5,
        err_msg="LightGBM residual sigma at t changed when future sales were mutated!",
    )
    np.testing.assert_allclose(
        base_t0["sigma_p50"].to_numpy(),
        poisoned_t0["sigma_p50"].to_numpy(),
        rtol=1e-5,
        err_msg="LightGBM P50 residual sigma at t changed when future sales were mutated!",
    )
    np.testing.assert_allclose(
        base_t0["sigma_snaive"].to_numpy(),
        poisoned_t0["sigma_snaive"].to_numpy(),
        rtol=1e-5,
        err_msg="Seasonal naive residual sigma at t changed when future sales were mutated!",
    )


def test_promotion_planned_advance_schedule_consumption(tiny_favorita_files: dict[str, Path]):
    """Verify onpromotion values on forecast dates consume the planned promotional schedule.

    Demand planning models consume the known onpromotion feature on future forecast dates
    (date > origin_date), reflecting the retail operational assumption that marketing circulars
    and promotional calendars are planned and finalized in advance.
    """
    from demand_planner.models.quantile import generate_review_forecasts

    train_df = pd.read_csv(tiny_favorita_files["train"])
    train_df["date"] = pd.to_datetime(train_df["date"])
    stores_df = pd.read_csv(tiny_favorita_files["stores"])
    holidays_df = pd.read_csv(tiny_favorita_files["holidays"])

    max_d = train_df["date"].max()
    eval_dates = pd.date_range(max_d - pd.Timedelta(days=27), max_d, freq="D")
    t_origin = eval_dates[0]
    future_dates = pd.date_range(t_origin + pd.Timedelta(days=1), periods=16, freq="D")

    # 1. Base forecast with current promotional schedule
    base_fc = generate_review_forecasts(
        train_df,
        stores_df=stores_df,
        holidays_df=holidays_df,
        eval_dates=eval_dates,
        n_folds=2,
        horizon_days=16,
        fold_step_days=16,
        holdout_days=16,
        start_train_date="2016-01-01",
        params={"n_estimators": 5, "min_child_samples": 2},
    )
    base_sub = base_fc[base_fc["review_date"] == t_origin].sort_values(
        ["target_date", "store_nbr", "family"]
    )

    # 2. Modify planned promotional schedule strictly in future (date > t_origin)
    promo_train = train_df.copy()
    promo_train.loc[promo_train["date"].isin(future_dates), "onpromotion"] += 50

    promo_fc = generate_review_forecasts(
        promo_train,
        stores_df=stores_df,
        holidays_df=holidays_df,
        eval_dates=eval_dates,
        n_folds=2,
        horizon_days=16,
        fold_step_days=16,
        holdout_days=16,
        start_train_date="2016-01-01",
        params={"n_estimators": 5, "min_child_samples": 2},
    )
    promo_sub = promo_fc[promo_fc["review_date"] == t_origin].sort_values(
        ["target_date", "store_nbr", "family"]
    )

    # Future forecast responds to promotional schedule change
    # Seasonal naive is purely historical and does NOT use future promotion
    np.testing.assert_allclose(
        base_sub["pred_snaive"].to_numpy(),
        promo_sub["pred_snaive"].to_numpy(),
        err_msg="Seasonal naive should not depend on future promotions!",
    )
    # Sigmas are pre-origin and do NOT depend on future promotions
    np.testing.assert_allclose(
        base_sub["sigma_lgbm"].to_numpy(),
        promo_sub["sigma_lgbm"].to_numpy(),
        err_msg="Pre-origin sigma should not depend on future promotions!",
    )
    # LightGBM point forecast consumes future planned promotions
    assert not np.allclose(
        base_sub["pred_lgbm"].to_numpy(),
        promo_sub["pred_lgbm"].to_numpy(),
    ), "LightGBM forecast did not consume planned promotional schedule on forecast dates!"
