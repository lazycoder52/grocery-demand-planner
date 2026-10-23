"""Tests for evaluation metrics, heuristic baselines, and temporal leakage prevention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand_planner.evaluation.backtest import (
    assert_no_temporal_leakage,
    evaluate_baselines_on_split,
    generate_rolling_folds,
    run_baseline_benchmarks,
)
from demand_planner.evaluation.baselines import (
    predict_moving_average,
    predict_naive,
    predict_seasonal_naive,
)
from demand_planner.evaluation.metrics import (
    calc_bias,
    calc_mae,
    calc_rmsle,
    calc_wape,
    evaluate_predictions,
)


def test_metric_calculations_exact():
    """Verify precision of WAPE, bias, MAE, and RMSLE."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    # Absolute errors: 10 + 10 + 0 = 20. Total actual = 600. WAPE = 20/600 * 100 = 3.3333%
    assert abs(calc_wape(y_true, y_pred) - 3.3333) < 0.01

    # Errors: +10 + (-10) + 0 = 0. Bias % = 0.0%
    assert calc_bias(y_true, y_pred) == 0.0

    # MAE = (10 + 10 + 0) / 3 = 6.6667
    assert abs(calc_mae(y_true, y_pred) - 6.6667) < 0.01

    # RMSLE check
    assert calc_rmsle(y_true, y_pred) > 0.0

    # Over-forecasting bias check
    y_pred_high = np.array([120.0, 220.0, 320.0])
    assert calc_bias(y_true, y_pred_high) > 0.0

    # Under-forecasting bias check
    y_pred_low = np.array([80.0, 180.0, 280.0])
    assert calc_bias(y_true, y_pred_low) < 0.0

    # Zero denominator edge case
    assert calc_wape(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0
    assert calc_wape(np.array([0.0, 0.0]), np.array([5.0, 5.0])) == 100.0


def test_evaluate_predictions_grouped():
    """Verify evaluate_predictions computes overall and family-level breakdowns."""
    df = pd.DataFrame(
        {
            "family": ["BEVERAGES", "BEVERAGES", "MEATS", "MEATS"],
            "sales": [10.0, 20.0, 50.0, 50.0],
            "pred": [12.0, 18.0, 40.0, 60.0],
        }
    )
    res = evaluate_predictions(df, true_col="sales", pred_col="pred", group_cols=["family"])
    assert "overall" in res
    assert "by_family" in res
    assert "BEVERAGES" in res["by_family"]
    assert "MEATS" in res["by_family"]
    assert res["overall"]["total_actual_sales"] == 130.0


def test_temporal_leakage_assertion_catches_violations():
    """Verify assert_no_temporal_leakage raises ValueError on any post-origin training data."""
    origin = pd.Timestamp("2017-06-01")

    # Clean setup
    train_clean = pd.DataFrame({"date": pd.date_range("2017-05-01", "2017-06-01", freq="D")})
    test_clean = pd.DataFrame({"date": pd.date_range("2017-06-02", "2017-06-17", freq="D")})
    assert_no_temporal_leakage(train_clean, test_clean, origin)

    # 1. Leakage in training: training contains date > origin
    train_leaky = pd.DataFrame({"date": pd.date_range("2017-05-01", "2017-06-05", freq="D")})
    with pytest.raises(ValueError, match="Temporal leakage detected in training set"):
        assert_no_temporal_leakage(train_leaky, test_clean, origin)

    # 2. Leakage in test: test contains date <= origin
    test_leaky = pd.DataFrame({"date": pd.date_range("2017-06-01", "2017-06-17", freq="D")})
    with pytest.raises(ValueError, match="Temporal leakage detected in test set"):
        assert_no_temporal_leakage(train_clean, test_leaky, origin)

    # 3. Leaky feature column (e.g. same-day transactions)
    with pytest.raises(ValueError, match="prohibited post-origin columns"):
        assert_no_temporal_leakage(
            train_clean, test_clean, origin, feature_cols=["dayofweek", "transactions"]
        )


def test_heuristic_baselines():
    """Verify naive, seasonal naive, and 28-day moving average on synthetic series."""
    dates_train = pd.date_range("2017-01-01", "2017-02-15", freq="D")
    dates_test = pd.date_range("2017-02-16", "2017-02-22", freq="D")

    # Create pattern: 10 * (day of week + 1)
    train_rows = []
    for d in dates_train:
        train_rows.append(
            {
                "date": d,
                "store_nbr": 1,
                "family": "GROCERY I",
                "sales": float(10 * (d.dayofweek + 1)),
            }
        )
    train_df = pd.DataFrame(train_rows)

    test_rows = []
    for d in dates_test:
        test_rows.append(
            {
                "date": d,
                "store_nbr": 1,
                "family": "GROCERY I",
                "sales": float(10 * (d.dayofweek + 1)),
            }
        )
    test_df = pd.DataFrame(test_rows)

    # Naive: predicts the last value of train_df (2017-02-15 is Wed -> dow=2 -> sales=30)
    res_naive = predict_naive(train_df, test_df)
    assert (res_naive["pred_naive"] == 30.0).all()

    # Seasonal Naive: should exactly match the day-of-week pattern
    res_snaive = predict_seasonal_naive(train_df, test_df)
    assert np.allclose(res_snaive["pred_seasonal_naive"], test_df["sales"])

    # 28-Day MA: mean over last 28 days (4 complete weeks of 10..70 -> mean is 40)
    res_ma28 = predict_moving_average(train_df, test_df, window_days=28)
    assert np.allclose(res_ma28["pred_ma28"], 40.0)


def test_rolling_folds_generation(tiny_favorita_files: dict[str, Path]):
    """Verify rolling folds generator creates expanding windows with zero leakage."""
    train_df = pd.read_csv(tiny_favorita_files["train"])

    cv_folds, holdout = generate_rolling_folds(
        train_df,
        n_folds=5,
        horizon_days=16,
        fold_step_days=16,
        holdout_days=16,
    )

    assert len(cv_folds) == 5
    assert holdout is not None

    # Verify expanding window property
    for i in range(len(cv_folds) - 1):
        assert len(cv_folds[i].train_df) < len(cv_folds[i + 1].train_df)
        assert cv_folds[i].origin_date < cv_folds[i + 1].origin_date

    # Verify holdout origin is newer than all CV origins
    assert holdout.origin_date > cv_folds[-1].origin_date

    # Verify evaluation benchmarks run cleanly
    bench_results = evaluate_baselines_on_split(holdout)
    assert "naive" in bench_results
    assert "seasonal_naive_7d" in bench_results
    assert "moving_average_28d" in bench_results
    assert bench_results["seasonal_naive_7d"]["overall"]["wape"] > 0.0


def test_run_baseline_benchmarks_end_to_end(tiny_favorita_files: dict[str, Path]):
    """Verify run_baseline_benchmarks executes all folds against tiny Favorita fixture."""
    train_df = pd.read_csv(tiny_favorita_files["train"])
    results = run_baseline_benchmarks(train_df)

    assert "holdout" in results
    assert "cv_folds" in results
    assert len(results["cv_folds"]) == 5
    for fold in results["cv_folds"]:
        assert "naive" in fold["results"]
        assert "seasonal_naive_7d" in fold["results"]
        assert "moving_average_28d" in fold["results"]
