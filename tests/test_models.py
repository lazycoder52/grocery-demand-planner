"""Unit tests for feature builder and LightGBM forecasting model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from demand_planner.evaluation.backtest import assert_no_temporal_leakage
from demand_planner.features.build import CATEGORICAL_COLUMNS, build_features
from demand_planner.models.lightgbm import (
    evaluate_model_vs_baselines,
    predict_lightgbm,
    run_rolling_origin_cv,
    train_lightgbm,
)


def test_build_features_schema_and_types(tiny_favorita_files: dict[str, Path]):
    """Verify feature builder produces expected columns and categories without leakage."""
    train_df = pd.read_csv(tiny_favorita_files["train"])
    stores_df = pd.read_csv(tiny_favorita_files["stores"])
    holidays_df = pd.read_csv(tiny_favorita_files["holidays"])

    feat_df, cols = build_features(train_df, stores_df=stores_df, holidays_df=holidays_df)

    # Core causal columns
    assert "lag_16" in cols
    assert "lag_21" in cols
    assert "lag_28" in cols
    assert "rolling_mean_7_lag16" in cols
    assert "rolling_mean_28_lag16" in cols
    assert "is_payday" in cols
    assert "dayofweek" in cols
    assert "is_holiday" in cols

    # Verify categorical types
    for cat in CATEGORICAL_COLUMNS:
        assert isinstance(feat_df[cat].dtype, pd.CategoricalDtype)

    # Verify no negative values in non-negative metrics
    assert (feat_df["lag_16"] >= 0).all()
    assert (feat_df["rolling_mean_7_lag16"] >= 0).all()


def test_lightgbm_training_and_inference():
    """Verify LightGBM model fits and predicts non-negative demand values."""
    rng = np.random.default_rng(42)
    n = 100
    X_train = pd.DataFrame(
        {
            "store_nbr": pd.Categorical([1] * n),
            "family": pd.Categorical(["GROCERY I"] * n),
            "store_type": pd.Categorical(["D"] * n),
            "cluster": pd.Categorical([13] * n),
            "dayofweek": rng.integers(0, 7, size=n),
            "day": rng.integers(1, 29, size=n),
            "month": [5] * n,
            "is_weekend": [0] * n,
            "is_payday": [0] * n,
            "is_holiday": [0] * n,
            "onpromotion": rng.integers(0, 5, size=n),
            "has_promotion": [0] * n,
            "lag_16": rng.uniform(10, 50, size=n),
            "lag_21": rng.uniform(10, 50, size=n),
            "lag_28": rng.uniform(10, 50, size=n),
            "rolling_mean_7_lag16": rng.uniform(10, 50, size=n),
            "rolling_mean_28_lag16": rng.uniform(10, 50, size=n),
        }
    )
    y_train = np.log1p(rng.uniform(10, 50, size=n))

    model = train_lightgbm(X_train, y_train, params={"n_estimators": 10, "min_child_samples": 2})
    preds = predict_lightgbm(model, X_train)

    assert len(preds) == n
    assert (preds >= 0.0).all()
    assert not np.isnan(preds).any()


def test_evaluate_model_vs_baselines_end_to_end(tiny_favorita_files: dict[str, Path]):
    """Verify full evaluation pipeline runs against tiny Favorita fixture."""
    train_df = pd.read_csv(tiny_favorita_files["train"])
    stores_df = pd.read_csv(tiny_favorita_files["stores"])
    holidays_df = pd.read_csv(tiny_favorita_files["holidays"])

    train_df["date"] = pd.to_datetime(train_df["date"])
    max_d = train_df["date"].max()
    holdout_origin = max_d - pd.Timedelta(days=16)

    summary, test_preds = evaluate_model_vs_baselines(
        train_df,
        holdout_origin=holdout_origin,
        stores_df=stores_df,
        holidays_df=holidays_df,
        start_train_date="2016-01-01",
        params={"n_estimators": 15, "min_child_samples": 5},
    )

    # Verify structure
    assert "models" in summary
    assert "lightgbm" in summary["models"]
    assert "seasonal_naive_7d" in summary["models"]
    assert "naive" in summary["models"]
    assert "moving_average_28d" in summary["models"]
    assert "forecast_value_added_vs_seasonal_naive" in summary
    assert "by_family" in summary

    # Verify prediction columns
    assert "pred_lgbm" in test_preds.columns
    assert "pred_naive" in test_preds.columns
    assert "pred_seasonal_naive" in test_preds.columns
    assert "pred_ma28" in test_preds.columns

    # Verify zero leakage
    train_split = train_df[train_df["date"] <= holdout_origin]
    assert_no_temporal_leakage(train_split, test_preds, holdout_origin)


def test_run_rolling_origin_cv_end_to_end(tiny_favorita_files: dict[str, Path]):
    """Verify rolling-origin CV runs across folds against tiny Favorita fixture."""
    train_df = pd.read_csv(tiny_favorita_files["train"])
    stores_df = pd.read_csv(tiny_favorita_files["stores"])
    holidays_df = pd.read_csv(tiny_favorita_files["holidays"])

    cv_results = run_rolling_origin_cv(
        train_df,
        stores_df=stores_df,
        holidays_df=holidays_df,
        n_folds=3,
        horizon_days=16,
        fold_step_days=16,
        holdout_days=16,
        start_train_date="2016-01-01",
        params={"n_estimators": 10, "min_child_samples": 5},
    )

    assert cv_results["n_folds"] == 3
    assert len(cv_results["folds"]) == 3
    assert "mean_metrics" in cv_results
    assert "lightgbm" in cv_results["mean_metrics"]
    assert "seasonal_naive_7d" in cv_results["mean_metrics"]
    assert cv_results["mean_metrics"]["lightgbm"]["mean_wape"] > 0.0
