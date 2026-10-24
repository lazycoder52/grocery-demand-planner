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

    # Core temporal feature columns
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


def test_lightgbm_quantile_training_and_monotonicity():
    """Verify LightGBM quantile regression fits P10, P50, P90 and enforces P10 <= P50 <= P90."""
    from demand_planner.models.quantile import (
        evaluate_quantiles,
        predict_lightgbm_quantiles,
        train_lightgbm_quantiles,
    )

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

    models = train_lightgbm_quantiles(
        X_train,
        y_train,
        alphas=(0.1, 0.5, 0.9),
        params={"n_estimators": 10, "min_child_samples": 2},
    )
    assert set(models.keys()) == {0.1, 0.5, 0.9}

    preds = predict_lightgbm_quantiles(models, X_train, enforce_monotonicity=True)
    assert "p10" in preds and "p50" in preds and "p90" in preds
    assert len(preds["p10"]) == n

    # Monotonicity check
    assert (preds["p10"] <= preds["p50"]).all()
    assert (preds["p50"] <= preds["p90"]).all()
    assert (preds["p10"] >= 0.0).all()

    # Evaluation check
    y_actual = np.expm1(y_train)
    eval_res = evaluate_quantiles(y_actual, preds)
    assert "pinball_loss_p10" in eval_res
    assert "pinball_loss_p50" in eval_res
    assert "pinball_loss_p90" in eval_res
    assert "coverage_p10_p90_pct" in eval_res
    assert 0.0 <= eval_res["coverage_p10_p90_pct"] <= 100.0


def test_quantile_cv_and_holdout_end_to_end(tiny_favorita_files: dict[str, Path]):
    """Verify quantile CV and holdout pipelines run against tiny Favorita fixture."""
    from demand_planner.models.quantile import evaluate_quantile_holdout, run_quantile_cv

    train_df = pd.read_csv(tiny_favorita_files["train"])
    stores_df = pd.read_csv(tiny_favorita_files["stores"])
    holidays_df = pd.read_csv(tiny_favorita_files["holidays"])

    train_df["date"] = pd.to_datetime(train_df["date"])
    max_d = train_df["date"].max()
    holdout_origin = max_d - pd.Timedelta(days=16)

    # 1. Quantile Holdout
    h_metrics, h_preds = evaluate_quantile_holdout(
        train_df,
        holdout_origin=holdout_origin,
        stores_df=stores_df,
        holidays_df=holidays_df,
        start_train_date="2016-01-01",
        params={"n_estimators": 10, "min_child_samples": 5},
    )
    assert "coverage_p10_p90_pct" in h_metrics
    assert "mean_pinball_loss" in h_metrics
    assert "pred_p10" in h_preds.columns
    assert "pred_p50" in h_preds.columns
    assert "pred_p90" in h_preds.columns
    assert (h_preds["pred_p10"] <= h_preds["pred_p50"]).all()
    assert (h_preds["pred_p50"] <= h_preds["pred_p90"]).all()

    # 2. Quantile CV (3 folds)
    cv_res = run_quantile_cv(
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
    assert cv_res["n_folds"] == 3
    assert len(cv_res["folds"]) == 3
    assert "mean_metrics" in cv_res
    assert cv_res["mean_metrics"]["mean_coverage_p10_p90_pct"] > 0.0
    assert cv_res["mean_metrics"]["mean_pinball_loss"] > 0.0


def test_get_store_opening_dates_and_preopening_exclusion():
    """Verify store opening dates identify first positive sale and exclude pre-opening rows."""
    from demand_planner.models.lightgbm import get_store_opening_dates

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"]),
            "store_nbr": [1, 1, 2, 2],
            "sales": [0.0, 50.0, 0.0, 100.0],
        }
    )

    open_dates = get_store_opening_dates(df)
    assert open_dates[1] == pd.Timestamp("2017-01-02")
    assert open_dates[2] == pd.Timestamp("2017-01-04")


def test_generate_all_predictions_and_oof_export(tiny_favorita_files: dict[str, Path]):
    """Verify generate_all_predictions generates valid out-of-fold and holdout predictions."""
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
        params={"n_estimators": 10, "min_child_samples": 5},
    )

    expected_cols = {
        "date",
        "store_nbr",
        "family",
        "sales",
        "pred_lgbm",
        "pred_snaive",
        "pred_p10",
        "pred_p50",
        "pred_p90",
        "split",
        "fold_idx",
    }
    assert expected_cols.issubset(set(oof_df.columns))
    assert (oof_df["pred_lgbm"] >= 0.0).all()
    assert (oof_df["pred_p10"] <= oof_df["pred_p50"]).all()
    assert (oof_df["pred_p50"] <= oof_df["pred_p90"]).all()
    assert set(oof_df["split"].unique()) == {"fold_1", "fold_2", "holdout"}

    # If the output file exists, verify CI constraints (< 5 MB)
    output_file = Path("outputs/oof_and_holdout_predictions.parquet")
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        assert size_mb < 5.0, f"oof_and_holdout_predictions.parquet exceeds 5 MB: {size_mb:.2f} MB"
        output_df = pd.read_parquet(output_file)
        assert len(output_df) == 171072
        assert (output_df["pred_p10"] <= output_df["pred_p50"]).all()
        assert (output_df["pred_p50"] <= output_df["pred_p90"]).all()
