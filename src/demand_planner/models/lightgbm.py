"""Global LightGBM demand forecasting model and baseline benchmarking."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand_planner.evaluation.baselines import (
    predict_moving_average,
    predict_naive,
    predict_seasonal_naive,
)
from demand_planner.evaluation.metrics import evaluate_predictions
from demand_planner.features.build import build_features

log = logging.getLogger("demand_planner.models.lightgbm")


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    params: dict[str, Any] | None = None,
) -> lgb.LGBMRegressor:
    """Train a global LightGBM regressor on store-family tabular features."""
    default_params: dict[str, Any] = {
        "objective": "regression",
        "n_estimators": 200,
        "learning_rate": 0.08,
        "num_leaves": 63,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def predict_lightgbm(
    model: lgb.LGBMRegressor,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Generate non-negative demand forecasts from log-transformed target predictions."""
    raw_preds = model.predict(X_test)
    return np.expm1(raw_preds).clip(min=0.0)


def get_store_opening_dates(
    sales_df: pd.DataFrame,
    date_col: str = "date",
    store_col: str = "store_nbr",
    target_col: str = "sales",
) -> dict[int, pd.Timestamp]:
    """Find the earliest date with positive sales for each store."""
    pos = sales_df[sales_df[target_col] > 0]
    return pos.groupby(store_col)[date_col].min().to_dict()


def compute_daytype_metrics(
    test_df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "sales",
    model_pred_col: str = "pred_lgbm",
    baseline_pred_col: str = "pred_seasonal_naive",
) -> dict[str, Any]:
    """Compute WAPE and Bias partitioned by payday week vs other weeks and exact paydays."""
    from demand_planner.evaluation.metrics import calc_bias, calc_wape

    df = test_df.copy()
    dates = pd.to_datetime(df[date_col])
    all_dates = pd.date_range(
        dates.min() - pd.Timedelta(days=7), dates.max() + pd.Timedelta(days=7)
    )
    cal = pd.DataFrame({"date": all_dates})
    cal["week"] = cal["date"].dt.to_period("W-SUN")
    cal["is_payday"] = (cal["date"].dt.day == 15) | (cal["date"].dt.is_month_end)
    payday_weeks = set(cal.groupby("week")["is_payday"].any()[lambda x: x].index)

    df["_week"] = dates.dt.to_period("W-SUN")
    df["is_payday_week"] = df["_week"].isin(payday_weeks)
    df["is_exact_payday"] = (dates.dt.day == 15) | (dates.dt.is_month_end)

    results: dict[str, Any] = {}
    for flag_col, key in [
        ("is_payday_week", "by_payday_week"),
        ("is_exact_payday", "by_exact_payday"),
    ]:
        regimes: dict[str, Any] = {}
        for val, grp in df.groupby(flag_col):
            label = "payday_regime" if val else "other_regime"
            y_true = grp[target_col].to_numpy()
            y_mod = grp[model_pred_col].to_numpy()
            y_base = grp[baseline_pred_col].to_numpy()
            w_mod = calc_wape(y_true, y_mod)
            w_base = calc_wape(y_true, y_base)
            b_mod = calc_bias(y_true, y_mod)
            b_base = calc_bias(y_true, y_base)
            fva_pts = float(round(w_base - w_mod, 4))
            regimes[label] = {
                "n_samples": int(len(grp)),
                "model_wape": w_mod,
                "model_bias": b_mod,
                "baseline_wape": w_base,
                "baseline_bias": b_base,
                "fva_wape_pts": fva_pts,
            }
        results[key] = regimes
    return results


def evaluate_model_vs_baselines(
    sales_df: pd.DataFrame,
    holdout_origin: pd.Timestamp | str,
    stores_df: pd.DataFrame | None = None,
    holidays_df: pd.DataFrame | None = None,
    start_train_date: str = "2015-01-01",
    target_col: str = "sales",
    params: dict[str, Any] | None = None,
    exclude_preopening: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train LightGBM and benchmark against Naive, Seasonal Naive, and MA28 on holdout.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Complete historical store-family daily sales data.
    holdout_origin : pd.Timestamp | str
        Date representing t_0; test period is all dates > holdout_origin.
    stores_df : pd.DataFrame | None
        Store metadata.
    holidays_df : pd.DataFrame | None
        Holidays table.
    start_train_date : str
        Earliest training date to include (post-promo tracking inception).
    target_col : str
        Target sales column.
    params : dict[str, Any] | None
        Optional LightGBM hyperparameters.
    exclude_preopening : bool
        If True, excludes rows before a store's first opening date from training.

    Returns
    -------
    tuple[dict[str, Any], pd.DataFrame]
        (Summary metrics dictionary, Test DataFrame with all predictions)
    """
    origin = pd.to_datetime(holdout_origin)
    df_raw = sales_df.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    log.info("Building causal features...")
    df_feat, feature_names = build_features(
        df_raw, stores_df=stores_df, holidays_df=holidays_df, target_col=target_col
    )

    # Training and test masks
    train_mask = (df_feat["date"] <= origin) & (df_feat["date"] >= pd.to_datetime(start_train_date))
    if exclude_preopening:
        opening_dates = get_store_opening_dates(df_raw, target_col=target_col)
        is_open = df_feat["store_nbr"].map(opening_dates)
        train_mask = train_mask & (df_feat["date"] >= is_open)

    test_mask = df_feat["date"] > origin

    X_train = df_feat.loc[train_mask, feature_names]
    y_train = np.log1p(df_feat.loc[train_mask, target_col])

    X_test = df_feat.loc[test_mask, feature_names]
    test_df = df_feat.loc[test_mask].copy()

    log.info("Training LightGBM on %d rows with %d features...", len(X_train), len(feature_names))
    model = train_lightgbm(X_train, y_train, params=params)

    # 1. LightGBM Predictions
    test_df["pred_lgbm"] = predict_lightgbm(model, X_test)

    # 2. Heuristic Baselines on raw sales history <= origin
    history_raw = df_raw[df_raw["date"] <= origin]

    test_df = predict_naive(history_raw, test_df, target_col=target_col, pred_col="pred_naive")
    test_df = predict_seasonal_naive(
        history_raw, test_df, target_col=target_col, pred_col="pred_seasonal_naive"
    )
    test_df = predict_moving_average(
        history_raw, test_df, window_days=28, target_col=target_col, pred_col="pred_ma28"
    )

    # Evaluate all models
    lgbm_eval = evaluate_predictions(
        test_df, true_col=target_col, pred_col="pred_lgbm", group_cols=["family"]
    )
    naive_eval = evaluate_predictions(
        test_df, true_col=target_col, pred_col="pred_naive", group_cols=["family"]
    )
    snaive_eval = evaluate_predictions(
        test_df, true_col=target_col, pred_col="pred_seasonal_naive", group_cols=["family"]
    )
    ma28_eval = evaluate_predictions(
        test_df, true_col=target_col, pred_col="pred_ma28", group_cols=["family"]
    )

    # Forecast Value Added (FVA) vs best baseline (Seasonal Naive 7-day)
    snaive_wape = snaive_eval["overall"]["wape"]
    lgbm_wape = lgbm_eval["overall"]["wape"]
    fva_wape_pts = float(round(snaive_wape - lgbm_wape, 4))
    fva_relative_pct = float(round((fva_wape_pts / snaive_wape) * 100.0, 2))

    daytype_metrics = compute_daytype_metrics(
        test_df,
        target_col=target_col,
        model_pred_col="pred_lgbm",
        baseline_pred_col="pred_seasonal_naive",
    )

    summary: dict[str, Any] = {
        "holdout_origin": origin.strftime("%Y-%m-%d"),
        "test_start_date": test_df["date"].min().strftime("%Y-%m-%d"),
        "test_end_date": test_df["date"].max().strftime("%Y-%m-%d"),
        "test_n_rows": int(len(test_df)),
        "models": {
            "lightgbm": lgbm_eval["overall"],
            "seasonal_naive_7d": snaive_eval["overall"],
            "moving_average_28d": ma28_eval["overall"],
            "naive": naive_eval["overall"],
        },
        "forecast_value_added_vs_seasonal_naive": {
            "wape_reduction_percentage_points": fva_wape_pts,
            "relative_error_reduction_pct": fva_relative_pct,
        },
        "daytype_breakdown": daytype_metrics,
        "by_family": {
            fam: {
                "lightgbm": lgbm_eval["by_family"][fam],
                "seasonal_naive_7d": snaive_eval["by_family"][fam],
                "fva_wape_pts": float(
                    round(
                        snaive_eval["by_family"][fam]["wape"] - lgbm_eval["by_family"][fam]["wape"],
                        4,
                    )
                ),
            }
            for fam in lgbm_eval["by_family"]
        },
    }

    log.info(
        "Evaluation Complete: LightGBM WAPE=%.2f%% vs Seasonal Naive WAPE=%.2f%% (FVA: +%.2f pts)",
        lgbm_wape,
        snaive_wape,
        fva_wape_pts,
    )

    return summary, test_df


def run_rolling_origin_cv(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame | None = None,
    holidays_df: pd.DataFrame | None = None,
    n_folds: int = 5,
    horizon_days: int = 16,
    fold_step_days: int = 16,
    holdout_days: int = 16,
    start_train_date: str = "2015-01-01",
    target_col: str = "sales",
    params: dict[str, Any] | None = None,
    exclude_preopening: bool = True,
) -> dict[str, Any]:
    """Execute rolling-origin cross-validation backtest across N expanding-window folds.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Complete historical store-family daily sales data.
    stores_df : pd.DataFrame | None
        Store metadata.
    holidays_df : pd.DataFrame | None
        Holidays table.
    n_folds : int
        Number of rolling-origin folds.
    horizon_days : int
        Forecast horizon per fold.
    fold_step_days : int
        Days between consecutive origins.
    holdout_days : int
        Days reserved for final untouched holdout period.
    start_train_date : str
        Earliest training date to include.
    target_col : str
        Target sales column.
    params : dict[str, Any] | None
        Optional LightGBM hyperparameters.
    exclude_preopening : bool
        If True, excludes rows before a store's first opening date from training.

    Returns
    -------
    dict[str, Any]
        Fold-level and aggregated cross-validation metrics across all models.
    """
    from demand_planner.evaluation.backtest import (
        assert_no_temporal_leakage,
        generate_rolling_folds,
    )

    df_raw = sales_df.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    log.info("Building causal features for rolling-origin CV...")
    df_feat, feature_names = build_features(
        df_raw, stores_df=stores_df, holidays_df=holidays_df, target_col=target_col
    )

    cv_folds, _ = generate_rolling_folds(
        df_raw,
        n_folds=n_folds,
        horizon_days=horizon_days,
        fold_step_days=fold_step_days,
        holdout_days=holdout_days,
    )

    start_date = pd.to_datetime(start_train_date)
    fold_records: list[dict[str, Any]] = []

    for f in cv_folds:
        log.info(
            "Running CV Fold %d/%d (Origin: %s)...",
            f.fold_idx,
            len(cv_folds),
            f.origin_date.strftime("%Y-%m-%d"),
        )
        train_mask = (df_feat["date"] <= f.origin_date) & (df_feat["date"] >= start_date)
        if exclude_preopening:
            opening_dates = get_store_opening_dates(df_raw, target_col=target_col)
            is_open = df_feat["store_nbr"].map(opening_dates)
            train_mask = train_mask & (df_feat["date"] >= is_open)

        test_mask = (df_feat["date"] > f.origin_date) & (
            df_feat["date"] <= f.origin_date + pd.Timedelta(days=horizon_days)
        )

        X_train = df_feat.loc[train_mask, feature_names]
        y_train = np.log1p(df_feat.loc[train_mask, target_col])

        X_test = df_feat.loc[test_mask, feature_names]
        test_df = df_feat.loc[test_mask].copy()

        assert_no_temporal_leakage(df_feat.loc[train_mask], test_df, f.origin_date)

        # Train model
        model = train_lightgbm(X_train, y_train, params=params)
        test_df["pred_lgbm"] = predict_lightgbm(model, X_test)

        # Baselines on raw history
        hist_raw = df_raw[df_raw["date"] <= f.origin_date]
        test_df = predict_naive(hist_raw, test_df, target_col=target_col, pred_col="pred_naive")
        test_df = predict_seasonal_naive(
            hist_raw, test_df, target_col=target_col, pred_col="pred_snaive"
        )
        test_df = predict_moving_average(
            hist_raw, test_df, window_days=28, target_col=target_col, pred_col="pred_ma28"
        )

        lgbm_eval = evaluate_predictions(test_df, true_col=target_col, pred_col="pred_lgbm")
        snaive_eval = evaluate_predictions(test_df, true_col=target_col, pred_col="pred_snaive")
        ma28_eval = evaluate_predictions(test_df, true_col=target_col, pred_col="pred_ma28")
        naive_eval = evaluate_predictions(test_df, true_col=target_col, pred_col="pred_naive")

        fva_pts = float(round(snaive_eval["overall"]["wape"] - lgbm_eval["overall"]["wape"], 4))
        daytype_metrics = compute_daytype_metrics(
            test_df,
            target_col=target_col,
            model_pred_col="pred_lgbm",
            baseline_pred_col="pred_snaive",
        )

        fold_records.append(
            {
                "fold_idx": f.fold_idx,
                "origin_date": f.origin_date.strftime("%Y-%m-%d"),
                "test_start_date": test_df["date"].min().strftime("%Y-%m-%d"),
                "test_end_date": test_df["date"].max().strftime("%Y-%m-%d"),
                "n_samples": int(len(test_df)),
                "lightgbm": lgbm_eval["overall"],
                "seasonal_naive_7d": snaive_eval["overall"],
                "moving_average_28d": ma28_eval["overall"],
                "naive": naive_eval["overall"],
                "fva_wape_pts": fva_pts,
                "daytype_breakdown": daytype_metrics,
            }
        )

    # Summary statistics
    lgbm_wapes = [r["lightgbm"]["wape"] for r in fold_records]
    snaive_wapes = [r["seasonal_naive_7d"]["wape"] for r in fold_records]
    ma28_wapes = [r["moving_average_28d"]["wape"] for r in fold_records]
    naive_wapes = [r["naive"]["wape"] for r in fold_records]
    fva_wapes = [r["fva_wape_pts"] for r in fold_records]

    lgbm_biases = [r["lightgbm"]["bias"] for r in fold_records]
    lgbm_maes = [r["lightgbm"]["mae"] for r in fold_records]

    mean_lgbm_wape = float(round(float(np.mean(lgbm_wapes)), 4))
    mean_snaive_wape = float(round(float(np.mean(snaive_wapes)), 4))
    mean_fva_pts = float(round(float(np.mean(fva_wapes)), 4))

    return {
        "n_folds": len(cv_folds),
        "horizon_days": horizon_days,
        "fold_step_days": fold_step_days,
        "folds": fold_records,
        "mean_metrics": {
            "lightgbm": {
                "mean_wape": mean_lgbm_wape,
                "std_wape": float(round(float(np.std(lgbm_wapes)), 4)),
                "mean_bias": float(round(float(np.mean(lgbm_biases)), 4)),
                "mean_mae": float(round(float(np.mean(lgbm_maes)), 4)),
            },
            "seasonal_naive_7d": {
                "mean_wape": mean_snaive_wape,
                "std_wape": float(round(float(np.std(snaive_wapes)), 4)),
                "mean_bias": float(
                    round(float(np.mean([r["seasonal_naive_7d"]["bias"] for r in fold_records])), 4)
                ),
                "mean_mae": float(
                    round(float(np.mean([r["seasonal_naive_7d"]["mae"] for r in fold_records])), 4)
                ),
            },
            "moving_average_28d": {
                "mean_wape": float(round(float(np.mean(ma28_wapes)), 4)),
                "std_wape": float(round(float(np.std(ma28_wapes)), 4)),
            },
            "naive": {
                "mean_wape": float(round(float(np.mean(naive_wapes)), 4)),
                "std_wape": float(round(float(np.std(naive_wapes)), 4)),
            },
            "forecast_value_added_vs_seasonal_naive": {
                "mean_wape_reduction_pts": mean_fva_pts,
                "relative_error_reduction_pct": float(
                    round((mean_fva_pts / mean_snaive_wape) * 100.0, 2)
                ),
            },
        },
    }
