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


def evaluate_model_vs_baselines(
    sales_df: pd.DataFrame,
    holdout_origin: pd.Timestamp | str,
    stores_df: pd.DataFrame | None = None,
    holidays_df: pd.DataFrame | None = None,
    start_train_date: str = "2015-01-01",
    target_col: str = "sales",
    params: dict[str, Any] | None = None,
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
