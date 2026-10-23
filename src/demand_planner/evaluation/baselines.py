"""Heuristic benchmark forecasting models.

1. Naive: Last observed sales value before forecast origin.
2. Seasonal Naive (7-day): Sales from the same day of week in the most recent week.
3. 28-Day Moving Average: Arithmetic mean of sales over the 28 days preceding origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_naive(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "date",
    group_cols: tuple[str, ...] = ("store_nbr", "family"),
    target_col: str = "sales",
    pred_col: str = "pred_naive",
) -> pd.DataFrame:
    """Predict the last observed value prior to test start for each group."""
    hist = history_df.sort_values(date_col)
    last_vals = (
        hist.groupby(list(group_cols))[target_col]
        .last()
        .reset_index()
        .rename(columns={target_col: pred_col})
    )
    result = test_df.merge(last_vals, on=list(group_cols), how="left")
    result[pred_col] = result[pred_col].fillna(0.0).clip(lower=0.0)
    return result


def predict_seasonal_naive(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    season_length: int = 7,
    date_col: str = "date",
    group_cols: tuple[str, ...] = ("store_nbr", "family"),
    target_col: str = "sales",
    pred_col: str = "pred_seasonal_naive",
) -> pd.DataFrame:
    """Predict sales from the same day of week in the most recent week before origin."""
    hist = history_df.copy()
    hist[date_col] = pd.to_datetime(hist[date_col])
    hist["_dow"] = hist[date_col].dt.dayofweek
    hist = hist.sort_values(date_col)

    # For each group and day of week, take the last observed occurrence
    last_dow = (
        hist.groupby([*group_cols, "_dow"])[target_col]
        .last()
        .reset_index()
        .rename(columns={target_col: pred_col})
    )

    result = test_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result["_dow"] = result[date_col].dt.dayofweek
    result = result.merge(last_dow, on=[*group_cols, "_dow"], how="left")

    # Fallback to group overall last observed value if a specific DOW is missing
    if result[pred_col].isna().any():
        fallback = (
            hist.groupby(list(group_cols))[target_col]
            .last()
            .reset_index()
            .rename(columns={target_col: "_fb"})
        )
        result = result.merge(fallback, on=list(group_cols), how="left")
        result[pred_col] = result[pred_col].fillna(result["_fb"])
        result.drop(columns=["_fb"], inplace=True)

    result[pred_col] = result[pred_col].fillna(0.0).clip(lower=0.0)
    result.drop(columns=["_dow"], inplace=True)
    return result


def predict_moving_average(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window_days: int = 28,
    date_col: str = "date",
    group_cols: tuple[str, ...] = ("store_nbr", "family"),
    target_col: str = "sales",
    pred_col: str = "pred_ma28",
) -> pd.DataFrame:
    """Predict the mean sales over the last ``window_days`` before forecast origin."""
    hist = history_df.copy()
    hist[date_col] = pd.to_datetime(hist[date_col])
    origin = hist[date_col].max()
    window_start = origin - pd.Timedelta(days=window_days - 1)
    window_data = hist[hist[date_col] >= window_start]

    ma_vals = (
        window_data.groupby(list(group_cols))[target_col]
        .mean()
        .reset_index()
        .rename(columns={target_col: pred_col})
    )
    result = test_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.merge(ma_vals, on=list(group_cols), how="left")

    if result[pred_col].isna().any():
        fallback = (
            hist.groupby(list(group_cols))[target_col]
            .last()
            .reset_index()
            .rename(columns={target_col: "_fb"})
        )
        result = result.merge(fallback, on=list(group_cols), how="left")
        result[pred_col] = result[pred_col].fillna(result["_fb"])
        result.drop(columns=["_fb"], inplace=True)

    result[pred_col] = result[pred_col].fillna(0.0).clip(lower=0.0)
    return result


def predict_seasonal_naive_quantiles(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lookback_days: int = 28,
    season_length: int = 7,
    date_col: str = "date",
    group_cols: tuple[str, ...] = ("store_nbr", "family"),
    family_col: str = "family",
    target_col: str = "sales",
    alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Compute seasonal-naive quantile forecasts using empirical residuals.

    1. Fits point seasonal-naive predictions on test_df.
    2. Computes empirical 7-day seasonal residuals: e_t = y_t - y_{t-7} on the
       `lookback_days` before origin in history_df.
    3. Calculates empirical quantiles of residuals per product family (with global fallback).
    4. Adds residual quantile shifts to the point forecast and non-negatively clips.
    """

    # 1. Point prediction
    test_pred_df = predict_seasonal_naive(
        history_df=history_df,
        test_df=test_df,
        season_length=season_length,
        date_col=date_col,
        group_cols=group_cols,
        target_col=target_col,
        pred_col="pred_snaive",
    )

    # 2. Empirical residuals on lookback window
    hist = history_df.copy()
    hist[date_col] = pd.to_datetime(hist[date_col])
    origin = hist[date_col].max()
    window_start = origin - pd.Timedelta(days=lookback_days + season_length)
    hist_sub = hist[hist[date_col] >= window_start].sort_values([*group_cols, date_col])

    # 7-day lagged sales
    hist_sub["_lag7"] = hist_sub.groupby(list(group_cols))[target_col].shift(season_length)
    eval_window = (
        hist_sub[hist_sub[date_col] > origin - pd.Timedelta(days=lookback_days)]
        .dropna(subset=["_lag7"])
        .copy()
    )
    eval_window["_residual"] = eval_window[target_col] - eval_window["_lag7"]

    # 3. Residual quantiles
    preds_dict: dict[str, np.ndarray] = {}
    for alpha in alphas:
        key = f"p{int(round(alpha * 100))}"
        col_name = f"snaive_{key}"
        if len(eval_window) > 0:
            fam_q = eval_window.groupby(family_col)["_residual"].quantile(alpha).to_dict()
            global_q = float(eval_window["_residual"].quantile(alpha))
            q_shifts = test_pred_df[family_col].map(fam_q).fillna(global_q).to_numpy()
        else:
            q_shifts = np.zeros(len(test_pred_df))

        q_pred = (test_pred_df["pred_snaive"].to_numpy() + q_shifts).clip(min=0.0)
        test_pred_df[col_name] = q_pred
        preds_dict[key] = q_pred

    return preds_dict, test_pred_df
