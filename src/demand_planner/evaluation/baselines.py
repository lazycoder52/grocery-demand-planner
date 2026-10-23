"""Heuristic benchmark forecasting models.

1. Naive: Last observed sales value before forecast origin.
2. Seasonal Naive (7-day): Sales from the same day of week in the most recent week.
3. 28-Day Moving Average: Arithmetic mean of sales over the 28 days preceding origin.
"""

from __future__ import annotations

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
