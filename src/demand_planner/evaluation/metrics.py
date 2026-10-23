"""Evaluation metrics for demand forecasting.

Primary metric is WAPE (Weighted Absolute Percentage Error) to handle intermittent
and zero-sales series without division-by-zero instability.
Also provides Bias % and MAE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calc_wape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE) as a percentage.

    WAPE = (sum |y - y_hat|) / (sum y) * 100.
    If sum(y) is 0: returns 0.0 if predictions are also 0, else 100.0.
    """
    y = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_pred, dtype=np.float64)
    total_actual = float(np.sum(y))
    total_abs_error = float(np.sum(np.abs(y - y_hat)))
    if total_actual == 0.0:
        return 0.0 if total_abs_error == 0.0 else 100.0
    return float(round((total_abs_error / total_actual) * 100.0, 4))


def calc_bias(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Calculate Mean Percentage Bias: sum(y_hat - y) / sum(y) * 100.

    Positive bias indicates over-forecasting; negative indicates under-forecasting.
    """
    y = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_pred, dtype=np.float64)
    total_actual = float(np.sum(y))
    total_error = float(np.sum(y_hat - y))
    if total_actual == 0.0:
        return 0.0
    return float(round((total_error / total_actual) * 100.0, 4))


def calc_mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    y = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_pred, dtype=np.float64)
    return float(round(float(np.mean(np.abs(y - y_hat))), 4))


def calc_rmsle(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Calculate Root Mean Squared Logarithmic Error (RMSLE).

    Predictions are clipped to 0 prior to log transformation.
    """
    y = np.asarray(y_true, dtype=np.float64)
    y_hat = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    log_actual = np.log1p(np.clip(y, 0.0, None))
    log_pred = np.log1p(y_hat)
    return float(round(float(np.sqrt(np.mean((log_pred - log_actual) ** 2))), 4))


def evaluate_predictions(
    df: pd.DataFrame,
    true_col: str = "sales",
    pred_col: str = "pred",
    group_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Compute overall and optional per-group WAPE, bias, and MAE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ground truth and prediction columns.
    true_col : str
        Column name for actual sales.
    pred_col : str
        Column name for predicted sales.
    group_cols : list[str] | None
        Optional columns to break down metrics by (e.g. ['family']).
    """
    y_true = df[true_col].to_numpy()
    y_pred = df[pred_col].to_numpy()

    overall = {
        "wape": calc_wape(y_true, y_pred),
        "bias": calc_bias(y_true, y_pred),
        "mae": calc_mae(y_true, y_pred),
        "rmsle": calc_rmsle(y_true, y_pred),
        "total_actual_sales": float(round(float(np.sum(y_true)), 2)),
        "total_predicted_sales": float(round(float(np.sum(y_pred)), 2)),
        "n_samples": int(len(df)),
    }

    results: dict[str, Any] = {"overall": overall}

    if group_cols:
        by_group: dict[str, dict[str, Any]] = {}
        for keys, grp in df.groupby(group_cols):
            key_str = str(keys) if not isinstance(keys, tuple) else "/".join(str(k) for k in keys)
            gy_true = grp[true_col].to_numpy()
            gy_pred = grp[pred_col].to_numpy()
            by_group[key_str] = {
                "wape": calc_wape(gy_true, gy_pred),
                "bias": calc_bias(gy_true, gy_pred),
                "mae": calc_mae(gy_true, gy_pred),
                "total_actual": float(round(float(np.sum(gy_true)), 2)),
                "n_samples": int(len(grp)),
            }
        group_name = "_".join(group_cols)
        results[f"by_{group_name}"] = by_group

    return results
