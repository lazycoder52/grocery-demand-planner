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


def calc_pinball_loss(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, alpha: float
) -> float:
    """Calculate Pinball Loss (Quantile Loss) for quantile alpha in (0, 1).

    L_alpha(y, y_hat) = mean(max(alpha * (y - y_hat), (alpha - 1.0) * (y - y_hat)))
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Quantile alpha must be strictly between 0 and 1, got {alpha}")
    y = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_pred, dtype=np.float64)
    diff = y - y_hat
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(round(float(np.mean(loss)), 4))


def calc_empirical_coverage(
    y_true: np.ndarray | pd.Series,
    y_lower: np.ndarray | pd.Series,
    y_upper: np.ndarray | pd.Series,
) -> float:
    """Calculate empirical coverage percentage: fraction of y_true in [y_lower, y_upper] * 100."""
    y = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(y_lower, dtype=np.float64)
    upper = np.asarray(y_upper, dtype=np.float64)
    in_interval = (y >= lower) & (y <= upper)
    return float(round(float(np.mean(in_interval) * 100.0), 4))


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


def compute_promotion_terciles_residual_sigma(
    df: pd.DataFrame,
    family: str | None = "CLEANING",
    pred_lgbm_col: str = "pred_lgbm",
    pred_snaive_col: str = "pred_snaive",
    sales_col: str = "sales",
    promo_col: str = "onpromotion",
) -> list[dict[str, Any]]:
    """Compare residual sigma for LightGBM vs seasonal naive across terciles of onpromotion count.

    Partitions store-family-day observations by promotional activity terciles (low, mid, high)
    and computes the sample standard deviation (ddof=1) of forecast residuals for both models.
    """
    data = df.copy()
    if family is not None and "family" in data.columns:
        data = data[data["family"] == family].copy()

    if len(data) == 0:
        return []

    y = data[sales_col].to_numpy(dtype=np.float64)
    data["_res_lgbm"] = y - data[pred_lgbm_col].to_numpy(dtype=np.float64)
    data["_res_snaive"] = y - data[pred_snaive_col].to_numpy(dtype=np.float64)

    promos = data[promo_col].to_numpy(dtype=np.float64)
    q33 = float(np.percentile(promos, 100.0 / 3.0))
    q66 = float(np.percentile(promos, 200.0 / 3.0))

    t1_mask = promos <= q33
    t2_mask = (promos > q33) & (promos <= q66)
    t3_mask = promos > q66

    tercile_specs = [
        ("Tercile 1 (Low)", t1_mask),
        ("Tercile 2 (Mid)", t2_mask),
        ("Tercile 3 (High)", t3_mask),
    ]

    results = []
    for label, mask in tercile_specs:
        sub = data[mask]
        n_obs = len(sub)
        if n_obs < 2:
            continue
        p_min = float(sub[promo_col].min())
        p_max = float(sub[promo_col].max())
        s_lgbm = float(round(float(np.std(sub["_res_lgbm"].to_numpy(), ddof=1)), 4))
        s_snaive = float(round(float(np.std(sub["_res_snaive"].to_numpy(), ddof=1)), 4))
        pct_diff = (
            float(round((s_lgbm - s_snaive) / s_snaive * 100.0, 2)) if s_snaive > 0.0 else 0.0
        )
        results.append(
            {
                "tercile": label,
                "promo_min": p_min,
                "promo_max": p_max,
                "n_obs": n_obs,
                "sigma_lgbm": s_lgbm,
                "sigma_snaive": s_snaive,
                "pct_diff": pct_diff,
            }
        )

    return results
