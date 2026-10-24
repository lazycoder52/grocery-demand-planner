"""Rolling-origin backtest harness and temporal leakage verification.

Supports expanding-window backtesting across N folds with fixed forecast horizons,
matching configuration parameters from config/model.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from demand_planner.config import load_config
from demand_planner.evaluation.baselines import (
    predict_moving_average,
    predict_naive,
    predict_seasonal_naive,
)
from demand_planner.evaluation.metrics import evaluate_predictions


@dataclass
class FoldSplit:
    """Container for a single backtest fold."""

    fold_idx: int
    origin_date: pd.Timestamp
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def assert_no_temporal_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    origin_date: pd.Timestamp | str,
    feature_cols: list[str] | None = None,
    date_col: str = "date",
) -> None:
    """Verify that train and test splits satisfy strict temporal boundaries.

    Raises
    ------
    ValueError
        If training data contains dates after the forecast origin, if test data
        contains dates on or before the origin, or if test features contain
        forbidden post-checkout signals like same-day transactions.
    """
    origin = pd.to_datetime(origin_date)
    train_dates = pd.to_datetime(train_df[date_col])
    test_dates = pd.to_datetime(test_df[date_col])

    if len(train_dates) > 0 and train_dates.max() > origin:
        violators = train_df[train_dates > origin]
        raise ValueError(
            f"Temporal leakage detected in training set: {len(violators)} rows "
            f"have {date_col} > origin ({origin:%Y-%m-%d}). Max date: {train_dates.max()}"
        )

    if len(test_dates) > 0 and test_dates.min() <= origin:
        violators = test_df[test_dates <= origin]
        raise ValueError(
            f"Temporal leakage detected in test set: {len(violators)} rows "
            f"have {date_col} <= origin ({origin:%Y-%m-%d}). Min date: {test_dates.min()}"
        )

    # Check for forbidden leaky features
    forbidden_features = {"transactions", "target_lag_0", "sales_same_day"}
    if feature_cols:
        leaky_cols = [c for c in feature_cols if c.lower() in forbidden_features]
        if leaky_cols:
            raise ValueError(
                "Temporal leakage detected: test features contain prohibited "
                f"post-origin columns: {leaky_cols}"
            )


def generate_rolling_folds(
    df: pd.DataFrame,
    date_col: str = "date",
    n_folds: int = 5,
    horizon_days: int = 16,
    fold_step_days: int = 16,
    holdout_days: int = 16,
) -> tuple[list[FoldSplit], FoldSplit]:
    """Generate expanding-window rolling-origin cross-validation folds and a final holdout.

    Parameters
    ----------
    df : pd.DataFrame
        Complete historical dataset with date column.
    date_col : str
        Name of date column.
    n_folds : int
        Number of rolling-origin folds.
    horizon_days : int
        Forecast horizon per fold (e.g. 16 days).
    fold_step_days : int
        Days between consecutive origins.
    holdout_days : int
        Number of days reserved for the final holdout split at the end of the dataset.

    Returns
    -------
    tuple[list[FoldSplit], FoldSplit]
        (cv_folds, holdout_split)
    """
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    max_date = data[date_col].max()

    # Final holdout: last `holdout_days`
    holdout_origin = max_date - pd.Timedelta(days=holdout_days)
    holdout_train = data[data[date_col] <= holdout_origin].copy()
    holdout_test = data[
        (data[date_col] > holdout_origin)
        & (data[date_col] <= holdout_origin + pd.Timedelta(days=holdout_days))
    ].copy()

    holdout_split = FoldSplit(
        fold_idx=0,
        origin_date=holdout_origin,
        train_df=holdout_train,
        test_df=holdout_test,
    )
    assert_no_temporal_leakage(holdout_train, holdout_test, holdout_origin, date_col=date_col)

    # Generate CV folds stepping backwards before the holdout origin
    cv_folds: list[FoldSplit] = []
    for k in range(n_folds, 0, -1):
        step_offset = (n_folds - k + 1) * fold_step_days
        origin = holdout_origin - pd.Timedelta(days=step_offset)
        train_sub = data[data[date_col] <= origin].copy()
        test_sub = data[
            (data[date_col] > origin) & (data[date_col] <= origin + pd.Timedelta(days=horizon_days))
        ].copy()

        assert_no_temporal_leakage(train_sub, test_sub, origin, date_col=date_col)
        cv_folds.append(
            FoldSplit(
                fold_idx=k,
                origin_date=origin,
                train_df=train_sub,
                test_df=test_sub,
            )
        )

    # Sort folds chronologically (fold 1 oldest origin to fold n_folds newest)
    cv_folds.sort(key=lambda f: f.origin_date)
    for idx, f in enumerate(cv_folds, 1):
        f.fold_idx = idx

    return cv_folds, holdout_split


def evaluate_baselines_on_split(
    split: FoldSplit,
    target_col: str = "sales",
    group_cols: tuple[str, ...] = ("store_nbr", "family"),
    breakdown_col: str = "family",
) -> dict[str, Any]:
    """Evaluate naive, seasonal naive, and 28-day MA on a single train/test split."""
    test = split.test_df.copy()

    # 1. Naive
    test_naive = predict_naive(
        split.train_df,
        test,
        group_cols=group_cols,
        target_col=target_col,
        pred_col="pred_naive",
    )
    res_naive = evaluate_predictions(
        test_naive, true_col=target_col, pred_col="pred_naive", group_cols=[breakdown_col]
    )

    # 2. Seasonal Naive (7-day)
    test_snaive = predict_seasonal_naive(
        split.train_df,
        test,
        group_cols=group_cols,
        target_col=target_col,
        pred_col="pred_snaive",
    )
    res_snaive = evaluate_predictions(
        test_snaive, true_col=target_col, pred_col="pred_snaive", group_cols=[breakdown_col]
    )

    # 3. 28-day Moving Average
    test_ma28 = predict_moving_average(
        split.train_df,
        test,
        window_days=28,
        group_cols=group_cols,
        target_col=target_col,
        pred_col="pred_ma28",
    )
    res_ma28 = evaluate_predictions(
        test_ma28, true_col=target_col, pred_col="pred_ma28", group_cols=[breakdown_col]
    )

    return {
        "naive": res_naive,
        "seasonal_naive_7d": res_snaive,
        "moving_average_28d": res_ma28,
    }


def run_baseline_benchmarks(
    df: pd.DataFrame,
    config_name: str = "model",
) -> dict[str, Any]:
    """Execute baseline evaluation across rolling folds and final holdout using YAML config."""
    cfg = load_config(config_name).get("evaluation", {})
    horizon_days = cfg.get("horizon_days", 16)
    n_folds = cfg.get("n_folds", 5)
    fold_step_days = cfg.get("fold_step_days", 16)
    holdout_days = cfg.get("holdout_days", 16)

    cv_folds, holdout = generate_rolling_folds(
        df,
        n_folds=n_folds,
        horizon_days=horizon_days,
        fold_step_days=fold_step_days,
        holdout_days=holdout_days,
    )

    holdout_results = evaluate_baselines_on_split(holdout)

    cv_results = []
    for f in cv_folds:
        f_res = evaluate_baselines_on_split(f)
        cv_results.append(
            {
                "fold": f.fold_idx,
                "origin": f.origin_date.strftime("%Y-%m-%d"),
                "results": f_res,
            }
        )

    return {
        "holdout": holdout_results,
        "cv_folds": cv_results,
    }
