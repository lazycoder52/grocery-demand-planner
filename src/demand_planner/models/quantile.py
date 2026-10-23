"""Quantile regression modeling using LightGBM for grocery demand uncertainty estimation.

Trains models for specified quantiles (default P10, P50, P90) using the same strictly
causal feature pipeline as the point forecasting model. Evaluates pinball loss
and empirical interval coverage across rolling-origin backtest folds and holdout.

Empirical Coverage & Uncertainty Notes:
- CLEANING (72.49%) and PERSONAL CARE (77.60%) exhibit notable under-coverage relative to
  the nominal 80% target interval (P10-P90), reflecting high promotional volatility and
  promotional variance not fully captured by linear interval expansion.
- Nominal coverage for zero-heavy/intermittent families (e.g. BOOKS 98.15%, BABY CARE 94.00%)
  is inflated due to zero point-mass at y=0 which automatically falls inside [0, P90].
- The higher aggregate WAPE of the P50 model (14.27% CV vs 13.49% for point model) occurs
  because P50 minimizes pinball loss (L1 conditional median), which does not sum to the
  total volume evaluated by volume-weighted WAPE, whereas the point model minimizes squared error
  (conditional expectation) with volume shrinkage.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand_planner.evaluation.metrics import calc_empirical_coverage, calc_pinball_loss
from demand_planner.features.build import build_features

log = logging.getLogger("demand_planner.models.quantile")

DEFAULT_QUANTILE_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def train_lightgbm_quantiles(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
    params: dict[str, Any] | None = None,
) -> dict[float, lgb.LGBMRegressor]:
    """Train separate LightGBM models for each specified quantile alpha.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series | np.ndarray
        Log-transformed target sales (log1p).
    alphas : tuple[float, ...]
        Quantiles to fit, e.g. (0.1, 0.5, 0.9).
    params : dict[str, Any] | None
        Optional hyperparameters overriding defaults.

    Returns
    -------
    dict[float, lgb.LGBMRegressor]
        Mapping from alpha to fitted LightGBM regressor.
    """
    cfg = dict(DEFAULT_QUANTILE_PARAMS)
    if params:
        cfg.update(params)

    models: dict[float, lgb.LGBMRegressor] = {}
    for alpha in alphas:
        log.debug("Fitting LightGBM quantile alpha=%.2f...", alpha)
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            **cfg,
        )
        model.fit(X_train, y_train)
        models[alpha] = model

    return models


def predict_lightgbm_quantiles(
    models: dict[float, lgb.LGBMRegressor],
    X_test: pd.DataFrame,
    enforce_monotonicity: bool = True,
) -> dict[str, np.ndarray]:
    """Generate inverted, non-negative quantile predictions.

    Parameters
    ----------
    models : dict[float, lgb.LGBMRegressor]
        Fitted quantile models.
    X_test : pd.DataFrame
        Test features.
    enforce_monotonicity : bool
        If True and standard alphas (0.1, 0.5, 0.9) are present, clips predictions
        so that P10 <= P50 <= P90.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys 'p10', 'p50', 'p90' (or 'p{int(alpha*100)}').
    """
    raw_preds: dict[float, np.ndarray] = {}
    for alpha, model in models.items():
        log_pred = model.predict(X_test)
        raw_preds[alpha] = np.expm1(log_pred).clip(min=0.0)

    if enforce_monotonicity and set(models.keys()) >= {0.1, 0.5, 0.9}:
        p10 = raw_preds[0.1]
        p50 = np.maximum(p10, raw_preds[0.5])
        p90 = np.maximum(p50, raw_preds[0.9])
        return {"p10": p10, "p50": p50, "p90": p90}

    return {f"p{int(round(a * 100))}": arr for a, arr in raw_preds.items()}


def evaluate_quantiles(
    y_true: np.ndarray | pd.Series,
    preds: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute pinball losses and empirical coverage for quantile predictions.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground truth actual sales.
    preds : dict[str, np.ndarray]
        Predicted quantiles with keys 'p10', 'p50', 'p90'.

    Returns
    -------
    dict[str, Any]
        Dictionary of pinball losses, empirical coverage %, and boundary stats.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p10 = np.asarray(preds["p10"], dtype=np.float64)
    p50 = np.asarray(preds["p50"], dtype=np.float64)
    p90 = np.asarray(preds["p90"], dtype=np.float64)

    pb10 = calc_pinball_loss(y, p10, 0.1)
    pb50 = calc_pinball_loss(y, p50, 0.5)
    pb90 = calc_pinball_loss(y, p90, 0.9)
    mean_pb = float(round(float((pb10 + pb50 + pb90) / 3.0), 4))

    coverage = calc_empirical_coverage(y, p10, p90)
    below_p10 = float(round(float(np.mean(y < p10) * 100.0), 4))
    above_p90 = float(round(float(np.mean(y > p90) * 100.0), 4))

    return {
        "pinball_loss_p10": pb10,
        "pinball_loss_p50": pb50,
        "pinball_loss_p90": pb90,
        "mean_pinball_loss": mean_pb,
        "coverage_p10_p90_pct": coverage,
        "nominal_coverage_pct": 80.0,
        "below_p10_pct": below_p10,
        "above_p90_pct": above_p90,
        "n_samples": int(len(y)),
    }


def run_quantile_cv(
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
    """Execute expanding-window cross-validation for LightGBM quantile regression.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Historical sales dataset.
    stores_df : pd.DataFrame | None
        Store metadata.
    holidays_df : pd.DataFrame | None
        Holidays table.
    n_folds : int
        Number of rolling-origin folds.
    horizon_days : int
        Forecast horizon in days.
    fold_step_days : int
        Days between consecutive origins.
    holdout_days : int
        Days reserved for final holdout.
    start_train_date : str
        Earliest training date to include.
    target_col : str
        Target column name.
    params : dict[str, Any] | None
        Optional model hyperparameters.
    exclude_preopening : bool
        If True, excludes rows before store opening date from training.

    Returns
    -------
    dict[str, Any]
        Per-fold and aggregated cross-validation quantile evaluation metrics.
    """
    from demand_planner.evaluation.backtest import generate_rolling_folds
    from demand_planner.evaluation.baselines import predict_seasonal_naive_quantiles
    from demand_planner.models.lightgbm import get_store_opening_dates

    df_raw = sales_df.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    log.info("Building causal features for quantile CV backtest...")
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
            "Running Quantile CV Fold %d/%d (Origin: %s)...",
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
        y_test = df_feat.loc[test_mask, target_col].to_numpy()
        test_df = df_feat.loc[test_mask].copy()

        q_models = train_lightgbm_quantiles(X_train, y_train, alphas=(0.1, 0.5, 0.9), params=params)
        q_preds = predict_lightgbm_quantiles(q_models, X_test)
        q_eval = evaluate_quantiles(y_test, q_preds)

        # Baseline: seasonal naive with empirical residuals
        hist_raw = df_raw[df_raw["date"] <= f.origin_date]
        base_preds, _ = predict_seasonal_naive_quantiles(
            hist_raw, test_df, target_col=target_col, alphas=(0.1, 0.5, 0.9)
        )
        base_eval = evaluate_quantiles(y_test, base_preds)

        fold_records.append(
            {
                "fold_idx": f.fold_idx,
                "origin_date": f.origin_date.strftime("%Y-%m-%d"),
                "test_start_date": (f.origin_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "test_end_date": (f.origin_date + pd.Timedelta(days=horizon_days)).strftime(
                    "%Y-%m-%d"
                ),
                "metrics": q_eval,
                "baseline_seasonal_naive_empirical": base_eval,
            }
        )

    # Aggregated CV metrics
    pb10_vals = [rec["metrics"]["pinball_loss_p10"] for rec in fold_records]
    pb50_vals = [rec["metrics"]["pinball_loss_p50"] for rec in fold_records]
    pb90_vals = [rec["metrics"]["pinball_loss_p90"] for rec in fold_records]
    mean_pb_vals = [rec["metrics"]["mean_pinball_loss"] for rec in fold_records]
    cov_vals = [rec["metrics"]["coverage_p10_p90_pct"] for rec in fold_records]

    base_pb_vals = [
        rec["baseline_seasonal_naive_empirical"]["mean_pinball_loss"] for rec in fold_records
    ]
    base_cov_vals = [
        rec["baseline_seasonal_naive_empirical"]["coverage_p10_p90_pct"] for rec in fold_records
    ]

    return {
        "n_folds": n_folds,
        "horizon_days": horizon_days,
        "fold_step_days": fold_step_days,
        "folds": fold_records,
        "mean_metrics": {
            "mean_pinball_loss_p10": float(round(float(np.mean(pb10_vals)), 4)),
            "mean_pinball_loss_p50": float(round(float(np.mean(pb50_vals)), 4)),
            "mean_pinball_loss_p90": float(round(float(np.mean(pb90_vals)), 4)),
            "mean_pinball_loss": float(round(float(np.mean(mean_pb_vals)), 4)),
            "mean_coverage_p10_p90_pct": float(round(float(np.mean(cov_vals)), 4)),
            "std_coverage_p10_p90_pct": float(round(float(np.std(cov_vals, ddof=1)), 4)),
            "nominal_coverage_pct": 80.0,
            "baseline_mean_pinball_loss": float(round(float(np.mean(base_pb_vals)), 4)),
            "baseline_mean_coverage_pct": float(round(float(np.mean(base_cov_vals)), 4)),
            "pinball_loss_reduction_pts": float(
                round(float(np.mean(base_pb_vals) - np.mean(mean_pb_vals)), 4)
            ),
        },
    }


def evaluate_quantile_holdout(
    sales_df: pd.DataFrame,
    holdout_origin: pd.Timestamp | str,
    stores_df: pd.DataFrame | None = None,
    holidays_df: pd.DataFrame | None = None,
    start_train_date: str = "2015-01-01",
    target_col: str = "sales",
    params: dict[str, Any] | None = None,
    exclude_preopening: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train LightGBM quantiles and evaluate on the untouched holdout window.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Complete historical store-family daily sales data.
    holdout_origin : pd.Timestamp | str
        Date representing t_0; test period is dates > holdout_origin.
    stores_df : pd.DataFrame | None
        Store metadata.
    holidays_df : pd.DataFrame | None
        Holidays table.
    start_train_date : str
        Earliest training date to include.
    target_col : str
        Target sales column.
    params : dict[str, Any] | None
        Optional hyperparameters.
    exclude_preopening : bool
        If True, excludes rows before store opening date from training.

    Returns
    -------
    tuple[dict[str, Any], pd.DataFrame]
        (Quantile holdout metrics, Test DataFrame with pred_p10, pred_p50, pred_p90)
    """
    from demand_planner.evaluation.baselines import predict_seasonal_naive_quantiles
    from demand_planner.evaluation.metrics import calc_wape
    from demand_planner.models.lightgbm import get_store_opening_dates

    origin = pd.to_datetime(holdout_origin)
    df_raw = sales_df.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    log.info("Building causal features for quantile holdout...")
    df_feat, feature_names = build_features(
        df_raw, stores_df=stores_df, holidays_df=holidays_df, target_col=target_col
    )

    train_mask = (df_feat["date"] <= origin) & (df_feat["date"] >= pd.to_datetime(start_train_date))
    if exclude_preopening:
        opening_dates = get_store_opening_dates(df_raw, target_col=target_col)
        is_open = df_feat["store_nbr"].map(opening_dates)
        train_mask = train_mask & (df_feat["date"] >= is_open)

    test_mask = df_feat["date"] > origin

    X_train = df_feat.loc[train_mask, feature_names]
    y_train = np.log1p(df_feat.loc[train_mask, target_col])
    X_test = df_feat.loc[test_mask, feature_names]
    y_test = df_feat.loc[test_mask, target_col].to_numpy()
    test_df = df_feat.loc[test_mask].copy()

    log.info("Training LightGBM quantiles (P10, P50, P90) on %d rows...", len(X_train))
    models = train_lightgbm_quantiles(X_train, y_train, alphas=(0.1, 0.5, 0.9), params=params)
    preds = predict_lightgbm_quantiles(models, X_test)

    test_df["pred_p10"] = preds["p10"]
    test_df["pred_p50"] = preds["p50"]
    test_df["pred_p90"] = preds["p90"]

    metrics = evaluate_quantiles(y_test, preds)
    metrics["holdout_origin"] = origin.strftime("%Y-%m-%d")
    metrics["test_start_date"] = test_df["date"].min().strftime("%Y-%m-%d")
    metrics["test_end_date"] = test_df["date"].max().strftime("%Y-%m-%d")

    # Evaluate seasonal-naive empirical residual baseline
    hist_raw = df_raw[df_raw["date"] <= origin]
    base_preds, _ = predict_seasonal_naive_quantiles(
        hist_raw, test_df, target_col=target_col, alphas=(0.1, 0.5, 0.9)
    )
    base_eval = evaluate_quantiles(y_test, base_preds)
    metrics["baseline_seasonal_naive_empirical"] = base_eval
    metrics["pinball_loss_reduction_pts"] = float(
        round(base_eval["mean_pinball_loss"] - metrics["mean_pinball_loss"], 4)
    )

    # WAPE of P50
    p50_wape = calc_wape(y_test, preds["p50"])
    metrics["wape_p50"] = p50_wape

    # Coverage by family
    by_family: dict[str, Any] = {}
    for fam, grp in test_df.groupby("family"):
        fam_y = grp[target_col].to_numpy()
        fam_p10 = grp["pred_p10"].to_numpy()
        fam_p90 = grp["pred_p90"].to_numpy()
        fam_p50 = grp["pred_p50"].to_numpy()
        cov = calc_empirical_coverage(fam_y, fam_p10, fam_p90)
        fam_p50_wape = calc_wape(fam_y, fam_p50)
        by_family[str(fam)] = {
            "coverage_p10_p90_pct": cov,
            "wape_p50": fam_p50_wape,
            "n_samples": int(len(grp)),
        }
    metrics["by_family"] = by_family

    return metrics, test_df


def generate_all_predictions(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame | None = None,
    holidays_df: pd.DataFrame | None = None,
    start_train_date: str = "2015-01-01",
    target_col: str = "sales",
    exclude_preopening: bool = True,
    n_folds: int = 5,
    horizon_days: int = 16,
    fold_step_days: int = 16,
    holdout_days: int = 16,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate out-of-fold and holdout predictions for point and quantile models."""
    from demand_planner.evaluation.backtest import generate_rolling_folds
    from demand_planner.evaluation.baselines import predict_seasonal_naive
    from demand_planner.models.lightgbm import (
        get_store_opening_dates,
        predict_lightgbm,
        train_lightgbm,
    )

    df_raw = sales_df.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    df_feat, feature_names = build_features(
        df_raw, stores_df=stores_df, holidays_df=holidays_df, target_col=target_col
    )

    cv_folds, holdout = generate_rolling_folds(
        df_raw,
        n_folds=n_folds,
        horizon_days=horizon_days,
        fold_step_days=fold_step_days,
        holdout_days=holdout_days,
    )

    all_splits = [(f"fold_{f.fold_idx}", f.fold_idx, f) for f in cv_folds] + [
        ("holdout", 0, holdout)
    ]

    start_date = pd.to_datetime(start_train_date)
    opening_dates = (
        get_store_opening_dates(df_raw, target_col=target_col) if exclude_preopening else None
    )
    is_open = df_feat["store_nbr"].map(opening_dates) if opening_dates else None

    records = []
    for split_name, fold_idx, split in all_splits:
        if split_name == "holdout":
            test_mask = df_feat["date"] > split.origin_date
        else:
            test_mask = (df_feat["date"] > split.origin_date) & (
                df_feat["date"] <= split.origin_date + pd.Timedelta(days=horizon_days)
            )

        train_mask = (df_feat["date"] <= split.origin_date) & (df_feat["date"] >= start_date)
        if is_open is not None:
            train_mask = train_mask & (df_feat["date"] >= is_open)

        X_train = df_feat.loc[train_mask, feature_names]
        y_train = np.log1p(df_feat.loc[train_mask, target_col])
        X_test = df_feat.loc[test_mask, feature_names]
        test_sub = df_feat.loc[test_mask, ["date", "store_nbr", "family", target_col]].copy()

        pt_mod = train_lightgbm(X_train, y_train, params=params)
        test_sub["pred_lgbm"] = predict_lightgbm(pt_mod, X_test).astype(np.float32)

        hist_raw = df_raw[df_raw["date"] <= split.origin_date]
        test_sub = predict_seasonal_naive(
            hist_raw, test_sub, target_col=target_col, pred_col="pred_snaive"
        )
        test_sub["pred_snaive"] = test_sub["pred_snaive"].astype(np.float32)

        q_mods = train_lightgbm_quantiles(X_train, y_train, alphas=(0.1, 0.5, 0.9), params=params)
        q_preds = predict_lightgbm_quantiles(q_mods, X_test)
        test_sub["pred_p10"] = q_preds["p10"].astype(np.float32)
        test_sub["pred_p50"] = q_preds["p50"].astype(np.float32)
        test_sub["pred_p90"] = q_preds["p90"].astype(np.float32)

        # Pre-origin empirical residual standard deviation strictly from date <= split.origin_date
        lookback_start = split.origin_date - pd.Timedelta(days=56)
        lb_mask = train_mask & (df_feat["date"] > lookback_start)

        # 1. LightGBM pre-origin empirical residual std
        if lb_mask.any():
            lb_preds = predict_lightgbm(pt_mod, df_feat.loc[lb_mask, feature_names])
            lb_res = df_feat.loc[lb_mask, target_col].to_numpy() - lb_preds
            df_lb = df_feat.loc[lb_mask, ["store_nbr", "family"]].copy()
            df_lb["res_lgbm"] = lb_res
            sig_lgbm = (
                df_lb.groupby(["store_nbr", "family"], observed=False)["res_lgbm"]
                .std()
                .fillna(0.0)
                .reset_index()
            )
            sig_lgbm.rename(columns={"res_lgbm": "sigma_lgbm"}, inplace=True)
            test_sub = test_sub.merge(sig_lgbm, on=["store_nbr", "family"], how="left")
        else:
            test_sub["sigma_lgbm"] = np.float32(0.0)

        # 2. Seasonal Naive pre-origin empirical residual std
        hist_sub = hist_raw[hist_raw["date"] > lookback_start].copy()
        if len(hist_sub) > 0:
            hist_sub["_lag7"] = hist_sub.groupby(["store_nbr", "family"], observed=False)[
                target_col
            ].shift(7)
            hist_sub["res_snaive"] = hist_sub[target_col] - hist_sub["_lag7"]
            sig_snaive = (
                hist_sub.groupby(["store_nbr", "family"], observed=False)["res_snaive"]
                .std()
                .fillna(0.0)
                .reset_index()
            )
            sig_snaive.rename(columns={"res_snaive": "sigma_snaive"}, inplace=True)
            test_sub = test_sub.merge(sig_snaive, on=["store_nbr", "family"], how="left")
        else:
            test_sub["sigma_snaive"] = np.float32(0.0)

        test_sub["sigma_lgbm"] = test_sub["sigma_lgbm"].fillna(0.0).astype(np.float32)
        test_sub["sigma_snaive"] = test_sub["sigma_snaive"].fillna(0.0).astype(np.float32)

        test_sub["split"] = split_name
        test_sub["fold_idx"] = np.int8(fold_idx)
        test_sub[target_col] = test_sub[target_col].astype(np.float32)
        test_sub["store_nbr"] = test_sub["store_nbr"].astype(np.int16)
        test_sub["family"] = test_sub["family"].astype("category")
        test_sub["split"] = test_sub["split"].astype("category")

        records.append(test_sub)

    return pd.concat(records, ignore_index=True)
