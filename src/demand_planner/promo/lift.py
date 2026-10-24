"""Causal promotion-lift analysis with fixed effects and clustered bootstrap inference.

Estimates the average sales lift of promotions (onpromotion > 0) strictly on post 2014-04-01
data (when Corporación Favorita began systematically recording promotional activities).

Methodology:
- Log-linear store-family fixed-effects regression with calendar controls (day-of-week and month):
    log(sales + 1) = alpha_{store, family} + beta * is_promo + gamma_{dow} + delta_{month} + eps
- % Sales Lift = (exp(beta) - 1) * 100
- Statistical inference: 95% cluster-bootstrap confidence intervals clustered by store (54 clusters)
  to account for arbitrary within-store spatial and temporal error correlation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger("demand_planner.promo")


def estimate_fe_log_linear(
    df_sub: pd.DataFrame,
    target_col: str = "sales",
    promo_col: str = "onpromotion",
) -> float:
    """Fast Frisch-Waugh-Lovell within-transformation OLS for promo coefficient."""
    y = np.log1p(df_sub[target_col].to_numpy(dtype=np.float64))
    p = (df_sub[promo_col].to_numpy(dtype=np.float64) > 0).astype(np.float64)

    # Dummies for dow (1..6 vs 0) and month (2..12 vs 1)
    dow = df_sub["dow"].to_numpy(dtype=np.int32)
    month = df_sub["month"].to_numpy(dtype=np.int32)

    X_dow = np.column_stack([dow == d for d in range(1, 7)]).astype(np.float64)
    X_mon = np.column_stack([month == m for m in range(2, 13)]).astype(np.float64)
    X_reg = np.column_stack([p, X_dow, X_mon])  # shape (N, 18)

    # Within-transformation: de-mean by series_idx (store x family)
    series_idx = df_sub["series_idx"].to_numpy(dtype=np.int32)
    n_series = int(series_idx.max()) + 1

    counts = np.bincount(series_idx, minlength=n_series).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    # Group means
    y_sums = np.bincount(series_idx, weights=y, minlength=n_series)
    y_means = y_sums / counts
    y_tilde = y - y_means[series_idx]

    X_tilde = np.empty_like(X_reg)
    for c_i in range(X_reg.shape[1]):
        col = X_reg[:, c_i]
        c_sums = np.bincount(series_idx, weights=col, minlength=n_series)
        c_means = c_sums / counts
        X_tilde[:, c_i] = col - c_means[series_idx]

    # Solve OLS: (X_tilde' X_tilde)^(-1) X_tilde' y_tilde
    try:
        beta, _, _, _ = np.linalg.lstsq(X_tilde, y_tilde, rcond=1e-5)
        return float(beta[0])
    except Exception:
        return 0.0


def _compute_fe_store_cross_products(
    df_sub: pd.DataFrame,
    target_col: str = "sales",
    promo_col: str = "onpromotion",
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], float, list[int], dict[int, int]]:
    """Precompute within-demeaned cross products A_s = X_s' X_s and b_s = X_s' y_s per store."""
    y = np.log1p(df_sub[target_col].to_numpy(dtype=np.float64))
    p = (df_sub[promo_col].to_numpy(dtype=np.float64) > 0).astype(np.float64)

    dow = df_sub["dow"].to_numpy(dtype=np.int32)
    month = df_sub["month"].to_numpy(dtype=np.int32)

    X_dow = np.column_stack([dow == d for d in range(1, 7)]).astype(np.float64)
    X_mon = np.column_stack([month == m for m in range(2, 13)]).astype(np.float64)
    X_reg = np.column_stack([p, X_dow, X_mon])

    series_idx = df_sub["series_idx"].to_numpy(dtype=np.int32)
    n_series = int(series_idx.max()) + 1

    counts = np.bincount(series_idx, minlength=n_series).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    y_sums = np.bincount(series_idx, weights=y, minlength=n_series)
    y_means = y_sums / counts
    y_tilde = y - y_means[series_idx]

    X_tilde = np.empty_like(X_reg)
    for c_i in range(X_reg.shape[1]):
        col = X_reg[:, c_i]
        c_sums = np.bincount(series_idx, weights=col, minlength=n_series)
        c_means = c_sums / counts
        X_tilde[:, c_i] = col - c_means[series_idx]

    store_col = df_sub["store_nbr"].to_numpy(dtype=np.int32)
    stores = sorted(df_sub["store_nbr"].unique())
    store_map = {s: idx for idx, s in enumerate(stores)}

    A_stores: dict[int, np.ndarray] = {}
    b_stores: dict[int, np.ndarray] = {}
    p_dim = X_reg.shape[1]

    for s in stores:
        mask = store_col == s
        Xs = X_tilde[mask]
        ys = y_tilde[mask]
        if len(ys) > 0:
            A_stores[s] = Xs.T @ Xs
            b_stores[s] = Xs.T @ ys
        else:
            A_stores[s] = np.zeros((p_dim, p_dim), dtype=np.float64)
            b_stores[s] = np.zeros(p_dim, dtype=np.float64)

    A_tot = sum(A_stores.values())
    b_tot = sum(b_stores.values())
    try:
        beta_hat, _, _, _ = np.linalg.lstsq(A_tot, b_tot, rcond=1e-5)
        beta = float(beta_hat[0])
    except Exception:
        beta = 0.0

    return A_stores, b_stores, beta, stores, store_map


def _cluster_bootstrap_lift(
    A_stores: dict[int, np.ndarray],
    b_stores: dict[int, np.ndarray],
    stores: list[int],
    store_map: dict[int, int],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[float]:
    """Fast cluster-bootstrap for OLS using precomputed cross-products."""
    n_stores = len(stores)
    boot_lifts: list[float] = []
    for _ in range(n_bootstrap):
        b_sample = rng.choice(stores, size=n_stores, replace=True)
        counts = np.bincount([store_map[s] for s in b_sample], minlength=n_stores).astype(
            np.float64
        )
        A_b = sum(c * A_stores[s] for c, s in zip(counts, stores, strict=False))
        b_b = sum(c * b_stores[s] for c, s in zip(counts, stores, strict=False))
        try:
            b_hat, _, _, _ = np.linalg.lstsq(A_b, b_b, rcond=1e-5)
            boot_lifts.append((np.exp(float(b_hat[0])) - 1.0) * 100.0)
        except Exception:
            continue
    return boot_lifts


def run_promo_lift_analysis(
    df_sales: pd.DataFrame,
    start_date: str = "2014-04-01",
    n_bootstrap: int = 500,
    seed: int = 42,
    target_col: str = "sales",
    promo_col: str = "onpromotion",
) -> dict[str, Any]:
    """Estimate causal promo lift overall and per product family with store-clustered bootstrap.

    Parameters
    ----------
    df_sales : pd.DataFrame
        Sales dataframe containing ['date', 'store_nbr', 'family', target_col, promo_col].
    start_date : str
        Start date filtering. Must be >= '2014-04-01' because onpromotion was unrecorded earlier.
    n_bootstrap : int
        Number of store-clustered bootstrap replications for 95% CIs.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary with overall lift, CIs, top 5 promo-sensitive families, and full breakdown.
    """
    rng = np.random.default_rng(seed)

    df = df_sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.to_datetime(start_date)].copy()
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Map store x family to integer index
    series_keys = df[["store_nbr", "family"]].drop_duplicates().itertuples(index=False)
    series_map = {k: idx for idx, k in enumerate(series_keys)}
    df["series_idx"] = [
        series_map[(s, f)] for s, f in zip(df["store_nbr"], df["family"], strict=False)
    ]

    # 1. Overall Lift Estimation
    log.info("Estimating overall store-family fixed-effects promo lift on %d rows...", len(df))
    A_stores, b_stores, beta_overall, stores, store_map = _compute_fe_store_cross_products(
        df, target_col=target_col, promo_col=promo_col
    )
    lift_overall_pct = (np.exp(beta_overall) - 1.0) * 100.0

    # Fast store-clustered bootstrap for overall lift
    boot_lifts_overall = _cluster_bootstrap_lift(
        A_stores, b_stores, stores, store_map, n_bootstrap, rng
    )

    ci_overall_low = float(np.percentile(boot_lifts_overall, 2.5))
    ci_overall_high = float(np.percentile(boot_lifts_overall, 97.5))

    # 2. Per-Family Lift Estimation
    families = sorted(df["family"].unique())
    family_results: list[dict[str, Any]] = []

    for fam in families:
        fam_df = df[df["family"] == fam].copy().reset_index(drop=True)
        fam_promo_share = float((fam_df[promo_col] > 0).mean()) * 100.0

        if fam_promo_share < 0.01:
            family_results.append(
                {
                    "family": fam,
                    "lift_pct": 0.0,
                    "ci_95_low": 0.0,
                    "ci_95_high": 0.0,
                    "promo_share_pct": round(fam_promo_share, 2),
                    "is_statistically_significant": False,
                }
            )
            continue

        fam_stores = sorted(fam_df["store_nbr"].unique())
        fam_series_map = {s: idx for idx, s in enumerate(fam_stores)}
        fam_df["series_idx"] = fam_df["store_nbr"].map(fam_series_map)

        A_fam, b_fam, b_fam_est, s_fam, sm_fam = _compute_fe_store_cross_products(
            fam_df, target_col=target_col, promo_col=promo_col
        )
        l_fam = (np.exp(b_fam_est) - 1.0) * 100.0

        boot_fam = _cluster_bootstrap_lift(A_fam, b_fam, s_fam, sm_fam, n_bootstrap, rng)

        c_low = float(np.percentile(boot_fam, 2.5))
        c_high = float(np.percentile(boot_fam, 97.5))
        sig = c_low > 0 or c_high < 0

        family_results.append(
            {
                "family": fam,
                "lift_pct": round(l_fam, 2),
                "ci_95_low": round(c_low, 2),
                "ci_95_high": round(c_high, 2),
                "promo_share_pct": round(fam_promo_share, 2),
                "is_statistically_significant": sig,
            }
        )

    # Sort families by lift
    family_results.sort(key=lambda x: x["lift_pct"], reverse=True)
    top_5 = family_results[:5]

    return {
        "analysis_period": {
            "start_date": start_date,
            "end_date": df["date"].max().strftime("%Y-%m-%d"),
            "n_days": int((df["date"].max() - pd.to_datetime(start_date)).days) + 1,
            "note": (
                "Promotion tracking in Corporación Favorita began 2014-04-01; "
                "earlier dates excluded."
            ),
        },
        "methodology": (
            "Store-family fixed effects log-linear regression with day-of-week and month controls. "
            "Inference via 95% store-clustered bootstrap (54 clusters) with 500 resamples."
        ),
        "overall_lift_pct": round(lift_overall_pct, 2),
        "ci_95_overall_low": round(ci_overall_low, 2),
        "ci_95_overall_high": round(ci_overall_high, 2),
        "top_5_promo_sensitive_families": top_5,
        "by_family": family_results,
    }


def plot_promo_lift_by_family(
    promo_results: dict[str, Any],
    output_path: Path | str = "docs/img/promo_lift_by_family.png",
) -> Path:
    """Generate visualization: Estimated Promotional Lift by Product Family with 95% CIs."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fams = promo_results.get("by_family", [])
    if not fams:
        return out_p

    df_plot = pd.DataFrame(fams).sort_values("lift_pct", ascending=True)

    style_name = (
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.style.use(style_name)
    fig, ax = plt.subplots(figsize=(10, 11), dpi=300)

    y_pos = np.arange(len(df_plot))
    lifts = df_plot["lift_pct"].to_numpy()
    err_low = np.maximum(0.0, lifts - df_plot["ci_95_low"].to_numpy())
    err_high = np.maximum(0.0, df_plot["ci_95_high"].to_numpy() - lifts)

    # Highlight top 5 with vibrant emerald, others with steel blue
    top_5_fams = {f["family"] for f in promo_results.get("top_5_promo_sensitive_families", [])}
    colors = ["#2ca02c" if fam in top_5_fams else "#1f77b4" for fam in df_plot["family"]]

    ax.barh(
        y_pos,
        lifts,
        xerr=[err_low, err_high],
        align="center",
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
    )

    ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    ov = promo_results.get("overall_lift_pct", 0.0)
    ax.axvline(
        ov,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Chain Average Lift: +{ov:.1f}%",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["family"], fontsize=8.5)
    ax.set_xlabel(
        "Estimated Promotional Sales Lift (%) with 95% Clustered CI",
        fontsize=10,
        labelpad=8,
    )
    ax.set_title(
        "Estimated Causal Promotional Lift by Product Family (Post 2014-04-01)\n"
        "Store-Family Fixed Effects Log-Linear Model with Store-Clustered 95% Bootstrap CIs",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9, loc="lower right")

    fig.text(
        0.5,
        0.01,
        "Note: Fixed-effects model controls for store-family base velocity, "
        "day-of-week, and month. Top 5 promo-sensitive categories highlighted in green.",
        ha="center",
        fontsize=8,
        color="#555555",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    return out_p
