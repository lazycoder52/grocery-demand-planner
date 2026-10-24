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

    stores = sorted(df["store_nbr"].unique())
    n_stores = len(stores)
    store_indices = {s: np.where(df["store_nbr"] == s)[0] for s in stores}

    # 1. Overall Lift Estimation
    log.info("Estimating overall store-family fixed-effects promo lift on %d rows...", len(df))
    beta_overall = estimate_fe_log_linear(df, target_col=target_col, promo_col=promo_col)
    lift_overall_pct = (np.exp(beta_overall) - 1.0) * 100.0

    # Store-clustered bootstrap for overall lift
    boot_lifts_overall: list[float] = []
    for _ in range(n_bootstrap):
        b_stores = rng.choice(stores, size=n_stores, replace=True)
        b_row_indices = np.concatenate([store_indices[s] for s in b_stores])
        b_df = df.iloc[b_row_indices].copy()
        # Re-index series within bootstrap sample
        b_series_keys = b_df[["store_nbr", "family"]].drop_duplicates().itertuples(index=False)
        b_series_map = {k: idx for idx, k in enumerate(b_series_keys)}
        b_df["series_idx"] = [
            b_series_map[(s, f)] for s, f in zip(b_df["store_nbr"], b_df["family"], strict=False)
        ]
        b_beta = estimate_fe_log_linear(b_df, target_col=target_col, promo_col=promo_col)
        boot_lifts_overall.append((np.exp(b_beta) - 1.0) * 100.0)

    ci_overall_low = float(np.percentile(boot_lifts_overall, 2.5))
    ci_overall_high = float(np.percentile(boot_lifts_overall, 97.5))

    # 2. Per-Family Lift Estimation
    families = sorted(df["family"].unique())
    family_results: list[dict[str, Any]] = []

    for fam in families:
        fam_df = df[df["family"] == fam].copy().reset_index(drop=True)
        fam_stores = sorted(fam_df["store_nbr"].unique())
        fam_store_idx = {s: np.where(fam_df["store_nbr"] == s)[0] for s in fam_stores}
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

        # Point estimate
        fam_series_map = {s: idx for idx, s in enumerate(fam_stores)}
        fam_df["series_idx"] = fam_df["store_nbr"].map(fam_series_map)
        b_fam = estimate_fe_log_linear(fam_df, target_col=target_col, promo_col=promo_col)
        l_fam = (np.exp(b_fam) - 1.0) * 100.0

        # Store-clustered bootstrap
        boot_fam: list[float] = []
        for _ in range(n_bootstrap):
            b_s = rng.choice(fam_stores, size=len(fam_stores), replace=True)
            b_rows = np.concatenate([fam_store_idx[s] for s in b_s])
            sub_b = fam_df.iloc[b_rows].copy()
            sub_series_map = {s: idx for idx, s in enumerate(sorted(sub_b["store_nbr"].unique()))}
            sub_b["series_idx"] = sub_b["store_nbr"].map(sub_series_map)
            bb = estimate_fe_log_linear(sub_b, target_col=target_col, promo_col=promo_col)
            boot_fam.append((np.exp(bb) - 1.0) * 100.0)

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
