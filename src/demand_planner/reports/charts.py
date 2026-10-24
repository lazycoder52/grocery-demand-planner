"""Generate visual artifacts for documentation and portfolio product reporting.

Charts produced:
1. docs/img/wape_by_family_vs_baseline.png: Comparison of LightGBM vs Seasonal Naive WAPE.
2. docs/img/actual_vs_forecast_stores.png: Daily actuals vs forecasts for two representative stores.
3. docs/img/zero_rate_by_family.png: Zero-sales percentage illustrating demand intermittency.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from demand_planner.config import DOCS, METRICS_FILE, OUTPUTS

# Report visualization styling
_style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
plt.style.use(_style)
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_wape_by_family(
    metrics_path: Path | str = METRICS_FILE,
    output_path: Path | str = DOCS / "img" / "wape_by_family_vs_baseline.png",
) -> Path:
    """Generate horizontal bar chart comparing LightGBM vs Seasonal Naive WAPE by family."""
    with Path(metrics_path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    by_fam = data.get("models", {}).get("by_family", {})
    if not by_fam:
        raise ValueError("No by_family metrics found in metrics.json")

    rows = []
    for fam, stats in by_fam.items():
        rows.append(
            {
                "family": fam,
                "wape_lgbm": stats["lightgbm"]["wape"],
                "wape_snaive": stats["seasonal_naive_7d"]["wape"],
                "actual_sales": stats["lightgbm"]["total_actual"],
            }
        )
    df = pd.DataFrame(rows).sort_values("actual_sales", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12), dpi=150)
    y_pos = np.arange(len(df))
    height = 0.38

    ax.barh(
        y_pos + height / 2,
        df["wape_snaive"],
        height=height,
        label="Seasonal Naive (7-Day Lag)",
        color="#e06666",
        alpha=0.85,
    )
    ax.barh(
        y_pos - height / 2,
        df["wape_lgbm"],
        height=height,
        label="LightGBM Global Regressor",
        color="#2b5c8f",
        alpha=0.9,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["family"], fontsize=9)
    ax.set_xlabel("WAPE % (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Forecast Performance by Family: LightGBM vs Seasonal Naive\n"
        "16-Day Holdout Period (Aug 2017) - Sorted by Sales Volume",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.set_xlim(0, max(df["wape_snaive"].max(), df["wape_lgbm"].max()) * 1.08)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_actual_vs_forecast_stores(
    preds_path: Path | str = OUTPUTS / "holdout_predictions.parquet",
    output_path: Path | str = DOCS / "img" / "actual_vs_forecast_stores.png",
    stores: tuple[int, int] = (1, 2),
) -> Path:
    """Generate daily time series comparing actuals vs forecasts for two stores."""
    preds_p = Path(preds_path)
    if not preds_p.exists():
        raise FileNotFoundError(f"Predictions parquet not found at {preds_p}")

    df = pd.read_parquet(preds_p)
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate daily totals per store
    daily = (
        df[df["store_nbr"].isin(stores)]
        .groupby(["store_nbr", "date"])[["sales", "pred_lgbm", "pred_seasonal_naive"]]
        .sum()
        .reset_index()
    )

    fig, axes = plt.subplots(len(stores), 1, figsize=(11, 7), sharex=True, dpi=150)
    if len(stores) == 1:
        axes = [axes]

    for ax, s_nbr in zip(axes, stores, strict=False):
        sub = daily[daily["store_nbr"] == s_nbr].sort_values("date")
        ax.plot(
            sub["date"],
            sub["sales"],
            marker="o",
            linewidth=2.2,
            color="#222222",
            label="Actual Sales",
        )
        ax.plot(
            sub["date"],
            sub["pred_lgbm"],
            marker="s",
            linewidth=1.8,
            color="#2b5c8f",
            linestyle="--",
            label="LightGBM Forecast",
        )
        ax.plot(
            sub["date"],
            sub["pred_seasonal_naive"],
            marker="^",
            linewidth=1.4,
            color="#e06666",
            linestyle=":",
            alpha=0.7,
            label="Seasonal Naive (7d)",
        )

        ax.set_title(
            f"Store #{s_nbr} (Total Store Daily Volume)",
            fontsize=11,
            fontweight="bold",
            loc="left",
        )
        ax.set_ylabel("Daily Units Sold", fontsize=10)
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle(
        "Actual Sales vs Forecasts: 16-Day Holdout (2017-07-31 to 2017-08-15)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_zero_rate_by_family(
    metrics_path: Path | str = METRICS_FILE,
    output_path: Path | str = DOCS / "img" / "zero_rate_by_family.png",
) -> Path:
    """Generate bar chart of zero-sales rate across all product families."""
    with Path(metrics_path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    family_stats = data.get("audit", {}).get("family_stats", {})
    if not family_stats:
        raise ValueError("No family_stats found in audit section of metrics.json")

    rows = []
    for fam, stats in family_stats.items():
        rows.append(
            {
                "family": fam,
                "zero_share_pct": stats["zero_share_pct"],
                "is_intermittent": stats["is_intermittent"],
            }
        )
    df = pd.DataFrame(rows).sort_values("zero_share_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 11), dpi=150)
    y_pos = np.arange(len(df))

    colors = ["#e06666" if is_i else "#5b9bd5" for is_i in df["is_intermittent"]]

    ax.barh(y_pos, df["zero_share_pct"], color=colors, alpha=0.9, height=0.65)
    ax.axvline(
        50.0,
        color="#cc0000",
        linestyle="--",
        linewidth=1.5,
        label="Intermittent Threshold (50% Zeros)",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["family"], fontsize=9)
    ax.set_xlabel("Zero-Sales Days Share (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Demand Intermittency: Share of Zero-Sales Observations by Family\n"
        "Full Training History (2013-2017) - 7 Families Exceed 50% Intermittency",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.set_xlim(0, 105)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def generate_all_charts() -> list[Path]:
    """Generate all 3 report charts."""
    p1 = plot_wape_by_family()
    p2 = plot_actual_vs_forecast_stores()
    p3 = plot_zero_rate_by_family()
    return [p1, p2, p3]


if __name__ == "__main__":
    generated = generate_all_charts()
    for p in generated:
        print(f"Generated chart: {p}")
