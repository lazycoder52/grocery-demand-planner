"""Periodic-review (R, S) inventory replenishment simulator.

CRITICAL ASSUMPTIONS & LIMITATIONS:
1. Corporación Favorita historical data does not contain physical inventory balances,
   on-hand stock counts, or supplier lead-time variances.
2. Fulfilled store checkout sales are utilized as an observational proxy for customer demand.
3. Because sales are censored whenever physical stockouts occur in the historical stores,
   unmet demand cannot be directly observed from point-of-sale logs. Consequently,
   simulated fill rates are optimistic upper bounds relative to true unconstrained demand.
4. All replenishment simulation outcomes are assumption-driven and parameter-contingent.
   No holding costs, obsolescence penalties, or lost-sale penalty costs are assumed or invented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("demand_planner.simulation")


@dataclass
class SimConfig:
    """Configuration parameters for periodic-review replenishment simulation."""

    seed: int = 42
    review_period_days: int = 7
    lead_times_days: tuple[int, ...] = (3, 7, 14)
    service_levels: tuple[float, ...] = (0.90, 0.95, 0.98)

    @classmethod
    def from_yaml(cls, path: Path | str = "config/sim_config.yaml") -> SimConfig:
        """Load configuration from a YAML file."""
        p = Path(path)
        if not p.exists():
            log.warning("Config file %s not found; using default parameters", path)
            return cls()
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            seed=int(raw.get("seed", 42)),
            review_period_days=int(raw.get("review_period_days", 7)),
            lead_times_days=tuple(int(x) for x in raw.get("lead_times_days", [3, 7, 14])),
            service_levels=tuple(float(x) for x in raw.get("service_levels", [0.90, 0.95, 0.98])),
        )


class PeriodicReviewSimulator:
    """Simulates periodic-review (R, S) order-up-to inventory replenishment.

    Parameters
    ----------
    df_predictions : pd.DataFrame
        DataFrame containing out-of-fold and holdout predictions. Must contain:
        ['date', 'store_nbr', 'family', 'sales', 'pred_lgbm', 'pred_snaive',
         'pred_p10', 'pred_p50', 'pred_p90'].
    config : SimConfig | None
        Simulation configuration.
    """

    def __init__(
        self,
        df_predictions: pd.DataFrame,
        config: SimConfig | None = None,
    ) -> None:
        self.config = config or SimConfig()
        self.rng = np.random.default_rng(self.config.seed)

        df = df_predictions.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

        self.dates = sorted(df["date"].unique())
        self.n_days = len(self.dates)
        date_map = {d: i for i, d in enumerate(self.dates)}
        df["day_idx"] = df["date"].map(date_map)

        self.series_keys = df[["store_nbr", "family"]].drop_duplicates().reset_index(drop=True)
        self.n_series = len(self.series_keys)

        # Pre-pivot to 2D numpy arrays of shape (n_series, n_days)
        self.sales = df.pivot(
            index=["store_nbr", "family"], columns="day_idx", values="sales"
        ).to_numpy(dtype=np.float32)
        self.pred_snaive = df.pivot(
            index=["store_nbr", "family"], columns="day_idx", values="pred_snaive"
        ).to_numpy(dtype=np.float32)
        self.pred_p50 = df.pivot(
            index=["store_nbr", "family"], columns="day_idx", values="pred_p50"
        ).to_numpy(dtype=np.float32)
        self.pred_p90 = df.pivot(
            index=["store_nbr", "family"], columns="day_idx", values="pred_p90"
        ).to_numpy(dtype=np.float32)

        # Residuals: actual sales - P50 forecast
        self.residuals = self.sales - self.pred_p50

    def _get_forecast(
        self,
        policy_name: str,
        t: int,
        protection_days: int,
    ) -> np.ndarray:
        """Sum forecasted demand over protection interval [t, t + protection_days].

        Extrapolation Rule for L=14 (protection_days=21 > 16-day horizon):
        Days within the available matrix horizon (up to n_days) are summed directly.
        If the remaining horizon falls short of the protection window, the forecast is
        extrapolated proportionally using the average daily rate over the observed window:
        extrapolated_sum = observed_sum * (protection_days / observed_days).
        """
        end = min(t + protection_days, self.n_days)
        obs_days = max(end - t, 1)

        if policy_name == "seasonal_naive":
            fc = self.pred_snaive[:, t:end]
        elif policy_name == "summed_p90":
            fc = self.pred_p90[:, t:end]
        else:  # lightgbm_p50 or lightgbm_p50_safety_stock
            fc = self.pred_p50[:, t:end]

        summed = fc.sum(axis=1)
        if obs_days < protection_days:
            summed = summed * (protection_days / obs_days)

        return summed

    def simulate_policy(
        self,
        policy: str,
        lead_time_days: int,
        service_level: float | None = None,
    ) -> dict[str, Any]:
        """Execute a single replenishment simulation run across all series.

        Parameters
        ----------
        policy : str
            One of: 'seasonal_naive', 'lightgbm_p50',
            'lightgbm_p50_safety_stock', 'summed_p90'.
        lead_time_days : int
            Supplier delivery lead time in days.
        service_level : float | None
            Cycle service level for safety stock calibration (e.g. 0.90, 0.95, 0.98).

        Returns
        -------
        dict[str, Any]
            Dictionary of operational inventory metrics.
        """
        L = lead_time_days
        R = self.config.review_period_days
        protection_days = R + L

        # Calculate safety stock for Policy 3
        if "safety_stock" in policy:
            from scipy.stats import norm

            sl = service_level if service_level is not None else 0.95
            z = float(norm.ppf(sl))
            # Empirical residual standard deviation per store-family series
            res_std = np.std(self.residuals, axis=1)
            safety_stock = np.maximum(0.0, z * res_std * np.sqrt(protection_days)).astype(
                np.float32
            )
        else:
            safety_stock = np.zeros(self.n_series, dtype=np.float32)

        # Warm-start on-hand inventory to S_0 to avoid artificial startup ordering shock
        S_0 = self._get_forecast(policy, 0, protection_days) + safety_stock
        on_hand = S_0.copy()

        # Inflow pipeline dictionary: arrival_day -> order_quantities_array
        pipeline_orders: dict[int, np.ndarray] = {}

        total_sales = 0.0
        total_demand = 0.0
        stockout_series_days = 0
        inventory_sum = 0.0

        for t in range(self.n_days):
            # 1. Receive incoming shipments placed at t - L
            if t in pipeline_orders:
                on_hand += pipeline_orders[t]
                del pipeline_orders[t]

            # 2. Satisfy daily demand (censored by physical on-hand stock)
            d_t = self.sales[:, t]
            fulfilled = np.minimum(on_hand, d_t)
            unmet = d_t - fulfilled

            total_sales += float(fulfilled.sum())
            total_demand += float(d_t.sum())
            stockout_series_days += int(np.count_nonzero(unmet > 1e-4))

            # Deduct sold inventory; on-hand is never negative
            on_hand -= fulfilled
            inventory_sum += float(on_hand.sum())

            # 3. Periodic review order placement on review days (t % R == 0)
            if t % R == 0:
                in_transit = sum(
                    orders for arr_day, orders in pipeline_orders.items() if arr_day > t
                )
                inventory_position = on_hand + in_transit

                S_t = self._get_forecast(policy, t, protection_days) + safety_stock
                order_qty = np.maximum(0.0, S_t - inventory_position)

                arrival_day = t + L
                if arrival_day not in pipeline_orders:
                    pipeline_orders[arrival_day] = order_qty.copy()
                else:
                    pipeline_orders[arrival_day] += order_qty

        fill_rate_pct = 100.0 if total_demand == 0.0 else (total_sales / total_demand) * 100.0
        stockout_share_pct = (stockout_series_days / (self.n_series * self.n_days)) * 100.0
        avg_inventory = inventory_sum / self.n_days
        annualized_sales = total_sales * (365.0 / self.n_days)
        turns = annualized_sales / max(avg_inventory, 1e-6)

        is_sensitivity = L == 14
        policy_label = policy
        if "safety_stock" in policy:
            sl_pct = int(round((service_level or 0.95) * 100))
            policy_label = f"LightGBM P50 + SS ({sl_pct}%)"
        elif policy == "summed_p90":
            policy_label = "Summed Daily P90 (Conservative Upper Bound)"
        elif policy == "lightgbm_p50":
            policy_label = "LightGBM P50"
        elif policy == "seasonal_naive":
            policy_label = "Seasonal Naive (7-Day)"

        return {
            "policy": policy,
            "policy_label": policy_label,
            "lead_time_days": L,
            "is_sensitivity_extrapolation": is_sensitivity,
            "service_level": service_level if "safety_stock" in policy else None,
            "fill_rate_pct": float(round(fill_rate_pct, 2)),
            "stockout_day_share_pct": float(round(stockout_share_pct, 2)),
            "average_inventory_units": float(round(avg_inventory, 2)),
            "inventory_turns": float(round(turns, 2)),
            "total_sales_units": float(round(total_sales, 2)),
            "total_demand_units": float(round(total_demand, 2)),
            "simulation_days": self.n_days,
            "n_store_families": self.n_series,
        }

    def run_all_simulations(self) -> pd.DataFrame:
        """Run all configured policies across lead times and return summary DataFrame."""
        records: list[dict[str, Any]] = []

        for L in self.config.lead_times_days:
            # 1. Seasonal Naive
            records.append(self.simulate_policy("seasonal_naive", lead_time_days=L))

            # 2. LightGBM P50
            records.append(self.simulate_policy("lightgbm_p50", lead_time_days=L))

            # 3. LightGBM P50 + Safety Stock
            for sl in self.config.service_levels:
                records.append(
                    self.simulate_policy(
                        "lightgbm_p50_safety_stock", lead_time_days=L, service_level=sl
                    )
                )

            # 4. Summed daily P90
            records.append(self.simulate_policy("summed_p90", lead_time_days=L))

        return pd.DataFrame(records)


def plot_replenishment_tradeoff(
    sim_df: pd.DataFrame,
    output_path: Path | str = "docs/img/replenishment_policy_tradeoff.png",
) -> Path:
    """Generate trade-off visualization: Fill Rate vs Average Inventory Units."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    style_name = (
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.style.use(style_name)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Distinct markers for lead times
    marker_map = {3: "o", 7: "s", 14: "^"}
    # Palette for policies
    color_map = {
        "seasonal_naive": "#7f7f7f",  # Gray
        "lightgbm_p50": "#1f77b4",  # Blue
        "lightgbm_p50_safety_stock": "#2ca02c",  # Green
        "summed_p90": "#d62728",  # Red
    }

    plotted_labels: set[str] = set()

    for _, row in sim_df.iterrows():
        pol = row["policy"]
        L = row["lead_time_days"]
        x = row["average_inventory_units"] / 1e6  # In Millions of units
        y = row["fill_rate_pct"]

        color = color_map.get(pol, "#17becf")
        marker = marker_map.get(L, "o")

        if pol == "lightgbm_p50_safety_stock":
            sl_pct = int(round(row["service_level"] * 100))
            legend_key = f"LightGBM P50 + SS ({sl_pct}%)"
        elif pol == "summed_p90":
            legend_key = "Summed P90 (Upper Bound)"
        elif pol == "lightgbm_p50":
            legend_key = "LightGBM P50"
        else:
            legend_key = "Seasonal Naive (7-Day)"

        show_legend = legend_key not in plotted_labels
        if show_legend:
            plotted_labels.add(legend_key)

        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=120,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
            label=legend_key if show_legend else None,
            zorder=3,
        )

        # Label sensitivity points
        if L == 14:
            ax.annotate(
                "L=14d*",
                (x, y),
                textcoords="offset points",
                xytext=(6, -4),
                fontsize=7.5,
                color="#555555",
                fontstyle="italic",
            )
        elif L == 3:
            ax.annotate(
                "L=3d",
                (x, y),
                textcoords="offset points",
                xytext=(6, 2),
                fontsize=7.5,
                color="#555555",
            )

    ax.set_title(
        "Replenishment Policy Trade-off: Fill Rate (%) vs. Average Inventory (M Units)\n"
        "96-Day Evaluation (Out-of-Fold + Holdout) | Periodic Review R=7 Days | No Assumed Costs",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Average Working Inventory (Million Units)", fontsize=10, labelpad=8)
    ax.set_ylabel("Simulated Fill Rate (%)", fontsize=10, labelpad=8)

    ax.set_ylim(96.5, 100.0)
    ax.axhline(99.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(
        ax.get_xlim()[0] + 0.1,
        99.05,
        "99.0% Target Fill Rate Threshold",
        fontsize=8,
        color="#666666",
        fontstyle="italic",
    )

    # Lead-time legend handles
    from matplotlib.lines import Line2D

    lt_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Lead Time L=3d",
            markerfacecolor="#444",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="Lead Time L=7d",
            markerfacecolor="#444",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Lead Time L=14d (Sensitivity Extrapolation*)",
            markerfacecolor="#444",
            markersize=8,
        ),
    ]

    h1, l1 = ax.get_legend_handles_labels()
    first_legend = ax.legend(
        h1, l1, loc="lower right", frameon=True, fontsize=8.5, title="Replenishment Policies"
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=lt_legend_elements, loc="center right", frameon=True, fontsize=8, title="Lead Times"
    )

    fig.text(
        0.5,
        0.01,
        (
            "*Note: L=14 days (R+L=21d) exceeds the 16-day forecast horizon and uses weekly "
            "seasonal extrapolation. Sales proxy demand without inventory balance data; "
            "fill rates are optimistic."
        ),
        ha="center",
        fontsize=7.5,
        color="#777777",
    )

    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved replenishment trade-off chart to %s", out_p)
    return out_p
