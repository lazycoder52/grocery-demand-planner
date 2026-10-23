"""Periodic-review (R, S) inventory replenishment simulator.

CRITICAL ASSUMPTIONS & LIMITATIONS:
1. Corporación Favorita historical data does not contain physical inventory balances,
   on-hand stock counts, or supplier lead-time variances.
2. Fulfilled store checkout sales are utilized as an observational proxy for customer demand.
3. Because sales are censored whenever physical stockouts occur in historical stores,
   unmet demand cannot be directly observed from point-of-sale logs. Consequently,
   simulated fill rates are optimistic upper bounds relative to true unconstrained demand.
4. All replenishment simulation outcomes are assumption-driven and parameter-contingent.
   No holding costs, obsolescence penalties, or lost-sale penalty costs are assumed or invented.
5. Annualized inventory turns represent an idealized theoretical metric reflecting continuous
   daily fractional ordering; real retail operations face case-pack rounding, minimum order
   quantities (MOQs), pallet constraints, and supplier order schedules.
6. The quantile models (P10, P50, P90) are trained on the log scale (log1p) and inverted via
   expm1. The "LightGBM P50" policy uses the LightGBM quantile regression model fitted at
   alpha=0.5, while point model predictions (L2 conditional mean) are recorded separately.
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
from scipy.interpolate import interp1d
from scipy.stats import norm

log = logging.getLogger("demand_planner.simulation")


@dataclass
class SimConfig:
    """Configuration parameters for periodic-review replenishment simulation."""

    seed: int = 42
    review_period_days: int = 7
    lead_times_days: tuple[int, ...] = (3, 7, 14)
    service_levels: tuple[float, ...] = (0.90, 0.95, 0.98)
    warmup_days: int = 14
    z_grid: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.282,
        1.5,
        1.645,
        1.75,
        2.0,
        2.054,
        2.25,
        2.5,
    )

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
            warmup_days=int(raw.get("warmup_days", 14)),
        )


class PeriodicReviewSimulator:
    """Simulates periodic-review (R, S) order-up-to inventory replenishment.

    Parameters
    ----------
    df_predictions : pd.DataFrame
        DataFrame containing out-of-fold and holdout predictions. Must contain:
        ['date', 'store_nbr', 'family', 'sales', 'pred_lgbm', 'pred_snaive',
         'pred_p10', 'pred_p50', 'pred_p90'].
        May optionally contain 'sigma_lgbm' and 'sigma_snaive' containing strictly
        pre-origin empirical residual standard deviations.
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

        # LightGBM point model predictions (L2 conditional mean)
        if "pred_lgbm" in df.columns:
            self.pred_point = df.pivot(
                index=["store_nbr", "family"], columns="day_idx", values="pred_lgbm"
            ).to_numpy(dtype=np.float32)
        else:
            self.pred_point = self.pred_p50.copy()

        # Map each day to the end of its active fold to prevent future-fold model leakage
        if "split" in df.columns:
            split_max_day = df.groupby("split", observed=False)["day_idx"].max().to_dict()
            day_split = df.groupby("day_idx")["split"].first().to_dict()
            self.active_fold_end_idx = np.array(
                [split_max_day[day_split[i]] for i in range(self.n_days)], dtype=int
            )
        else:
            self.active_fold_end_idx = np.full(self.n_days, self.n_days - 1, dtype=int)

        # Pre-origin empirical residual standard deviations (strictly date <= split.origin_date)
        if "sigma_lgbm" in df.columns:
            self.sigma_lgbm = df.pivot(
                index=["store_nbr", "family"], columns="day_idx", values="sigma_lgbm"
            ).to_numpy(dtype=np.float32)
        else:
            # Fallback for standalone mock test fixtures
            res_l = self.sales - self.pred_p50
            self.sigma_lgbm = np.repeat(
                np.std(res_l[:, : min(7, self.n_days)], axis=1, keepdims=True),
                self.n_days,
                axis=1,
            ).astype(np.float32)

        if "sigma_snaive" in df.columns:
            self.sigma_snaive = df.pivot(
                index=["store_nbr", "family"], columns="day_idx", values="sigma_snaive"
            ).to_numpy(dtype=np.float32)
        else:
            res_s = self.sales - self.pred_snaive
            self.sigma_snaive = np.repeat(
                np.std(res_s[:, : min(7, self.n_days)], axis=1, keepdims=True),
                self.n_days,
                axis=1,
            ).astype(np.float32)

    def _get_forecast(
        self,
        policy_name: str,
        t: int,
        protection_days: int,
    ) -> np.ndarray:
        """Sum forecasted demand over protection interval [t, t + protection_days].

        Strictly clamps the forecast slice to the end of the active fold containing day t
        so that no future-fold models (whose origin date > order date) can be accessed.
        If the protection window exceeds the remaining days of the active fold, the available
        causal forecast is scaled proportionally over the full protection window.
        """
        fold_end = self.active_fold_end_idx[t]
        end = min(t + protection_days, fold_end + 1, self.n_days)
        obs_days = max(end - t, 1)

        if "seasonal_naive" in policy_name:
            fc = self.pred_snaive[:, t:end]
        elif "point" in policy_name or "pred_lgbm" in policy_name:
            fc = self.pred_point[:, t:end]
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
        z_value: float | None = None,
        return_series_metrics: bool = False,
    ) -> dict[str, Any]:
        """Execute a single replenishment simulation run across all series."""
        L = lead_time_days
        R = self.config.review_period_days
        protection_days = R + L

        # Calculate safety stock factor z
        z = 0.0
        if "safety_stock" in policy:
            if z_value is not None:
                z = float(z_value)
            elif service_level is not None:
                z = float(norm.ppf(service_level))
            else:
                z = float(norm.ppf(0.95))

        # Select pre-origin calibrated residual standard deviation matrix
        if ("lightgbm" in policy or "point" in policy) and "safety_stock" in policy:
            sig_mat = self.sigma_lgbm
        elif "seasonal_naive" in policy and "safety_stock" in policy:
            sig_mat = self.sigma_snaive
        else:
            sig_mat = np.zeros_like(self.sales)

        # Initial inventory warm-start: S_0 = D_hat(0, R+L) + SS_0
        SS_0 = np.maximum(0.0, z * sig_mat[:, 0] * np.sqrt(protection_days)).astype(np.float32)
        S_0 = self._get_forecast(policy, 0, protection_days) + SS_0
        on_hand = S_0.copy()

        # Inflow pipeline dictionary: arrival_day -> order_quantities_array
        pipeline_orders: dict[int, np.ndarray] = {}

        # Warm-up handling: exclude first warmup_days from metric accumulation
        eval_start = min(self.config.warmup_days, self.n_days - 1)
        eval_days = self.n_days - eval_start

        total_sales = 0.0
        total_demand = 0.0
        stockout_series_days = 0
        inventory_sum = 0.0

        # Per-series arrays for bootstrap and family breakdown
        series_sales = np.zeros(self.n_series, dtype=np.float64)
        series_demand = np.zeros(self.n_series, dtype=np.float64)
        series_inv_sum = np.zeros(self.n_series, dtype=np.float64)
        series_stockouts = np.zeros(self.n_series, dtype=np.int32)

        for t in range(self.n_days):
            # 1. Receive incoming shipments placed at t - L
            if t in pipeline_orders:
                on_hand += pipeline_orders[t]
                del pipeline_orders[t]

            # 2. Satisfy daily demand (censored by physical on-hand stock)
            d_t = self.sales[:, t]
            fulfilled = np.minimum(on_hand, d_t)
            unmet = d_t - fulfilled

            if t >= eval_start:
                total_sales += float(fulfilled.sum())
                total_demand += float(d_t.sum())
                stockout_series_days += int(np.count_nonzero(unmet > 1e-4))

                series_sales += fulfilled.astype(np.float64)
                series_demand += d_t.astype(np.float64)
                series_stockouts += (unmet > 1e-4).astype(np.int32)

            # Deduct sold inventory; on-hand is never negative
            on_hand -= fulfilled
            if t >= eval_start:
                inventory_sum += float(on_hand.sum())
                series_inv_sum += on_hand.astype(np.float64)

            # 3. Periodic review order placement on review days (t % R == 0)
            if t % R == 0:
                in_transit = sum(
                    orders for arr_day, orders in pipeline_orders.items() if arr_day > t
                )
                inventory_position = on_hand + in_transit

                SS_t = np.maximum(0.0, z * sig_mat[:, t] * np.sqrt(protection_days)).astype(
                    np.float32
                )
                S_t = self._get_forecast(policy, t, protection_days) + SS_t
                order_qty = np.maximum(0.0, S_t - inventory_position)

                arrival_day = t + L
                if arrival_day not in pipeline_orders:
                    pipeline_orders[arrival_day] = order_qty.copy()
                else:
                    pipeline_orders[arrival_day] += order_qty

        fill_rate_pct = 100.0 if total_demand == 0.0 else (total_sales / total_demand) * 100.0
        stockout_share_pct = (stockout_series_days / (self.n_series * eval_days)) * 100.0
        avg_inventory = inventory_sum / eval_days
        annualized_sales = total_sales * (365.0 / eval_days)
        turns = annualized_sales / max(avg_inventory, 1e-6)

        is_sensitivity = L == 14
        policy_label = policy
        if "safety_stock" in policy:
            if "point" in policy:
                label_prefix = "LightGBM Point + SS"
            elif "lightgbm" in policy:
                label_prefix = "LightGBM P50 + SS"
            else:
                label_prefix = "Seasonal Naive + SS"
            if service_level is not None:
                sl_pct = int(round(service_level * 100))
                policy_label = f"{label_prefix} ({sl_pct}%)"
            else:
                policy_label = f"{label_prefix} (z={z:.2f})"
        elif policy == "summed_p90":
            policy_label = "Summed Daily P90 (Conservative Upper Bound)"
        elif policy == "lightgbm_p50":
            policy_label = "LightGBM P50"
        elif policy in ("lightgbm_point", "point"):
            policy_label = "LightGBM Point"
        elif policy == "seasonal_naive":
            policy_label = "Seasonal Naive (7-Day)"

        out: dict[str, Any] = {
            "policy": policy,
            "policy_label": policy_label,
            "lead_time_days": L,
            "is_sensitivity_extrapolation": is_sensitivity,
            "service_level": service_level if "safety_stock" in policy else None,
            "z_value": z if "safety_stock" in policy else 0.0,
            "fill_rate_pct": float(round(fill_rate_pct, 2)),
            "stockout_day_share_pct": float(round(stockout_share_pct, 2)),
            "average_inventory_units": float(round(avg_inventory, 2)),
            "inventory_turns": float(round(turns, 2)),
            "inventory_turns_note": (
                "Idealized metric under continuous fractional ordering without "
                "case packs, MOQs, or warehouse dispatch constraints."
            ),
            "total_sales_units": float(round(total_sales, 2)),
            "total_demand_units": float(round(total_demand, 2)),
            "simulation_days": self.n_days,
            "evaluated_days": eval_days,
            "warmup_days_excluded": eval_start,
            "n_store_families": self.n_series,
        }

        if return_series_metrics:
            out["series_metrics"] = {
                "sales": series_sales,
                "demand": series_demand,
                "inv_sum": series_inv_sum,
                "stockouts": series_stockouts,
                "eval_days": eval_days,
            }

        return out

    def run_all_simulations(self) -> pd.DataFrame:
        """Run standard configured policies across lead times and return summary DataFrame."""
        records: list[dict[str, Any]] = []

        for L in self.config.lead_times_days:
            # 1. Seasonal Naive point
            records.append(self.simulate_policy("seasonal_naive", lead_time_days=L))

            # 2. LightGBM P50 point
            records.append(self.simulate_policy("lightgbm_p50", lead_time_days=L))

            # 3. LightGBM P50 + Safety Stock
            for sl in self.config.service_levels:
                records.append(
                    self.simulate_policy(
                        "lightgbm_p50_safety_stock", lead_time_days=L, service_level=sl
                    )
                )

            # 4. Seasonal Naive + Safety Stock (matched service levels)
            for sl in self.config.service_levels:
                records.append(
                    self.simulate_policy(
                        "seasonal_naive_safety_stock", lead_time_days=L, service_level=sl
                    )
                )

            # 5. Summed daily P90
            records.append(self.simulate_policy("summed_p90", lead_time_days=L))

        return pd.DataFrame(records)

    def run_frontier(
        self,
        lead_times: tuple[int, ...] = (3, 7),
        z_values: tuple[float, ...] | None = None,
    ) -> pd.DataFrame:
        """Trace the fill-rate vs average-inventory frontier over a fine z grid."""
        grid = z_values or self.config.z_grid
        records: list[dict[str, Any]] = []

        for L in lead_times:
            for z in grid:
                res_lgbm = self.simulate_policy(
                    "lightgbm_p50_safety_stock",
                    lead_time_days=L,
                    z_value=z,
                    return_series_metrics=True,
                )
                res_snaive = self.simulate_policy(
                    "seasonal_naive_safety_stock",
                    lead_time_days=L,
                    z_value=z,
                    return_series_metrics=True,
                )
                records.append(res_lgbm)
                records.append(res_snaive)

        return pd.DataFrame(records)

    def compute_target_fill_rate_tradeoff(
        self,
        frontier_df: pd.DataFrame,
        target_fill_rates: tuple[float, ...] = (98.5, 99.0, 99.5),
        n_bootstrap: int = 200,
        lead_times: tuple[int, ...] = (3, 7),
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Interpolate required inventory to reach target fill rates with bootstrap CIs."""
        rng = np.random.default_rng(seed)
        results: list[dict[str, Any]] = []

        for L in lead_times:
            df_l = frontier_df[frontier_df["lead_time_days"] == L]
            df_lgbm = df_l[df_l["policy"] == "lightgbm_p50_safety_stock"].sort_values(
                "fill_rate_pct"
            )
            df_snaive = df_l[df_l["policy"] == "seasonal_naive_safety_stock"].sort_values(
                "fill_rate_pct"
            )

            # Point interpolation on aggregate curves
            fr_lgbm = df_lgbm["fill_rate_pct"].to_numpy()
            inv_lgbm = df_lgbm["average_inventory_units"].to_numpy()

            fr_snaive = df_snaive["fill_rate_pct"].to_numpy()
            inv_snaive = df_snaive["average_inventory_units"].to_numpy()

            interp_lgbm = interp1d(
                fr_lgbm, inv_lgbm, bounds_error=False, fill_value=(inv_lgbm[0], inv_lgbm[-1])
            )
            interp_snaive = interp1d(
                fr_snaive,
                inv_snaive,
                bounds_error=False,
                fill_value=(inv_snaive[0], inv_snaive[-1]),
            )

            # Pre-extract series metrics for fast bootstrapping
            series_data_lgbm = [row["series_metrics"] for _, row in df_lgbm.iterrows()]
            series_data_snaive = [row["series_metrics"] for _, row in df_snaive.iterrows()]

            for target_fr in target_fill_rates:
                req_inv_lgbm = float(interp_lgbm(target_fr))
                req_inv_snaive = float(interp_snaive(target_fr))
                pct_diff = ((req_inv_lgbm - req_inv_snaive) / req_inv_snaive) * 100.0

                # Block bootstrap over series
                boot_diffs: list[float] = []
                boot_lgbm_invs: list[float] = []
                boot_snaive_invs: list[float] = []

                for _ in range(n_bootstrap):
                    idx = rng.integers(0, self.n_series, size=self.n_series)

                    b_fr_lgbm: list[float] = []
                    b_inv_lgbm: list[float] = []
                    for m in series_data_lgbm:
                        s_tot = m["sales"][idx].sum()
                        d_tot = m["demand"][idx].sum()
                        fr = (s_tot / max(d_tot, 1e-6)) * 100.0
                        inv = m["inv_sum"][idx].sum() / m["eval_days"]
                        b_fr_lgbm.append(fr)
                        b_inv_lgbm.append(inv)

                    b_fr_snaive: list[float] = []
                    b_inv_snaive: list[float] = []
                    for m in series_data_snaive:
                        s_tot = m["sales"][idx].sum()
                        d_tot = m["demand"][idx].sum()
                        fr = (s_tot / max(d_tot, 1e-6)) * 100.0
                        inv = m["inv_sum"][idx].sum() / m["eval_days"]
                        b_fr_snaive.append(fr)
                        b_inv_snaive.append(inv)

                    b_interp_l = interp1d(
                        b_fr_lgbm,
                        b_inv_lgbm,
                        bounds_error=False,
                        fill_value=(b_inv_lgbm[0], b_inv_lgbm[-1]),
                    )
                    b_interp_s = interp1d(
                        b_fr_snaive,
                        b_inv_snaive,
                        bounds_error=False,
                        fill_value=(b_inv_snaive[0], b_inv_snaive[-1]),
                    )

                    b_req_l = float(b_interp_l(target_fr))
                    b_req_s = float(b_interp_s(target_fr))
                    b_diff = ((b_req_l - b_req_s) / max(b_req_s, 1e-6)) * 100.0

                    boot_lgbm_invs.append(b_req_l)
                    boot_snaive_invs.append(b_req_s)
                    boot_diffs.append(b_diff)

                ci_diff_low = float(np.percentile(boot_diffs, 2.5))
                ci_diff_high = float(np.percentile(boot_diffs, 97.5))
                is_within_noise = ci_diff_low <= 0.0 <= ci_diff_high

                results.append(
                    {
                        "lead_time_days": L,
                        "target_fill_rate_pct": target_fr,
                        "required_inventory_lightgbm": req_inv_lgbm,
                        "required_inventory_snaive": req_inv_snaive,
                        "pct_difference": float(round(pct_diff, 2)),
                        "ci_95_pct_diff_low": float(round(ci_diff_low, 2)),
                        "ci_95_pct_diff_high": float(round(ci_diff_high, 2)),
                        "is_within_noise": is_within_noise,
                        "interpretation": (
                            "Difference is within statistical noise (95% CI spans 0)"
                            if is_within_noise
                            else (
                                f"LightGBM saves {abs(pct_diff):.2f}% inventory "
                                f"(95% CI: [{ci_diff_low:.2f}%, {ci_diff_high:.2f}%])"
                                if pct_diff < 0
                                else f"Seasonal Naive uses {abs(pct_diff):.2f}% less inventory"
                            )
                        ),
                    }
                )

        return results

    def simulate_by_family(
        self,
        lead_time_days: int = 3,
        service_level: float = 0.95,
        families: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Compute per-family replenishment metrics for LightGBM+SS vs Seasonal Naive+SS."""
        target_fams = families or [
            "GROCERY I",
            "BEVERAGES",
            "PRODUCE",
            "CLEANING",
            "DAIRY",
            "BREAD/BAKERY",
            "POULTRY",
            "MEATS",
            "DELI",
            "PERSONAL CARE",
            "BABY CARE",
            "BOOKS",
            "LAWN AND GARDEN",
        ]

        res_lgbm = self.simulate_policy(
            "lightgbm_p50_safety_stock",
            lead_time_days=lead_time_days,
            service_level=service_level,
            return_series_metrics=True,
        )
        res_snaive = self.simulate_policy(
            "seasonal_naive_safety_stock",
            lead_time_days=lead_time_days,
            service_level=service_level,
            return_series_metrics=True,
        )

        m_l = res_lgbm["series_metrics"]
        m_s = res_snaive["series_metrics"]
        eval_days = m_l["eval_days"]

        family_series = self.series_keys["family"].to_numpy()
        results: list[dict[str, Any]] = []

        for fam in target_fams:
            idx = np.where(family_series == fam)[0]
            if len(idx) == 0:
                continue

            # LightGBM metrics for family
            dem_l = float(m_l["demand"][idx].sum())
            sal_l = float(m_l["sales"][idx].sum())
            inv_l = float(m_l["inv_sum"][idx].sum() / eval_days)
            stk_l = int(m_l["stockouts"][idx].sum())
            fr_l = (sal_l / max(dem_l, 1e-6)) * 100.0
            stk_pct_l = (stk_l / (len(idx) * eval_days)) * 100.0

            # Seasonal Naive metrics for family
            dem_s = float(m_s["demand"][idx].sum())
            sal_s = float(m_s["sales"][idx].sum())
            inv_s = float(m_s["inv_sum"][idx].sum() / eval_days)
            stk_s = int(m_s["stockouts"][idx].sum())
            fr_s = (sal_s / max(dem_s, 1e-6)) * 100.0
            stk_pct_s = (stk_s / (len(idx) * eval_days)) * 100.0

            inv_diff_pct = ((inv_l - inv_s) / max(inv_s, 1e-6)) * 100.0

            results.append(
                {
                    "family": fam,
                    "lead_time_days": lead_time_days,
                    "service_level": service_level,
                    "lightgbm": {
                        "fill_rate_pct": float(round(fr_l, 2)),
                        "average_inventory_units": float(round(inv_l, 2)),
                        "stockout_day_share_pct": float(round(stk_pct_l, 2)),
                    },
                    "seasonal_naive": {
                        "fill_rate_pct": float(round(fr_s, 2)),
                        "average_inventory_units": float(round(inv_s, 2)),
                        "stockout_day_share_pct": float(round(stk_pct_s, 2)),
                    },
                    "inventory_difference_pct": float(round(inv_diff_pct, 2)),
                }
            )

        return results


def plot_replenishment_tradeoff(
    sim_df: pd.DataFrame,
    frontier_df: pd.DataFrame | None = None,
    output_path: Path | str = "docs/img/replenishment_policy_tradeoff.png",
) -> Path:
    """Generate trade-off visualization: Fill Rate vs Average Inventory Units."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    style_name = (
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.style.use(style_name)
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)

    # Plot continuous frontiers if provided
    if frontier_df is not None and len(frontier_df) > 0:
        for L, ls in [(3, "-"), (7, "--")]:
            df_fl = frontier_df[frontier_df["lead_time_days"] == L]

            # LightGBM line
            fl_lgbm = df_fl[df_fl["policy"] == "lightgbm_p50_safety_stock"].sort_values(
                "average_inventory_units"
            )
            if len(fl_lgbm) > 0:
                ax.plot(
                    fl_lgbm["average_inventory_units"] / 1e6,
                    fl_lgbm["fill_rate_pct"],
                    color="#1f77b4",
                    linestyle=ls,
                    linewidth=2.0,
                    alpha=0.85,
                    label=f"LightGBM + SS Frontier (L={L}d)",
                    zorder=2,
                )

            # Seasonal Naive line
            fl_snaive = df_fl[df_fl["policy"] == "seasonal_naive_safety_stock"].sort_values(
                "average_inventory_units"
            )
            if len(fl_snaive) > 0:
                ax.plot(
                    fl_snaive["average_inventory_units"] / 1e6,
                    fl_snaive["fill_rate_pct"],
                    color="#ff7f0e",
                    linestyle=ls,
                    linewidth=2.0,
                    alpha=0.85,
                    label=f"Seasonal Naive + SS Frontier (L={L}d)",
                    zorder=2,
                )

    # Plot discrete policy points
    marker_map = {3: "o", 7: "s", 14: "^"}
    color_map = {
        "seasonal_naive": "#7f7f7f",
        "seasonal_naive_safety_stock": "#ff7f0e",
        "lightgbm_p50": "#17becf",
        "lightgbm_p50_safety_stock": "#1f77b4",
        "summed_p90": "#d62728",
    }

    plotted_labels: set[str] = set()

    for _, row in sim_df.iterrows():
        pol = row["policy"]
        L = row["lead_time_days"]
        x = row["average_inventory_units"] / 1e6
        y = row["fill_rate_pct"]

        color = color_map.get(pol, "#bcbd22")
        marker = marker_map.get(L, "o")

        label_key = row.get("policy_label", pol)
        show_legend = label_key not in plotted_labels and frontier_df is None
        if show_legend:
            plotted_labels.add(label_key)

        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=110,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
            label=label_key if show_legend else None,
            zorder=4,
        )

        if L == 14:
            ax.annotate(
                "L=14d*",
                (x, y),
                textcoords="offset points",
                xytext=(6, -4),
                fontsize=8,
                color="#555555",
            )

    ax.set_title(
        "Periodic-Review (R=7) Inventory Replenishment Trade-Off Frontier\n"
        "Fill Rate vs Average Chain Inventory Units Across Lead Times (Zero Cost Assumptions)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Average Total Chain Inventory (Millions of Units)", fontsize=10, labelpad=8)
    ax.set_ylabel("Simulated Fill Rate (%) [Optimistic Upper Bound]", fontsize=10, labelpad=8)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5, loc="lower right")

    fig.text(
        0.5,
        0.01,
        "*Note: L=14d uses labelled extrapolation (21d > 16d model horizon). "
        "Sales proxy demand without inventory logs; fill rates are optimistic upper bounds.",
        ha="center",
        fontsize=8,
        color="#666666",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    return out_p
