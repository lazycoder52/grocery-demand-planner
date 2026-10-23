"""Pipeline entry point: ``python -m demand_planner.pipeline``.

Each stage is a callable registered in ``STAGES``. Stages are added by later
pull requests; running with no arguments executes every registered stage in
order.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable

log = logging.getLogger("demand_planner")

StageFn = Callable[[], None]

# Ordered registry of pipeline stages: name -> callable.
STAGES: dict[str, StageFn] = {}


def _register_data_stage() -> None:
    from demand_planner.data import (
        audit_dataset,
        generate_data_card,
        ingest_csvs,
        update_metrics_json,
    )

    def stage_data() -> None:
        con = ingest_csvs()
        audit = audit_dataset(con)
        update_metrics_json(audit, section="audit")
        generate_data_card(audit)

    STAGES["data"] = stage_data


_register_data_stage()


def _register_models_stage() -> None:
    import pandas as pd

    from demand_planner.config import DATA_RAW, OUTPUTS
    from demand_planner.data.audit import update_metrics_json
    from demand_planner.models.lightgbm import evaluate_model_vs_baselines

    def stage_models() -> None:
        raw_train = DATA_RAW / "train.csv"
        if not raw_train.exists():
            log.warning("Raw train.csv not found at %s, skipping models stage", raw_train)
            return

        stores_file = DATA_RAW / "stores.csv"
        holidays_file = DATA_RAW / "holidays_events.csv"
        stores_df = pd.read_csv(stores_file) if stores_file.exists() else None
        holidays_df = pd.read_csv(holidays_file) if holidays_file.exists() else None

        log.info("Loading train data from %s...", raw_train)
        train_df = pd.read_csv(raw_train)
        train_df["date"] = pd.to_datetime(train_df["date"])
        max_date = train_df["date"].max()
        holdout_origin = max_date - pd.Timedelta(days=16)

        log.info("Running 5-fold rolling-origin cross-validation...")
        from demand_planner.models.lightgbm import run_rolling_origin_cv

        cv_results = run_rolling_origin_cv(
            train_df,
            stores_df=stores_df,
            holidays_df=holidays_df,
            start_train_date="2015-01-01",
        )

        log.info("Evaluating final holdout confirmation...")
        holdout_summary, test_preds = evaluate_model_vs_baselines(
            train_df,
            holdout_origin=holdout_origin,
            stores_df=stores_df,
            holidays_df=holidays_df,
            start_train_date="2015-01-01",
        )

        log.info("Running 5-fold rolling-origin cross-validation for LightGBM quantiles...")
        from demand_planner.models.quantile import (
            evaluate_quantile_holdout,
            generate_all_predictions,
            generate_review_forecasts,
            run_quantile_cv,
        )

        quantile_cv_results = run_quantile_cv(
            train_df,
            stores_df=stores_df,
            holidays_df=holidays_df,
            start_train_date="2015-01-01",
        )

        log.info("Evaluating quantile models on holdout...")
        quantile_holdout_summary, quantile_preds = evaluate_quantile_holdout(
            train_df,
            holdout_origin=holdout_origin,
            stores_df=stores_df,
            holidays_df=holidays_df,
            start_train_date="2015-01-01",
        )

        test_preds["pred_p10"] = quantile_preds["pred_p10"]
        test_preds["pred_p50"] = quantile_preds["pred_p50"]
        test_preds["pred_p90"] = quantile_preds["pred_p90"]

        models_data = {
            "model_status": "Selected model",
            "preopening_rows_handling": {
                "training": (
                    "Excluded pre-opening zero rows based on earliest positive sale "
                    "date per store (55,308 rows dropped post-2015)"
                ),
                "evaluation": (
                    "0 pre-opening rows in evaluation; all stores open prior to test "
                    "origin (Store 52 opened 2017-04-20)"
                ),
            },
            "cv_5_fold": cv_results,
            "holdout_confirmation": holdout_summary,
            "quantile_models": {
                "nominal_coverage_pct": 80.0,
                "cv_5_fold": quantile_cv_results,
                "holdout_confirmation": quantile_holdout_summary,
            },
            "by_family": holdout_summary["by_family"],
            "models": holdout_summary["models"],
            "forecast_value_added_vs_seasonal_naive": holdout_summary[
                "forecast_value_added_vs_seasonal_naive"
            ],
            "daytype_breakdown_holdout": holdout_summary.get("daytype_breakdown", {}),
        }

        update_metrics_json(models_data, section="models")

        OUTPUTS.mkdir(parents=True, exist_ok=True)
        out_preds = OUTPUTS / "holdout_predictions.parquet"
        test_preds.to_parquet(out_preds, index=False)
        log.info("Saved holdout predictions to %s (%d rows)", out_preds, len(test_preds))

        oof_path = OUTPUTS / "oof_and_holdout_predictions.parquet"
        log.info("Generating out-of-fold and holdout predictions matrix with pre-origin sigmas...")
        oof_df = generate_all_predictions(
            sales_df=train_df,
            stores_df=stores_df,
            holidays_df=holidays_df,
            start_train_date="2015-01-01",
            exclude_preopening=True,
        )
        oof_df.to_parquet(oof_path, index=False, compression="zstd", compression_level=9)
        log.info("Saved out-of-fold predictions to %s (%d rows)", oof_path, len(oof_df))

        review_path = OUTPUTS / "review_forecasts.parquet"
        log.info("Generating per-review-date forecasts matrix without extrapolation...")
        review_df = generate_review_forecasts(
            sales_df=train_df,
            stores_df=stores_df,
            holidays_df=holidays_df,
            start_train_date="2015-01-01",
            exclude_preopening=True,
        )
        review_df.to_parquet(review_path, index=False, compression="zstd", compression_level=9)
        log.info("Saved review forecasts to %s (%d rows)", review_path, len(review_df))

    STAGES["models"] = stage_models


_register_models_stage()


def _register_simulation_stage() -> None:
    from pathlib import Path

    import pandas as pd

    from demand_planner.config import OUTPUTS
    from demand_planner.data.audit import update_metrics_json
    from demand_planner.simulation import (
        PeriodicReviewSimulator,
        SimConfig,
        compute_rule3_inside_fold_sensitivity,
        plot_replenishment_tradeoff,
    )

    def stage_simulation() -> None:
        import hashlib
        import subprocess
        from datetime import UTC, datetime

        log.info("Starting replenishment simulation stage...")
        cfg = SimConfig.from_yaml("config/sim_config.yaml")
        oof_path = OUTPUTS / "oof_and_holdout_predictions.parquet"
        if not oof_path.exists():
            log.warning("Predictions file %s not found, skipping simulation stage", oof_path)
            return

        df_preds = pd.read_parquet(oof_path)
        review_path = OUTPUTS / "review_forecasts.parquet"
        review_df = pd.read_parquet(review_path) if review_path.exists() else None

        # Collect run metadata
        try:
            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:
            git_hash = "unknown"

        config_path = Path("config/sim_config.yaml")
        config_hash = (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            if config_path.exists()
            else "unknown"
        )

        run_metadata = {
            "git_commit": git_hash,
            "config_hash_sha256": config_hash,
            "seed": cfg.seed,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

        # Rule 2: Default per-review-date re-forecasting without scaling
        sim = PeriodicReviewSimulator(
            df_preds,
            review_forecasts_df=review_df,
            config=cfg,
            stitching_rule="reforecast",
        )
        sim_df = sim.run_all_simulations()

        log.info("Tracing fill-rate vs inventory frontiers across fine z grid...")
        frontier_df = sim.run_frontier(lead_times=(3, 7))

        log.info("Computing target fill rate trade-offs and bootstrap CIs...")
        tradeoffs = sim.compute_target_fill_rate_tradeoff(
            frontier_df, target_fill_rates=(98.5, 99.0, 99.5), n_bootstrap=1000, seed=cfg.seed
        )

        log.info("Computing per-family matched fill-rate frontiers across L in {3, 7}...")
        family_results = sim.simulate_by_family_matched_fill_rate(
            frontier_df=frontier_df, lead_times=(3, 7), target_fill_rates=(98.5, 99.0, 99.5)
        )

        tradeoff_chart = plot_replenishment_tradeoff(sim_df, frontier_df=frontier_df)
        log.info("Saved replenishment trade-off chart to %s", tradeoff_chart)

        # Save small summary parquet (< 15 KB) for tracking
        summary_path = OUTPUTS / "sim_policy_summary.parquet"
        sim_df.to_parquet(summary_path, index=False)
        log.info("Saved tracked policy summary to %s (%d rows)", summary_path, len(sim_df))

        # Compute Sensitivities: Rule 1 (clamped_scaled) and Rule 3 (inside_fold_only)
        log.info("Computing Rule 1 sensitivity ('known stitching artifact')...")
        sim_r1 = PeriodicReviewSimulator(
            df_preds,
            review_forecasts_df=review_df,
            config=cfg,
            stitching_rule="clamped_scaled",
        )
        f_r1 = sim_r1.run_frontier(lead_times=(3, 7))
        tradeoffs_r1 = sim_r1.compute_target_fill_rate_tradeoff(
            f_r1, target_fill_rates=(98.5, 99.0, 99.5), n_bootstrap=1000, seed=cfg.seed
        )

        log.info("Computing Rule 3 sensitivity ('Sensitivity: Inside-Fold Orders')...")
        rule3_sensitivity = compute_rule3_inside_fold_sensitivity(
            df_preds,
            lead_times=(3, 7),
            target_fill_rates=(98.5, 99.0, 99.5),
            n_bootstrap=1000,
            seed=cfg.seed,
        )

        sim_data = {
            "run_metadata": run_metadata,
            "stitching_rule": sim.stitching_rule,
            "simulation_window": {
                "start_date": sim.dates[0].strftime("%Y-%m-%d"),
                "end_date": sim.dates[-1].strftime("%Y-%m-%d"),
                "total_days": sim.n_days,
                "warmup_days_excluded": cfg.warmup_days,
                "evaluated_days": sim.n_days - cfg.warmup_days,
                "n_store_families": sim.n_series,
            },
            "parameters": {
                "review_period_days": cfg.review_period_days,
                "lead_times_days": list(cfg.lead_times_days),
                "service_levels": list(cfg.service_levels),
                "warmup_days": cfg.warmup_days,
                "seed": cfg.seed,
                "stitching_rule": sim.stitching_rule,
                "extrapolation_handling": (
                    "Using per-review-date re-forecasting (stitching_rule='reforecast') "
                    "with 16-day horizon covering R+L=10d and 14d without proportional scaling."
                ),
                "data_limitations": (
                    "Favorita has no inventory data; sales are a demand proxy and are "
                    "censored by stockouts, so fill rates are optimistic; "
                    "results are assumption-driven."
                ),
            },
            "policies": sim_df.to_dict(orient="records"),
            "target_fill_rate_tradeoffs": tradeoffs,
            "per_family_matched_fill_rate": family_results,
            "sensitivities": {
                "rule_1_clamped_scaled_artifact": {
                    "label": "known stitching artifact",
                    "stitching_rule": "clamped_scaled",
                    "description": (
                        "Boundary-clamped and proportionally scaled run-rates across "
                        "fold transitions; known stitching artifact causing rightward "
                        "distortion in base stocks."
                    ),
                    "target_fill_rate_tradeoffs": tradeoffs_r1,
                },
                "rule_3_inside_fold_orders": rule3_sensitivity,
            },
        }

        update_metrics_json(sim_data, section="replenishment_simulation")
        log.info("Saved simulation results, frontiers, metadata, and sensitivities to metrics.json")

    STAGES["simulation"] = stage_simulation


_register_simulation_stage()


def _register_reports_stage() -> None:
    from demand_planner.reports.charts import generate_all_charts

    def stage_reports() -> None:
        log.info("Generating report charts...")
        charts = generate_all_charts()
        for c in charts:
            log.info("Generated chart: %s", c)

    STAGES["reports"] = stage_reports


_register_reports_stage()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m demand_planner.pipeline",
        description="Run the grocery demand planning pipeline.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES) or None,
        help="Subset of stages to run (default: all, in order).",
    )
    parser.add_argument("--list", action="store_true", help="List stages and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.list:
        for name in STAGES:
            print(name)
        return 0
    selected = args.stages or list(STAGES)
    if not selected:
        log.info("No stages registered yet.")
        return 0
    for name in selected:
        start = time.perf_counter()
        log.info("stage %s: start", name)
        STAGES[name]()
        log.info("stage %s: done in %.1fs", name, time.perf_counter() - start)
    return 0
