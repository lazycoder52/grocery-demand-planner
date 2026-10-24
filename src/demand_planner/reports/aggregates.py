"""Export tracked aggregate datasets for headless Streamlit app and lightweight consumption.

Generates compact parquet aggregates (< 5 MB total) so the Streamlit dashboard
and interactive tools can run completely decoupled from multi-gigabyte raw CSVs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from demand_planner.config import OUTPUTS, SQL_DIR

log = logging.getLogger("demand_planner")


def export_forecast_sample_summary(
    oof_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Downsample and compress model prediction matrix to compact parquet summary (< 2 MB)."""
    oof_path = oof_path or (OUTPUTS / "oof_and_holdout_predictions.parquet")
    output_path = output_path or (OUTPUTS / "forecast_sample_summary.parquet")

    if not oof_path.exists():
        log.warning("OOF predictions file %s not found; skipping forecast summary export", oof_path)
        return output_path

    df = pd.read_parquet(oof_path)
    cols = ["date", "store_nbr", "family", "sales", "pred_lgbm", "pred_snaive", "split"]
    cols_present = [c for c in cols if c in df.columns]
    sub = df[cols_present].copy()

    # Downcast datatypes to minimize disk footprint
    if "store_nbr" in sub.columns:
        sub["store_nbr"] = sub["store_nbr"].astype("int16")
    if "family" in sub.columns:
        sub["family"] = sub["family"].astype("category")
    if "sales" in sub.columns:
        sub["sales"] = sub["sales"].astype("float32")
    if "pred_lgbm" in sub.columns:
        sub["pred_lgbm"] = sub["pred_lgbm"].astype("float32")
    if "pred_snaive" in sub.columns:
        sub["pred_snaive"] = sub["pred_snaive"].astype("float32")
    if "split" in sub.columns:
        sub["split"] = sub["split"].astype("category")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(output_path, index=False, compression="zstd", compression_level=9)
    log.info(
        "Exported compact forecast summary to %s (%d rows, %.2f MB)",
        output_path,
        len(sub),
        output_path.stat().st_size / (1024 * 1024),
    )
    return output_path


def export_sql_analytical_summaries(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Execute SQL analytical queries in DuckDB and export compact parquet views (< 3 MB total)."""
    from demand_planner.config import DATA_RAW
    from demand_planner.data.ingest import ingest_csvs

    raw_dir = raw_dir or DATA_RAW
    output_dir = output_dir or OUTPUTS
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (raw_dir / "train.csv").exists():
        log.warning("Raw data missing at %s; skipping SQL analytical summaries export", raw_dir)
        return {}

    con = ingest_csvs(raw_dir=raw_dir)

    queries = {
        "promo_elasticity_summary.parquet": SQL_DIR / "promo_elasticity.sql",
        "zero_sales_day_proxy_summary.parquet": SQL_DIR / "zero_sales_day_proxy.sql",
        "perishable_volatility_summary.parquet": SQL_DIR / "perishable_volatility_proxy.sql",
        "weekly_store_family_summary.parquet": SQL_DIR / "weekly_store_family.sql",
    }

    exported: dict[str, Path] = {}
    for filename, sql_path in queries.items():
        if not sql_path.exists():
            log.warning("SQL file %s not found", sql_path)
            continue

        query_sql = sql_path.read_text(encoding="utf-8")
        df: pd.DataFrame = con.execute(query_sql).df()

        # Optimize column datatypes for parquet storage
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category")
            elif "int" in str(df[col].dtype):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif "float" in str(df[col].dtype):
                df[col] = df[col].astype("float32")

        target_path = output_dir / filename
        df.to_parquet(target_path, index=False, compression="zstd", compression_level=9)
        exported[filename] = target_path
        log.info(
            "Exported SQL view %s to %s (%d rows, %.2f KB)",
            sql_path.name,
            target_path,
            len(df),
            target_path.stat().st_size / 1024,
        )

    return exported


def export_all_app_aggregates() -> None:
    """Convenience orchestrator to export all tracked aggregates required by app.py."""
    export_forecast_sample_summary()
    export_sql_analytical_summaries()
