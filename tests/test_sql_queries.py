"""Tests for production SQL queries in sql/."""

from __future__ import annotations

from pathlib import Path

from demand_planner.config import SQL_DIR
from demand_planner.data.ingest import ingest_csvs


def test_promo_elasticity_sql(tiny_favorita_dir: Path):
    """Test that promo_elasticity.sql executes cleanly and computes valid metrics."""
    con = ingest_csvs(raw_dir=tiny_favorita_dir)
    query = (SQL_DIR / "promo_elasticity.sql").read_text(encoding="utf-8")
    # For tiny_favorita synthetic data, loosen date filter if needed or verify execution
    df = con.execute(query).df()
    expected_cols = {
        "store_nbr",
        "city",
        "state",
        "store_type",
        "store_cluster",
        "family",
        "total_evaluated_days",
        "promo_days",
        "non_promo_days",
        "promo_day_share_pct",
        "promo_mean_daily_sales",
        "non_promo_mean_daily_sales",
        "promo_total_sales",
        "total_sales",
        "empirical_promo_lift_pct",
    }
    assert expected_cols.issubset(set(df.columns))
    if len(df) > 0:
        assert (df["promo_days"] >= 0).all()
        assert (df["promo_day_share_pct"] >= 0.0).all()
        assert (df["promo_day_share_pct"] <= 100.0).all()


def test_stockout_risk_sql(tiny_favorita_dir: Path):
    """Test that stockout_risk.sql executes cleanly and categorizes risk tiers."""
    con = ingest_csvs(raw_dir=tiny_favorita_dir)
    query = (SQL_DIR / "stockout_risk.sql").read_text(encoding="utf-8")
    df = con.execute(query).df()
    expected_cols = {
        "store_nbr",
        "city",
        "state",
        "store_type",
        "family",
        "total_days",
        "zero_sales_days",
        "zero_sales_rate_pct",
        "mean_daily_sales",
        "active_mean_daily_sales",
        "max_daily_sales",
        "total_sales",
        "stockout_risk_segment",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0
    assert (df["zero_sales_rate_pct"] >= 0.0).all()
    assert (df["zero_sales_rate_pct"] <= 100.0).all()
    valid_segments = {
        "CRITICAL_HIGH_VELOCITY_STOCKOUT_RISK",
        "MODERATE_STOCKOUT_RISK",
        "SLOW_MOVING_INTERMITTENT_TAIL",
        "STABLE_IN_STOCK",
    }
    assert set(df["stockout_risk_segment"].unique()).issubset(valid_segments)


def test_shrinkage_or_waste_proxy_sql(tiny_favorita_dir: Path):
    """Test that shrinkage_or_waste_proxy.sql executes cleanly and computes CV."""
    con = ingest_csvs(raw_dir=tiny_favorita_dir)
    query = (SQL_DIR / "shrinkage_or_waste_proxy.sql").read_text(encoding="utf-8")
    df = con.execute(query).df()
    expected_cols = {
        "store_nbr",
        "city",
        "store_type",
        "family",
        "perishability_class",
        "total_days",
        "mean_daily_sales",
        "std_daily_sales",
        "cv_demand",
        "max_daily_sales",
        "zero_sales_days",
        "spoilage_exposure_index",
        "spoilage_risk_tier",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0
    assert set(df["perishability_class"].unique()).issubset({"PERISHABLE", "NON_PERISHABLE"})
    valid_tiers = {"HIGH_SPOILAGE_RISK", "MODERATE_SPOILAGE_RISK", "LOW_NON_PERISHABLE"}
    assert set(df["spoilage_risk_tier"].unique()).issubset(valid_tiers)


def test_weekly_store_family_sql_joined_columns(tiny_favorita_dir: Path):
    """Test that weekly_store_family.sql joins sales, stores, transactions, oil, holidays."""
    con = ingest_csvs(raw_dir=tiny_favorita_dir)
    query = (SQL_DIR / "weekly_store_family.sql").read_text(encoding="utf-8")
    df = con.execute(query).df()
    expected_cols = {
        "store_nbr",
        "city",
        "state",
        "store_type",
        "store_cluster",
        "family",
        "week_start_date",
        "total_sales",
        "mean_daily_sales",
        "promo_days",
        "total_promo_count",
        "active_days",
        "weekly_transactions",
        "avg_oil_price",
        "holiday_count",
        "non_transferred_holidays",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0
    assert not df["city"].isna().all()
    assert not df["store_type"].isna().all()
