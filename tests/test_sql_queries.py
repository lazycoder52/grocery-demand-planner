"""Tests for analytical SQL queries in sql/."""

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


def test_zero_sales_day_proxy_sql(tiny_favorita_dir: Path):
    """Test that zero_sales_day_proxy.sql executes cleanly and categorizes proxy tiers."""
    con = ingest_csvs(raw_dir=tiny_favorita_dir)
    query = (SQL_DIR / "zero_sales_day_proxy.sql").read_text(encoding="utf-8")
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
        "zero_sales_proxy_tier",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0
    assert (df["zero_sales_rate_pct"] >= 0.0).all()
    assert (df["zero_sales_rate_pct"] <= 100.0).all()
    valid_tiers = {
        "HIGH_VELOCITY_ELEVATED_ZERO_SALES_DAYS_PROXY",
        "MODERATE_ZERO_SALES_DAYS_PROXY",
        "SLOW_MOVING_INTERMITTENT_TAIL",
        "LOW_ZERO_SALES_DAYS",
    }
    assert set(df["zero_sales_proxy_tier"].unique()).issubset(valid_tiers)


def test_perishable_volatility_proxy_sql(tiny_favorita_dir: Path):
    """Test that perishable_volatility_proxy.sql executes cleanly and computes CV."""
    con = ingest_csvs(raw_dir=tiny_favorita_dir)
    query = (SQL_DIR / "perishable_volatility_proxy.sql").read_text(encoding="utf-8")
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
        "volatility_exposure_index",
        "volatility_proxy_tier",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0
    assert set(df["perishability_class"].unique()).issubset({"PERISHABLE", "NON_PERISHABLE"})
    valid_tiers = {
        "HIGH_VOLATILITY_PERISHABLE_PROXY",
        "MODERATE_VOLATILITY_PERISHABLE_PROXY",
        "NON_PERISHABLE_OR_LOW_VOLATILITY",
    }
    assert set(df["volatility_proxy_tier"].unique()).issubset(valid_tiers)


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
