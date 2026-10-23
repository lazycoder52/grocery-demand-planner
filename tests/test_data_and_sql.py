"""Tests for data ingestion, auditing, and SQL vs Pandas aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from demand_planner.config import DATA_RAW, SQL_DIR
from demand_planner.data.audit import audit_dataset, generate_data_card, update_metrics_json
from demand_planner.data.ingest import ingest_csvs


def test_ingest_tiny_favorita(tiny_favorita_dir: Path):
    """Test DuckDB ingestion on the synthetic Favorita fixture."""
    con = ingest_csvs(raw_dir=tiny_favorita_dir)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    expected_tables = {
        "stores",
        "oil",
        "holidays_events",
        "transactions",
        "train",
        "test",
        "sample_submission",
    }
    assert expected_tables.issubset(set(tables))

    train_cnt = con.execute("SELECT count(*) FROM train").fetchone()[0]
    assert train_cnt > 0
    stores_cnt = con.execute("SELECT count(*) FROM stores").fetchone()[0]
    assert stores_cnt == 5


def test_audit_dataset_on_fixture(tiny_favorita_dir: Path, tmp_path: Path):
    """Test that audit_dataset detects quirks (missing Dec 25, late opening, zero shares)."""
    con = ingest_csvs(raw_dir=tiny_favorita_dir)
    audit = audit_dataset(con)

    # 1. Missing dates includes Christmas
    missing = audit["missing_calendar_dates"]
    assert any("-12-25" in d for d in missing)

    # 2. Late opening store detected
    late = audit["late_opening_stores"]
    late_store_nbrs = [s["store_nbr"] for s in late]
    assert 5 in late_store_nbrs

    # 3. Transferred holiday detected
    assert audit["holidays_summary"]["transferred_total"] >= 1

    # 4. Oil missing values detected
    assert audit["oil_summary"]["missing_prices"] > 0

    # 5. Data card generated
    card_path = tmp_path / "DATA_CARD.md"
    generate_data_card(audit, output_path=card_path)
    assert card_path.exists()
    content = card_path.read_text(encoding="utf-8")
    assert "Missing Calendar Dates" in content
    assert "Late-Opening Stores" in content
    assert "Sales Are Not True Demand" in content

    # 6. Metrics JSON updated
    metrics_path = tmp_path / "metrics.json"
    update_metrics_json(audit, metrics_path=metrics_path)
    assert metrics_path.exists()


def test_sql_equals_pandas_weekly_aggregation(tiny_favorita_dir: Path):
    """Verify that sql/weekly_store_family.sql produces identical results to pandas."""
    con = ingest_csvs(raw_dir=tiny_favorita_dir)

    # 1. Run DuckDB SQL query
    sql_path = SQL_DIR / "weekly_store_family.sql"
    sql_query = sql_path.read_text(encoding="utf-8")
    sql_df = con.execute(sql_query).df()

    # 2. Run equivalent Pandas aggregation
    train_csv = tiny_favorita_dir / "train.csv"
    pdf = pd.read_csv(train_csv)
    pdf["date"] = pd.to_datetime(pdf["date"])

    # Monday-based week start (same as date_trunc('week', date))
    pdf["week_start_date"] = (
        pdf["date"] - pd.to_timedelta(pdf["date"].dt.dayofweek, unit="D")
    ).dt.date

    pandas_agg = (
        pdf.groupby(["store_nbr", "family", "week_start_date"], as_index=False)
        .agg(
            total_sales=("sales", lambda s: round(float(s.sum()), 2)),
            mean_daily_sales=("sales", lambda s: round(float(s.mean()), 2)),
            promo_days=("onpromotion", lambda p: int((p > 0).sum())),
            total_promo_count=("onpromotion", lambda p: int(p.sum())),
            active_days=("sales", "count"),
        )
        .sort_values(["store_nbr", "family", "week_start_date"])
        .reset_index(drop=True)
    )

    # 3. Assert equality
    assert len(sql_df) == len(pandas_agg)
    assert (sql_df["store_nbr"].values == pandas_agg["store_nbr"].values).all()
    assert (sql_df["family"].values == pandas_agg["family"].values).all()
    sql_dates = pd.to_datetime(sql_df["week_start_date"]).dt.strftime("%Y-%m-%d")
    pandas_dates = pd.to_datetime(pandas_agg["week_start_date"]).dt.strftime("%Y-%m-%d")
    assert (sql_dates.values == pandas_dates.values).all()

    # Numeric comparisons
    pd.testing.assert_series_equal(
        sql_df["total_sales"],
        pandas_agg["total_sales"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        sql_df["mean_daily_sales"],
        pandas_agg["mean_daily_sales"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        sql_df["promo_days"].astype(int),
        pandas_agg["promo_days"].astype(int),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        sql_df["total_promo_count"].astype(int),
        pandas_agg["total_promo_count"].astype(int),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        sql_df["active_days"].astype(int),
        pandas_agg["active_days"].astype(int),
        check_names=False,
    )


@pytest.mark.data
def test_real_data_ingest_and_audit():
    """Integration test against the full Kaggle dataset (skipped in CI)."""
    con = ingest_csvs(raw_dir=DATA_RAW)
    audit = audit_dataset(con)

    assert audit["row_counts"]["train"] == 3000888
    assert audit["row_counts"]["test"] == 28512
    assert audit["total_stores"] == 54
    assert audit["total_families"] == 33
    # Exactly 4 missing dates (Christmas 2013, 2014, 2015, 2016)
    assert audit["missing_calendar_dates"] == [
        "2013-12-25",
        "2014-12-25",
        "2015-12-25",
        "2016-12-25",
    ]
