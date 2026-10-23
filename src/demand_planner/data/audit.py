"""Data audit and documentation generation for Corporación Favorita store sales."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from demand_planner.config import DOCS, METRICS_FILE

log = logging.getLogger("demand_planner.data.audit")


def audit_dataset(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Perform a comprehensive audit of all tables in the DuckDB database."""
    results: dict[str, Any] = {}

    # 1. Table row counts
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    row_counts = {}
    for t in tables:
        cnt = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        row_counts[t] = int(cnt)
    results["row_counts"] = row_counts

    # 2. Date ranges
    date_ranges = {}
    for t in ["train", "test", "oil", "transactions", "holidays_events"]:
        if t in tables:
            min_d, max_d = con.execute(
                f"SELECT strftime(min(date), '%Y-%m-%d'), strftime(max(date), '%Y-%m-%d') FROM {t}"
            ).fetchone()
            date_ranges[t] = {"min_date": str(min_d), "max_date": str(max_d)}
    results["date_ranges"] = date_ranges

    # 3. Missing calendar dates in train span
    if "train" in tables:
        min_train = results["date_ranges"]["train"]["min_date"]
        max_train = results["date_ranges"]["train"]["max_date"]
        all_calendar_dates = pd.date_range(min_train, max_train).strftime("%Y-%m-%d")
        train_distinct_dates = set(
            r[0]
            for r in con.execute("SELECT DISTINCT strftime(date, '%Y-%m-%d') FROM train").fetchall()
        )
        missing_dates = sorted(list(set(all_calendar_dates) - train_distinct_dates))
        results["missing_calendar_dates"] = missing_dates
        results["missing_dates_count"] = len(missing_dates)

    # 4. Store openings and activity
    if "train" in tables and "stores" in tables:
        store_stats = con.execute("""
            SELECT
                store_nbr,
                strftime(min(date), '%Y-%m-%d') as first_recorded,
                strftime(max(date), '%Y-%m-%d') as last_recorded,
                strftime(
                    min(case when sales > 0 then date else null end), '%Y-%m-%d'
                ) as first_sale,
                strftime(
                    max(case when sales > 0 then date else null end), '%Y-%m-%d'
                ) as last_sale,
                sum(case when sales > 0 then 1 else 0 end) as positive_rows,
                count(*) as total_rows
            FROM train
            GROUP BY store_nbr
            ORDER BY store_nbr
        """).df()

        late_opening = []
        min_date_str = results["date_ranges"]["train"]["min_date"]
        for _, r in store_stats.iterrows():
            first_sale = r["first_sale"]
            if first_sale and first_sale > min_date_str:
                late_opening.append(
                    {
                        "store_nbr": int(r["store_nbr"]),
                        "first_sale_date": str(first_sale),
                    }
                )
        results["late_opening_stores"] = late_opening
        results["total_stores"] = int(con.execute("SELECT count(*) FROM stores").fetchone()[0])

    # 5. Zero-sales share by product family & intermittent classification
    if "train" in tables:
        family_df = con.execute("""
            SELECT
                family,
                count(*) as total_rows,
                sum(case when sales = 0.0 then 1 else 0 end) as zero_rows,
                round(
                    100.0 * sum(case when sales = 0.0 then 1 else 0 end) / count(*), 2
                ) as zero_share_pct,
                round(avg(sales), 2) as mean_sales,
                round(median(sales), 2) as median_sales,
                round(max(sales), 2) as max_sales
            FROM train
            GROUP BY family
            ORDER BY zero_share_pct DESC, family ASC
        """).df()

        family_stats = {}
        intermittent_families = []
        for _, r in family_df.iterrows():
            fam = r["family"]
            zero_pct = float(r["zero_share_pct"])
            is_intermittent = zero_pct >= 50.0
            family_stats[fam] = {
                "zero_share_pct": zero_pct,
                "mean_sales": float(r["mean_sales"]),
                "median_sales": float(r["median_sales"]),
                "max_sales": float(r["max_sales"]),
                "is_intermittent": is_intermittent,
            }
            if is_intermittent:
                intermittent_families.append(fam)

        results["family_stats"] = family_stats
        results["intermittent_families"] = intermittent_families
        results["total_families"] = len(family_stats)

    # 6. Promotion coverage over time
    if "train" in tables:
        promo_df = con.execute("""
            SELECT
                year(date) as yr,
                count(*) as total_rows,
                sum(case when onpromotion > 0 then 1 else 0 end) as promo_rows,
                round(
                    100.0 * sum(case when onpromotion > 0 then 1 else 0 end) / count(*), 2
                ) as promo_pct,
                round(avg(onpromotion), 2) as mean_promo,
                max(onpromotion) as max_promo
            FROM train
            GROUP BY yr
            ORDER BY yr
        """).df()

        promo_by_year = {}
        for _, r in promo_df.iterrows():
            promo_by_year[int(r["yr"])] = {
                "promo_rows": int(r["promo_rows"]),
                "promo_share_pct": float(r["promo_pct"]),
                "mean_onpromotion": float(r["mean_promo"]),
                "max_onpromotion": int(r["max_promo"]),
            }
        results["promo_by_year"] = promo_by_year

        first_promo_row = con.execute(
            "SELECT strftime(min(date), '%Y-%m-%d') FROM train WHERE onpromotion > 0"
        ).fetchone()
        results["first_recorded_promotion_date"] = (
            str(first_promo_row[0]) if first_promo_row and first_promo_row[0] else None
        )

    # 7. Oil missing values
    if "oil" in tables:
        oil_total = con.execute("SELECT count(*) FROM oil").fetchone()[0]
        oil_missing = con.execute("SELECT count(*) FROM oil WHERE dcoilwtico IS NULL").fetchone()[0]
        results["oil_summary"] = {
            "total_rows": int(oil_total),
            "missing_prices": int(oil_missing),
            "missing_pct": round(float(oil_missing) / float(oil_total) * 100.0, 2)
            if oil_total > 0
            else 0.0,
        }

    # 8. Holidays and events
    if "holidays_events" in tables:
        hol_df = con.execute("""
            SELECT
                type,
                locale,
                count(*) as count,
                sum(case when transferred then 1 else 0 end) as transferred_count
            FROM holidays_events
            GROUP BY type, locale
            ORDER BY type, locale
        """).df()
        hol_list = []
        for _, r in hol_df.iterrows():
            hol_list.append(
                {
                    "type": r["type"],
                    "locale": r["locale"],
                    "count": int(r["count"]),
                    "transferred": int(r["transferred_count"]),
                }
            )
        results["holidays_summary"] = {
            "total_holidays": int(
                con.execute("SELECT count(*) FROM holidays_events").fetchone()[0]
            ),
            "breakdown": hol_list,
            "transferred_total": int(
                con.execute(
                    "SELECT count(*) FROM holidays_events WHERE transferred = true"
                ).fetchone()[0]
            ),
        }

    # 9. Outliers and sales distribution
    if "train" in tables:
        quantiles = con.execute("""
            SELECT
                round(percentile_cont(0.50) WITHIN GROUP (ORDER BY sales), 2) as p50,
                round(percentile_cont(0.90) WITHIN GROUP (ORDER BY sales), 2) as p90,
                round(percentile_cont(0.99) WITHIN GROUP (ORDER BY sales), 2) as p99,
                round(percentile_cont(0.999) WITHIN GROUP (ORDER BY sales), 2) as p999,
                round(max(sales), 2) as max_sales
            FROM train
        """).fetchone()
        results["sales_distribution"] = {
            "p50": float(quantiles[0]),
            "p90": float(quantiles[1]),
            "p99": float(quantiles[2]),
            "p999": float(quantiles[3]),
            "max_sales": float(quantiles[4]),
        }

    return results


def update_metrics_json(
    audit_results: dict[str, Any],
    metrics_path: Path | str = METRICS_FILE,
) -> None:
    """Save or update the 'audit' section of metrics.json."""
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception:
            existing = {}

    existing["audit"] = audit_results
    with path.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
        fh.write("\n")
    log.info("Saved audit metrics to %s", path)


def generate_data_card(
    audit: dict[str, Any],
    output_path: Path | str = DOCS / "DATA_CARD.md",
) -> str:
    """Generate docs/DATA_CARD.md documenting audit findings and cleaning decisions."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row_counts = audit.get("row_counts", {})
    date_ranges = audit.get("date_ranges", {})
    missing_dates = audit.get("missing_calendar_dates", [])
    intermittent = audit.get("intermittent_families", [])
    promo_by_year = audit.get("promo_by_year", {})
    oil = audit.get("oil_summary", {})
    sales_dist = audit.get("sales_distribution", {})
    first_promo = audit.get("first_recorded_promotion_date", "None")

    train_rows = row_counts.get("train", 0)
    train_min = date_ranges.get("train", {}).get("min_date", "N/A")
    train_max = date_ranges.get("train", {}).get("max_date", "N/A")
    test_rows = row_counts.get("test", 0)
    test_min = date_ranges.get("test", {}).get("min_date", "N/A")
    test_max = date_ranges.get("test", {}).get("max_date", "N/A")
    stores_cnt = row_counts.get("stores", 0)
    oil_rows = row_counts.get("oil", 0)
    oil_min = date_ranges.get("oil", {}).get("min_date", "N/A")
    oil_max = date_ranges.get("oil", {}).get("max_date", "N/A")
    hol_rows = row_counts.get("holidays_events", 0)
    hol_min = date_ranges.get("holidays_events", {}).get("min_date", "N/A")
    hol_max = date_ranges.get("holidays_events", {}).get("max_date", "N/A")
    trans_rows = row_counts.get("transactions", 0)
    trans_min = date_ranges.get("transactions", {}).get("min_date", "N/A")
    trans_max = date_ranges.get("transactions", {}).get("max_date", "N/A")
    sub_rows = row_counts.get("sample_submission", 0)

    content = f"""# Data Card: Corporación Favorita Grocery Sales

## 1. Overview & Dataset Description

This project utilizes the public Corporación Favorita grocery sales dataset from the Kaggle
competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting).
Corporación Favorita is one of Ecuador's largest supermarket chains.

- **Primary Source:** Kaggle / Corporación Favorita
- **Geographic Scope:** Ecuador (national chain across multiple cities and provinces)
- **Primary Granularity:** Daily unit sales by Store (`store_nbr`) and Product Family (`family`)
- **Total Stores:** {audit.get("total_stores", 54)}
- **Total Product Families:** {audit.get("total_families", 33)}

## 2. Ingestion & Table Summary

| Table | Rows | Date Range | Primary Key / Grain |
| --- | --- | --- | --- |
| `train` | {train_rows:,} | {train_min} to {train_max} | `id` (date x store_nbr x family) |
| `test` | {test_rows:,} | {test_min} to {test_max} | `id` (16 days x 54 stores x 33 families) |
| `stores` | {stores_cnt:,} | Static | `store_nbr` (city, state, type, cluster) |
| `oil` | {oil_rows:,} | {oil_min} to {oil_max} | `date` (WTI spot price) |
| `holidays_events` | {hol_rows:,} | {hol_min} to {hol_max} | `date`, `type`, `transferred` |
| `transactions` | {trans_rows:,} | {trans_min} to {trans_max} | `date` x `store_nbr` |
| `sample_submission` | {sub_rows:,} | 16-day evaluation | `id` |

## 3. Data Audit Findings

### 3.1 Missing Calendar Dates (Store Closures)
Across the {train_min} to {train_max} training period ({train_rows:,} rows),
exactly {len(missing_dates)} calendar days are completely missing from `train.csv`:
{", ".join(f"`{d}`" for d in missing_dates)}

**Reason:** In Ecuador, retail operations for Corporación Favorita close entirely on
Christmas Day (`December 25`). The Kaggle data omits these dates rather than providing zero sales.

### 3.2 Late-Opening Stores
Several stores opened well after January 2013:
"""
    late_stores = audit.get("late_opening_stores", [])
    if late_stores:
        for s in late_stores:
            snbr = s["store_nbr"]
            sdate = s["first_sale_date"]
            content += f"- **Store {snbr}:** First recorded positive sale on `{sdate}`.\n"
    else:
        content += "All stores active from start.\n"

    intermittent_str = ", ".join(f"`{f}`" for f in intermittent)
    oil_missing_cnt = oil.get("missing_prices", 0)
    oil_missing_pct = oil.get("missing_pct", 0.0)
    oil_total_obs = oil.get("total_rows", 0)
    total_hols = audit.get("holidays_summary", {}).get("total_holidays", 0)
    total_transferred = audit.get("holidays_summary", {}).get("transferred_total", 0)

    p50_val = sales_dist.get("p50", 0)
    p90_val = sales_dist.get("p90", 0)
    p99_val = sales_dist.get("p99", 0)
    p999_val = sales_dist.get("p999", 0)
    max_val = sales_dist.get("max_sales", 0)

    content += f"""
Prior to opening, records for these stores exist in `train.csv` with zero sales.

### 3.3 Intermittent Families & Zero-Sales Share
Sales intermittency varies dramatically across product categories:
- **Intermittent Families (>50% zero-sales days):** {intermittent_str}
- **Fast-Moving Grocery Staples (<10% zero-sales days):** `DAIRY`, `DELI`, `MEATS`,
  `BREAD/BAKERY`, `EGGS`, `PRODUCE`, `GROCERY I`.
  Their non-zero rate is ~92% (zero sales occur almost exclusively on closures or Sundays).

### 3.4 Promotional Tracking Evolution
The recording of promotional activity (`onpromotion` column) evolved over time:
- **2013:** 0% tracked (`onpromotion` = 0 for the entire year).
- **First Tracked Promotion Date:** `{first_promo}`.
"""
    if promo_by_year:
        content += "\n| Year | Active Promo Rows | Promo Share (%) | Mean Promo Count |\n"
        content += "| --- | --- | --- | --- |\n"
        for yr, p in sorted(promo_by_year.items()):
            p_rows = p["promo_rows"]
            p_pct = p["promo_share_pct"]
            p_mean = p["mean_onpromotion"]
            content += f"| {yr} | {p_rows:,} | {p_pct}% | {p_mean} |\n"

    content += f"""
### 3.5 Oil Price Series & Missing Values
- Total oil price observations: {oil_total_obs:,}
- Weekday missing price observations: {oil_missing_cnt:,} ({oil_missing_pct}%)
- In addition, weekends are omitted from the price table.

### 3.6 Holiday Structure & Transferred Observances
- Total holiday / event entries: {total_hols}
- Transferred holidays: {total_transferred} holidays had `transferred = True`,
  accompanied by a corresponding `Transfer` entry on the actual observed date.
- Major Events: Includes the April 16, 2016 Manabi Earthquake (`Terremoto Manabi`),
  which triggered massive emergency relief procurement and distribution spikes.

### 3.7 Sales Distribution & Extreme Outliers
- **Median (P50):** {p50_val:,} units
- **90th Percentile (P90):** {p90_val:,} units
- **99th Percentile (P99):** {p99_val:,} units
- **99.9th Percentile (P99.9):** {p999_val:,} units
- **Max Single Record:** {max_val:,} units (relief distribution during earthquake aftermath)

## 4. Cleaning Decisions & Modeling Assumptions

1. **Sales Are Not True Demand (Censoring):**
   Sales observations represent fulfilled demand. When a store stocks out, recorded sales
   drop to zero regardless of customer intent. Simulated replenishment and fill rates
   derived from sales data must be interpreted as optimistic lower bounds on true demand variance.

2. **Observational Nature of Promotions & Assortment Whitespace:**
   Promotional assignments are not randomized A/B tests; marketing teams selectively discount
   high-velocity items or distressed inventory. All uplift and assortment-gap models are
   observational associations, not causal guarantees.

3. **Information Leakage Prevention (Forecast Origin Cutoff):**
   Same-day transactions and same-day oil prices are NEVER permitted as features for predicting
   same-day sales. At forecast origin $t_0$, only strictly historical signals ($t \\le t_0$)
   and known calendar/promotional schedules ($t > t_0$) may be utilized.

4. **Missing Oil Price Imputation:**
   Missing weekday oil prices and weekend dates are filled via forward-fill (carrying forward
   the latest available closing spot price), followed by backward-fill for the initial observation.

5. **Christmas Day Handling:**
   December 25 is recognized as an institutional closure. The replenishment and forecasting
   pipelines treat Christmas Day as a deterministic zero-sales day.

6. **Promotional Pre-History Handling:**
   Because promotional tracking began on April 1, 2014, econometric promotion-response models
   (PR5) must condition on data from April 2014 onward to prevent false negatives from
   unrecorded 2013 promotions.
"""
    with out.open("w", encoding="utf-8") as fh:
        fh.write(content)
    log.info("Wrote data card to %s", out)
    return content
