"""Generator for a tiny Favorita-shaped synthetic dataset for fast CI tests.

Matches the exact schema, types, and quirks of the Corporación Favorita Kaggle data:
- 5 stores with city, state, type, and cluster metadata
- 6 product families
- ~400 training days + 16 test days
- Promotions with realistic demand variations
- A late-opening store (store 5 opens at day 180)
- Missing December 25 (stores closed for Christmas, no rows in train or transactions)
- Holidays table including a transferred holiday (transferred=True) and its transfer date
  (type='Transfer')
- Oil table with price trends and missing weekends/holidays
- Transactions table with daily counts per active store
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = (
    "GROCERY I",
    "BEVERAGES",
    "CLEANING",
    "DAIRY",
    "PRODUCE",
    "MEATS",
)

STORE_METADATA = [
    {"store_nbr": 1, "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 13},
    {"store_nbr": 2, "city": "Guayaquil", "state": "Guayas", "type": "B", "cluster": 6},
    {"store_nbr": 3, "city": "Cuenca", "state": "Azuay", "type": "D", "cluster": 8},
    {"store_nbr": 4, "city": "Quito", "state": "Pichincha", "type": "C", "cluster": 9},
    {"store_nbr": 5, "city": "Santo Domingo", "state": "Santo Domingo", "type": "A", "cluster": 4},
]


def generate_tiny_favorita(
    output_dir: Path | str,
    n_days: int = 400,
    start_date: str = "2016-01-01",
    seed: int = 42,
) -> dict[str, Path]:
    """Generate all 7 Favorita CSV files into ``output_dir`` and return their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. Stores
    stores_df = pd.DataFrame(STORE_METADATA)
    stores_path = out / "stores.csv"
    stores_df.to_csv(stores_path, index=False)

    # 2. Date ranges
    all_dates = pd.date_range(start_date, periods=n_days, freq="D")
    test_dates = pd.date_range(all_dates[-1] + pd.Timedelta(days=1), periods=16, freq="D")

    # Christmas is excluded in Favorita data (stores closed)
    train_dates = [d for d in all_dates if not (d.month == 12 and d.day == 25)]
    store_5_opening_date = all_dates[180]

    # 3. Holidays & events
    holidays_data = [
        {
            "date": "2016-01-01",
            "type": "Holiday",
            "locale": "National",
            "locale_name": "Ecuador",
            "description": "Primer dia del ano",
            "transferred": False,
        },
        {
            "date": "2016-05-24",
            "type": "Holiday",
            "locale": "National",
            "locale_name": "Ecuador",
            "description": "Batalla de Pichincha",
            "transferred": True,  # Holiday observed on another day
        },
        {
            "date": "2016-05-27",
            "type": "Transfer",
            "locale": "National",
            "locale_name": "Ecuador",
            "description": "Traslado Batalla de Pichincha",
            "transferred": False,
        },
        {
            "date": "2016-08-10",
            "type": "Holiday",
            "locale": "National",
            "locale_name": "Ecuador",
            "description": "Primer Grito de Independencia",
            "transferred": False,
        },
        {
            "date": "2016-12-06",
            "type": "Holiday",
            "locale": "Local",
            "locale_name": "Quito",
            "description": "Fundacion de Quito",
            "transferred": False,
        },
        {
            "date": "2016-12-25",
            "type": "Holiday",
            "locale": "National",
            "locale_name": "Ecuador",
            "description": "Navidad",
            "transferred": False,
        },
    ]
    holidays_df = pd.DataFrame(holidays_data)
    holidays_path = out / "holidays_events.csv"
    holidays_df.to_csv(holidays_path, index=False)

    # 4. Oil price (business days have prices, weekends/holidays have NaN)
    full_date_range = pd.date_range(start_date, test_dates[-1], freq="D")
    base_oil = 35.0 + np.cumsum(rng.normal(0.05, 0.4, len(full_date_range)))
    oil_series = []
    for d, val in zip(full_date_range, base_oil, strict=False):
        if d.weekday() in (5, 6) or rng.random() < 0.05:  # Weekend
            oil_series.append(np.nan)
        else:
            oil_series.append(round(float(val), 2))
    oil_df = pd.DataFrame(
        {
            "date": full_date_range.strftime("%Y-%m-%d"),
            "dcoilwtico": oil_series,
        }
    )
    oil_path = out / "oil.csv"
    oil_df.to_csv(oil_path, index=False)

    # 5. Train sales & Transactions
    train_rows = []
    trans_rows = []
    current_id = 0

    base_family_sales = {
        "GROCERY I": 800.0,
        "BEVERAGES": 600.0,
        "CLEANING": 350.0,
        "DAIRY": 400.0,
        "PRODUCE": 500.0,
        "MEATS": 250.0,
    }

    store_multipliers = {1: 1.2, 2: 1.0, 3: 0.9, 4: 0.8, 5: 1.4}

    for d in train_dates:
        d_str = d.strftime("%Y-%m-%d")
        day_of_week = d.weekday()
        dow_factor = 1.3 if day_of_week in (4, 5) else 1.0  # Weekend boost

        for store in stores_df["store_nbr"]:
            # Store 5 is unopened before day 180
            is_open = not (store == 5 and d < store_5_opening_date)
            store_daily_sales = 0.0

            for fam in FAMILIES:
                if not is_open:
                    sales = 0.0
                    onpromotion = 0
                else:
                    on_promo = int(rng.random() < 0.25)
                    onpromotion = int(rng.integers(1, 15)) if on_promo else 0
                    promo_lift = 1.35 if onpromotion > 0 else 1.0
                    mult = store_multipliers[store] * dow_factor * promo_lift
                    base = base_family_sales[fam] * mult
                    noise = rng.normal(1.0, 0.12)
                    sales = max(0.0, round(float(base * noise), 2))

                train_rows.append(
                    {
                        "id": current_id,
                        "date": d_str,
                        "store_nbr": store,
                        "family": fam,
                        "sales": sales,
                        "onpromotion": onpromotion,
                    }
                )
                current_id += 1
                store_daily_sales += sales

            if is_open:
                # Transactions proportional to total sales
                trans_count = int(max(10, round(store_daily_sales / 3.5 + rng.normal(0, 15))))
                trans_rows.append(
                    {
                        "date": d_str,
                        "store_nbr": store,
                        "transactions": trans_count,
                    }
                )

    train_df = pd.DataFrame(train_rows)
    train_path = out / "train.csv"
    train_df.to_csv(train_path, index=False)

    trans_df = pd.DataFrame(trans_rows)
    trans_path = out / "transactions.csv"
    trans_df.to_csv(trans_path, index=False)

    # 6. Test set (16 days)
    test_rows = []
    sub_rows = []
    for d in test_dates:
        d_str = d.strftime("%Y-%m-%d")
        for store in stores_df["store_nbr"]:
            for fam in FAMILIES:
                on_promo = int(rng.random() < 0.20)
                onpromotion = int(rng.integers(1, 10)) if on_promo else 0
                test_rows.append(
                    {
                        "id": current_id,
                        "date": d_str,
                        "store_nbr": store,
                        "family": fam,
                        "onpromotion": onpromotion,
                    }
                )
                sub_rows.append(
                    {
                        "id": current_id,
                        "sales": 0.0,
                    }
                )
                current_id += 1

    test_df = pd.DataFrame(test_rows)
    test_path = out / "test.csv"
    test_df.to_csv(test_path, index=False)

    sub_df = pd.DataFrame(sub_rows)
    sub_path = out / "sample_submission.csv"
    sub_df.to_csv(sub_path, index=False)

    return {
        "stores": stores_path,
        "holidays": holidays_path,
        "oil": oil_path,
        "transactions": trans_path,
        "train": train_path,
        "test": test_path,
        "sample_submission": sub_path,
    }
