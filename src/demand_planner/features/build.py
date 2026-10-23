"""Feature engineering for grocery store-family demand forecasting.

Strictly causal feature pipeline ensuring ZERO temporal leakage over a 16-day forecast horizon:
- Lagged sales: lag 16, lag 21, lag 28 (all known at forecast origin t_0 for all h in 1..16)
- Rolling aggregations over lag 16: 7-day and 28-day window means
- Calendar features: day of week, day of month, month, weekend flag, bi-weekly payday flags
- Holiday events: non-transferred national and regional holidays
- Promotions: promotional item counts (post April 2014)
- Store metadata: store type, store cluster, store number, and product family
"""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "store_nbr",
    "family",
    "store_type",
    "cluster",
    "dayofweek",
    "day",
    "month",
    "is_weekend",
    "is_payday",
    "is_holiday",
    "onpromotion",
    "has_promotion",
    "lag_16",
    "lag_21",
    "lag_28",
    "rolling_mean_7_lag16",
    "rolling_mean_28_lag16",
]

CATEGORICAL_COLUMNS = ["store_nbr", "family", "store_type", "cluster"]


def build_features(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame | None = None,
    holidays_df: pd.DataFrame | None = None,
    date_col: str = "date",
    target_col: str = "sales",
) -> tuple[pd.DataFrame, list[str]]:
    """Build causal tabular features for tree-based forecasting models.

    Parameters
    ----------
    sales_df : pd.DataFrame
        DataFrame with date, store_nbr, family, sales, onpromotion.
    stores_df : pd.DataFrame | None
        Store metadata with store_nbr, type, cluster.
    holidays_df : pd.DataFrame | None
        Holidays table with date, transferred.
    date_col : str
        Name of date column.
    target_col : str
        Target column to lag.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (Transformed DataFrame, list of feature column names)
    """
    df = sales_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(["store_nbr", "family", date_col]).reset_index(drop=True)

    # 1. Strictly causal lagged sales features (min lag = 16 days for 16-day horizon)
    grouped = df.groupby(["store_nbr", "family"])[target_col]
    df["lag_16"] = grouped.shift(16)
    df["lag_21"] = grouped.shift(21)
    df["lag_28"] = grouped.shift(28)

    # Rolling statistics over lag 16
    df["rolling_mean_7_lag16"] = df.groupby(["store_nbr", "family"])["lag_16"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["rolling_mean_28_lag16"] = df.groupby(["store_nbr", "family"])["lag_16"].transform(
        lambda s: s.rolling(28, min_periods=1).mean()
    )

    # Fill initial NaNs for series without sufficient pre-history
    for col in [
        "lag_16",
        "lag_21",
        "lag_28",
        "rolling_mean_7_lag16",
        "rolling_mean_28_lag16",
    ]:
        df[col] = df[col].fillna(0.0).clip(lower=0.0)

    # 2. Calendar features
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["day"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # Ecuadorian paydays: 15th and end of month
    df["is_payday"] = ((df["day"] == 15) | (df[date_col].dt.is_month_end)).astype(int)

    # 3. Holiday features
    if holidays_df is not None and len(holidays_df) > 0:
        hols = holidays_df.copy()
        if "transferred" in hols.columns:
            hols = hols[~hols["transferred"]]
        active_dates = set(pd.to_datetime(hols["date"]).dt.strftime("%Y-%m-%d"))
        df["is_holiday"] = df[date_col].dt.strftime("%Y-%m-%d").isin(active_dates).astype(int)
    else:
        df["is_holiday"] = 0

    # 4. Promotions
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].fillna(0).astype(int)
    else:
        df["onpromotion"] = 0
    df["has_promotion"] = (df["onpromotion"] > 0).astype(int)

    # 5. Store metadata
    if stores_df is not None and len(stores_df) > 0:
        st = stores_df.rename(columns={"type": "store_type"})[
            ["store_nbr", "store_type", "cluster"]
        ]
        df = df.merge(st, on="store_nbr", how="left")
    else:
        df["store_type"] = "Unknown"
        df["cluster"] = 0

    df["store_type"] = df["store_type"].fillna("Unknown")
    df["cluster"] = df["cluster"].fillna(0)

    # Convert categoricals
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    return df, [col for col in FEATURE_COLUMNS if col in df.columns]
