"""Grocery Demand Planner - Interactive Portfolio Streamlit Application.

Interactive multi-page dashboard for inventory replenishment, forecast exploration,
and operational supply-chain insights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Corporación Favorita Demand Planner",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

METRICS_PATH = Path("metrics/metrics.json")
OUTPUTS_DIR = Path("outputs")
SQL_DIR = Path("sql")
DOCS_IMG_DIR = Path("docs/img")


@st.cache_data
def load_metrics() -> dict[str, Any]:
    """Load precomputed metrics and evaluation results."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Load compact forecast summary for interactive inspection."""
    summary_path = OUTPUTS_DIR / "forecast_sample_summary.parquet"
    if summary_path.exists():
        df = pd.read_parquet(summary_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    pred_path = OUTPUTS_DIR / "oof_and_holdout_predictions.parquet"
    if pred_path.exists():
        df = pd.read_parquet(pred_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_sim_summary() -> pd.DataFrame:
    """Load replenishment simulation policy summary."""
    sim_path = OUTPUTS_DIR / "sim_policy_summary.parquet"
    if sim_path.exists():
        return pd.read_parquet(sim_path)
    return pd.DataFrame()


@st.cache_data
def load_sql_summary(filename: str) -> pd.DataFrame:
    """Load precomputed SQL analytical summary parquet (< 3 MB tracked)."""
    p = OUTPUTS_DIR / filename
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def render_summary_overview(metrics: dict[str, Any]):
    """Page 1: Portfolio summary of backtest, holdout, replenishment, and promo lift."""
    st.title("📦 Corporación Favorita Demand Planner & Inventory Replenishment")
    st.markdown(
        "**Target Persona:** Retail Product Manager & Senior Product Analyst  \n"
        "**Core Business Impact:** Quantifying working capital reduction from ML forecasting "
        "at matched on-shelf customer service levels."
    )

    st.markdown("---")

    # Headline KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="5-Fold Backtest WAPE",
            value="13.49%",
            delta="-2.84% vs Naive (16.33%)",
            delta_color="normal",
        )
    with col2:
        st.metric(
            label="Holdout (Aug 2017) WAPE",
            value="15.29%",
            delta="-5.38% vs Naive (20.67%)",
            delta_color="normal",
        )
    with col3:
        st.metric(
            label="Chain Inventory Savings",
            value="-12.3% to -14.5%",
            delta="Matched 99.0% Fill Rate",
            delta_color="normal",
        )
    with col4:
        st.metric(
            label="Promotional Sales Lift",
            value="+171.2%",
            delta="Store-Clustered 95% CI",
            delta_color="off",
        )

    st.markdown("---")

    # Replenishment Policy Benchmark Section
    st.subheader("1. Replenishment Policy Benchmark (Rule 2 Frontier)")
    st.markdown(
        "Under a periodic $(R, S)$ inventory policy with out-of-sample error safety stock, "
        "**LightGBM achieves a -12.27% to -14.51% average inventory reduction** compared to "
        "Seasonal Naive at a matched 99.0% fill rate. Both Series and Time-Block 95% bootstrap CIs "
        "strictly exclude 0 across both operational lead times ($L=3$ and $L=7$)."
    )

    frontier_data = [
        {
            "Lead Time": "L = 3 days",
            "Target Fill Rate": "98.5%",
            "LightGBM Point vs Naive": "-13.22%",
            "95% Series CI": "[-15.26%, -11.28%]",
            "95% Time-Block CI": "[-20.48%, -3.07%]",
            "Status": "Interior / Win",
        },
        {
            "Lead Time": "L = 3 days",
            "Target Fill Rate": "99.0%",
            "LightGBM Point vs Naive": "-12.27%",
            "95% Series CI": "[-14.59%, -10.05%]",
            "95% Time-Block CI": "[-19.97%, -2.25%]",
            "Status": "Primary Win",
        },
        {
            "Lead Time": "L = 3 days",
            "Target Fill Rate": "99.5%",
            "LightGBM Point vs Naive": "-11.83%",
            "95% Series CI": "[-14.49%, -9.33%]",
            "95% Time-Block CI": "[-21.24%, -1.41%]",
            "Status": "Interior / Win",
        },
        {
            "Lead Time": "L = 7 days",
            "Target Fill Rate": "98.5%",
            "LightGBM Point vs Naive": "-15.70%",
            "95% Series CI": "[-18.66%, -12.75%]",
            "95% Time-Block CI": "[-23.23%, -8.45%]",
            "Status": "Interior / Win",
        },
        {
            "Lead Time": "L = 7 days",
            "Target Fill Rate": "99.0%",
            "LightGBM Point vs Naive": "-14.51%",
            "95% Series CI": "[-17.53%, -11.37%]",
            "95% Time-Block CI": "[-21.22%, -7.31%]",
            "Status": "Primary Win",
        },
        {
            "Lead Time": "L = 7 days",
            "Target Fill Rate": "99.5%",
            "LightGBM Point vs Naive": "-13.84%",
            "95% Series CI": "[-16.89%, -10.74%]",
            "95% Time-Block CI": "[-20.73%, -6.83%]",
            "Status": "Interior / Win",
        },
    ]
    st.table(pd.DataFrame(frontier_data))

    # Primary Family Findings
    st.subheader("2. Pre-Specified Primary Family Comparisons (Holm-Bonferroni FWER α = 0.05)")
    st.markdown(
        "To safeguard against multiple comparisons, one primary comparison was pre-specified "
        "per family: **Target Fill Rate = 99.0%, LightGBM Point vs Seasonal Naive, evaluated at "
        "both L=3 and L=7**. Statistical significance requires both 95% Series and Time-Block CIs "
        "to exclude 0 under Holm-Bonferroni step-down adjustment across all 13 families."
    )

    family_primary = [
        {
            "Family": "LAWN AND GARDEN",
            "L=3 Diff": "-32.25%",
            "L=3 Status": "Win (p_adj < 0.001)",
            "L=7 Diff": "-38.02%",
            "L=7 Status": "Win (p_adj < 0.001)",
        },
        {
            "Family": "GROCERY I",
            "L=3 Diff": "-18.51%",
            "L=3 Status": "Exploratory (Holm-adj p >= 0.05)",
            "L=7 Diff": "-19.57%",
            "L=7 Status": "Win (p_adj < 0.001)",
        },
        {
            "Family": "DELI",
            "L=3 Diff": "-17.63%",
            "L=3 Status": "Exploratory (Holm-adj p >= 0.05)",
            "L=7 Diff": "-22.73%",
            "L=7 Status": "Win (p_adj < 0.001)",
        },
        {
            "Family": "POULTRY",
            "L=3 Diff": "-9.91%",
            "L=3 Status": "Exploratory (Holm-adj p >= 0.05)",
            "L=7 Diff": "-14.75%",
            "L=7 Status": "Win (p_adj < 0.001)",
        },
        {
            "Family": "BREAD/BAKERY",
            "L=3 Diff": "-7.52%",
            "L=3 Status": "Exploratory (Time CI crosses 0)",
            "L=7 Diff": "-10.02%",
            "L=7 Status": "Win (p_adj < 0.001)",
        },
        {
            "Family": "MEATS",
            "L=3 Diff": "-9.23%",
            "L=3 Status": "Exploratory (Holm-adj p >= 0.05)",
            "L=7 Diff": "-9.71%",
            "L=7 Status": "Win (p_adj = 0.032)",
        },
        {
            "Family": "DAIRY",
            "L=3 Diff": "N/A",
            "L=3 Status": "Target below frontier min (>99% at z=-1.5)",
            "L=7 Diff": "N/A",
            "L=7 Status": "Target below frontier min (>99% at z=-1.5)",
        },
        {
            "Family": "BABY CARE",
            "L=3 Diff": "N/A",
            "L=3 Status": "Target above frontier max (max 92.7%)",
            "L=7 Diff": "N/A",
            "L=7 Status": "Target above frontier max (max 91.8%)",
        },
        {
            "Family": "BOOKS",
            "L=3 Diff": "-13.23%",
            "L=3 Status": "Within noise (CI crosses 0)",
            "L=7 Diff": "+18.44%",
            "L=7 Status": "Within noise (CI crosses 0)",
        },
    ]
    st.dataframe(pd.DataFrame(family_primary), use_container_width=True)

    # Inline Visualizations
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        tradeoff_img = DOCS_IMG_DIR / "replenishment_policy_tradeoff.png"
        if tradeoff_img.exists():
            st.image(str(tradeoff_img), caption="Figure 1: Fill Rate vs Average Inventory Frontier")
    with col_img2:
        promo_img = DOCS_IMG_DIR / "promo_lift_by_family.png"
        if promo_img.exists():
            st.image(
                str(promo_img),
                caption="Figure 2: Promotional Lift Associations by Product Family",
            )

    # Critical Methodological Assumptions & Limitations
    st.info(
        "**Operational Assumptions & Caveats:**\n\n"
        "1. **Out-of-Sample Safety Stock Construction:** Safety stock buffer $\\sigma$ is strictly "
        "computed from out-of-sample forecast errors of forecasts generated at earlier origins "
        "whose actuals were fully completed before review date $t$.\n"
        "2. **Demand Censoring Caveat:** Observed sales represent fulfilled transactions, "
        "not unconstrained customer demand. When stockouts occur, lost sales are unrecorded, "
        "causing naive models to underestimate true demand during replenishment recovery."
    )


def render_forecast_explorer(df_preds: pd.DataFrame):
    """Page 2: Interactive inspection of actual sales vs forecaster predictions."""
    st.title("🔍 Interactive Forecast Explorer")
    st.markdown("Inspect store-family time series, actual sales, and model predictions.")

    if df_preds.empty:
        st.warning(
            "Prediction outputs not found. Run pipeline stage `models` to generate predictions."
        )
        return

    stores = sorted(df_preds["store_nbr"].unique().tolist())
    families = sorted(df_preds["family"].unique().tolist())

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        sel_store = st.selectbox("Select Store", stores, index=0)
    with col_ctrl2:
        default_fam_idx = families.index("GROCERY I") if "GROCERY I" in families else 0
        sel_family = st.selectbox("Select Product Family", families, index=default_fam_idx)
    with col_ctrl3:
        sel_split = st.selectbox(
            "Evaluation Split",
            ["All", "Validation Folds (OOF)", "Test Holdout (Aug 2017)"],
        )

    # Filter data
    mask = (df_preds["store_nbr"] == sel_store) & (df_preds["family"] == sel_family)
    if sel_split == "Validation Folds (OOF)":
        mask &= df_preds["split"].str.startswith("fold_")
    elif sel_split == "Test Holdout (Aug 2017)":
        mask &= df_preds["split"] == "holdout"

    sub_df = df_preds[mask].sort_values("date").copy()

    if sub_df.empty:
        st.info("No records found for the selected combination and filter.")
        return

    # Metrics for selected series
    actuals = sub_df["sales"].to_numpy()
    pred_lgb = sub_df["pred_lgbm"].to_numpy()
    pred_sn = sub_df["pred_snaive"].to_numpy()
    sum_actuals = float(np.sum(actuals))
    wape_lgb = float(np.sum(np.abs(actuals - pred_lgb)) / max(sum_actuals, 1e-6)) * 100.0
    wape_sn = float(np.sum(np.abs(actuals - pred_sn)) / max(sum_actuals, 1e-6)) * 100.0
    zero_days = int((actuals == 0).sum())
    total_days = len(actuals)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Observed Sales", f"{sum_actuals:,.1f} units")
    with m2:
        st.metric("LightGBM WAPE", f"{wape_lgb:.2f}%")
    with m3:
        st.metric("Seasonal Naive WAPE", f"{wape_sn:.2f}%")
    with m4:
        zero_pct = 100.0 * zero_days / max(total_days, 1)
        st.metric("Zero Sales Days", f"{zero_days} / {total_days} ({zero_pct:.1f}%)")

    # Plot time series
    chart_df = sub_df[["date", "sales", "pred_lgbm", "pred_snaive"]].set_index("date")
    chart_df.columns = ["Actual Sales", "LightGBM Point", "Seasonal Naive"]
    st.line_chart(chart_df, use_container_width=True)

    with st.expander("View Raw Prediction Table"):
        st.dataframe(sub_df, use_container_width=True)


def render_replenishment_simulator(sim_df: pd.DataFrame):
    """Page 3: Interactive (R, S) periodic replenishment simulator."""
    st.title("⚙️ Replenishment Policy Frontier Simulator")
    st.markdown(
        "Explore how replenishment lead time ($L$), safety stock multiplier ($z$), "
        "and forecaster choice alter achievable fill rates and required inventory holding."
    )

    if sim_df.empty:
        st.warning("Simulation summary data not found. Run pipeline stage `simulation`.")
        return

    # Filter to Rule 2 standard frontier
    r2_df = sim_df[sim_df["stitching_rule"] == "rule_2"].copy()
    if r2_df.empty:
        r2_df = sim_df.copy()

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        lead_time = st.radio(
            "Lead Time (L)",
            [3, 7],
            horizontal=True,
            format_func=lambda x: f"L = {x} days",
        )
    with col_sim2:
        target_fr = st.slider("Target Service Level (Fill Rate %)", 98.0, 99.8, 99.0, step=0.1)

    filtered_sim = r2_df[r2_df["lead_time_days"] == lead_time].sort_values("fill_rate_pct")

    # Display frontier curve
    chart_data = filtered_sim.pivot_table(
        index="fill_rate_pct",
        columns="policy_label",
        values="average_inventory_units",
    )
    st.line_chart(chart_data, use_container_width=True)

    # Benchmark stats at target
    st.subheader(f"Frontier Points at L = {lead_time} days (Target Selected: {target_fr:.1f}%)")
    cols_display = [
        "policy_label",
        "z_value",
        "fill_rate_pct",
        "average_inventory_units",
        "stockout_day_share_pct",
        "inventory_turns",
    ]
    st.dataframe(
        filtered_sim[cols_display].rename(
            columns={
                "policy_label": "Policy",
                "z_value": "Safety Factor (z)",
                "fill_rate_pct": "Fill Rate (%)",
                "average_inventory_units": "Average Inventory",
                "stockout_day_share_pct": "Stockout Days (%)",
                "inventory_turns": "Annual Turns",
            }
        ),
        use_container_width=True,
    )

    st.markdown(
        f"**Key Operational Takeaway:** For lead time $L={lead_time}$, reaching high service "
        "levels with Seasonal Naive demands substantial safety stock buffers due to high "
        "forecast uncertainty. LightGBM compresses residual variance, allowing the store "
        "to hit target availability while freeing working capital."
    )


def render_business_insights():
    """Page 4: Analytical SQL views, operational decisions, and CSV downloads."""
    st.title("📊 Business Insights & Analytical SQL Queries")
    st.markdown(
        "Explore analytical SQL queries designed for supply chain managers and category buyers."
    )

    queries = {
        "Promotional Elasticity & Lift": {
            "file": "promo_elasticity.sql",
            "summary_file": "promo_elasticity_summary.parquet",
            "summary": (
                "Measures empirical promotional lift by store-family post 2014-04-01 "
                "to optimize promo stock."
            ),
        },
        "Zero-Sales-Day Risk Proxy": {
            "file": "zero_sales_day_proxy.sql",
            "summary_file": "zero_sales_day_proxy_summary.parquet",
            "summary": (
                "Identifies fast-moving store departments with elevated zero-sales days "
                "as an empirical proxy for inventory availability risk."
            ),
        },
        "Weekly S&OP Multi-Source View": {
            "file": "weekly_store_family.sql",
            "summary_file": "weekly_store_family_summary.parquet",
            "summary": "Unifies weekly sales, store foot traffic, oil prices, and civic holidays.",
        },
        "Perishable Demand Volatility Proxy": {
            "file": "perishable_volatility_proxy.sql",
            "summary_file": "perishable_volatility_summary.parquet",
            "summary": (
                "Measures demand coefficient of variation (CV) in fresh departments as a "
                "proxy for perishable volatility exposure."
            ),
        },
    }

    choice = st.selectbox("Select SQL Analytical View", list(queries.keys()))
    q_info = queries[choice]
    q_file = SQL_DIR / q_info["file"]

    if not q_file.exists():
        st.error(f"SQL file {q_file} not found.")
        return

    sql_text = q_file.read_text(encoding="utf-8")

    st.subheader(choice)
    st.markdown(f"**Operational Decision:** {q_info['summary']}")

    with st.expander("View SQL Definition", expanded=False):
        st.code(sql_text, language="sql")

    # Display results from precomputed tracked aggregate (< 3 MB total)
    res_df = load_sql_summary(q_info["summary_file"])
    if not res_df.empty:
        st.markdown(f"**Returned Records:** `{len(res_df):,}` rows")
        st.dataframe(res_df.head(200), use_container_width=True)

        csv_data = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {choice} (CSV)",
            data=csv_data,
            file_name=q_info["file"].replace(".sql", ".csv"),
            mime="text/csv",
        )
    else:
        st.info(
            f"Precomputed aggregate `{q_info['summary_file']}` not found. "
            "Run pipeline stage `reports` to generate tracked summaries."
        )


def main():
    """Main application navigation."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Summary Overview",
            "Forecast Explorer",
            "Replenishment Simulator",
            "Business Insights & SQL",
        ],
    )

    metrics = load_metrics()
    sim_df = load_sim_summary()
    preds_df = load_predictions()

    if page == "Summary Overview":
        render_summary_overview(metrics)
    elif page == "Forecast Explorer":
        render_forecast_explorer(preds_df)
    elif page == "Replenishment Simulator":
        render_replenishment_simulator(sim_df)
    elif page == "Business Insights & SQL":
        render_business_insights()


if __name__ == "__main__":
    main()
