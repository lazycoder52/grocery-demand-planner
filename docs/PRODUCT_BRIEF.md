# Product Brief: ML Demand Forecasting & Replenishment System

**Product:** Grocery Demand Planner
**Role:** AI/ML Product Management & Product Analytics
**Target Organization:** Corporación Favorita (Ecuadorian Grocery Chain)
**Status:** MVP v1.0 Baseline & Selected Model Verified

---

## 1. Problem Statement & Opportunity

Grocery inventory management requires balancing two costly failure modes: over-ordering perishable items (spoilage, holding costs, and margin-eroding markdowns) and under-ordering fast-moving staples (stockouts, lost revenue, and degraded shopper loyalty).

Across **54 stores** and **33 product families** spanning **3,000,888 historical daily records** (2013-01-01 to 2017-08-15), store managers currently rely on static heuristic ordering rules—predominantly 7-day seasonal naive repeating and 28-day moving averages.

Across an expanding-window **5-fold rolling-origin backtest** (origins stepping 16 days from 2017-05-11 to 2017-07-14), static heuristics yield:
- **Last-Value Naive:** **27.83% Mean WAPE** (MAE: 134.20 units).
- **28-Day Moving Average:** **23.65% Mean WAPE** (MAE: 114.12 units).
- **7-Day Seasonal Naive:** **16.33% Mean WAPE** (MAE: 78.99 units).

Our **headline result** across the 5-fold backtest demonstrates that the **selected LightGBM global regressor** captures bi-weekly paydays (15th and month-end), non-transferred public holidays, and promotional velocity, achieving **13.56% Mean WAPE** (MAE: 65.55 units, Bias: **-0.81%**) compared to **16.33% Mean WAPE** for 7-day Seasonal Naive, delivering **+2.77 percentage points of Forecast Value Added (FVA)** (**16.95% relative error reduction**).

As a single-window operational confirmation, on the final untouched **16-day holdout** (2017-07-31 to 2017-08-15, representing **13,319,179.78 units sold** across **28,512 store-family-day combinations**), the selected model achieves **15.16% WAPE** (MAE: 70.81 units, Bias: **+2.85%**) vs. **20.67% WAPE** for 7-day Seasonal Naive (**+5.51 pts FVA**, a **26.65% relative error reduction**). The holdout window is noticeably harder for the heuristic baseline due to the compounding volatility of the month-turn restocking surge and the Tuesday, August 15 payday.

*Note on WAPE and Bias:* WAPE is volume-weighted ($\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y}$), evaluating aggregate unit error across total chain volume rather than averaging percentage errors across SKUs. Bias was +2.85% on the holdout and averaged -0.81% across folds (range -4.0% to +1.9%), showing no systematic directional bias.

---

## 2. Target Users & Decisions Supported

| Persona | Primary Workflow | Key Decision Supported |
| :--- | :--- | :--- |
| **Category Demand Planner** | Weekly replenishment review | Determines 16-day order quantities $(Q = \hat{y}_{16d} + SS - I_{hand})$ by store-family. |
| **Store Operations Manager** | Daily receiving & shelf stocking | Prepares backroom staging space based on predicted inflow peaks. |
| **Regional Supply Chain Lead** | Distribution center load balancing | Schedules fleet shipments from central warehouses in Quito/Guayaquil to outlying provinces. |

---

## 3. Metric Tree

```mermaid
graph TD
    A["North Star: Working Capital & On-Shelf Availability"] --> B["Inventory Holding & Shrinkage Cost"]
    A --> C["On-Shelf Fill Rate (%)"]
    B --> D["Forecast Value Added (FVA vs Baseline)"]
    C --> D
    D --> E["Mean WAPE: 13.56% (vs 16.33% Baseline)"]
    D --> F["Mean Bias: -0.81% (Range -4.0% to +1.9%)"]
    D --> G["Mean MAE: 65.55 units (vs 78.99 Baseline)"]
    E --> H["Intermittent Families (7 families >50% zero-rate)"]
    E --> I["High-Velocity Staples (Grocery I, Beverages)"]
```

---

## 4. Key Assumptions & Product Limits

1. **Sales Observations Are Censored Demand:**
   Historical point-of-sale data records fulfilled transactions, not customer intent. When a shelf stocks out, sales register as zero. Without unfulfilled order logs, our forecasts represent an optimistic lower bound on true customer demand.
2. **Absence of Direct Inventory Data:**
   The benchmark dataset provides transactions and sales without live on-hand stock or supplier lead-time variances. Inventory simulations must parameterize target service levels (e.g., 90%, 95%, 98%) under periodic-review $(R, S)$ models.
3. **Observational Nature of Promotions:**
   Promotional tracking commenced on **2014-04-01** (prior to which `onpromotion` was unrecorded). Promotional uplifts represent observational historical associations rather than randomized A/B trial increments.
4. **Deterministic Institutional Closures:**
   All stores close on Christmas Day (**4 dates missing**: 2013-12-25, 2014-12-25, 2015-12-25, 2016-12-25). Demand for December 25 is treated as deterministically zero.
5. **Demand Intermittency:**
   **7 product families** exhibit $>50\%$ zero-sales days (e.g., `BOOKS` at **96.96%**, `BABY CARE` at **94.13%**). WAPE is the mandatory evaluation metric; percentage metrics with individual item denominators (MAPE) are structurally invalid.
6. **Training Window Inception (2015-01-01):**
   Training intentionally begins on **2015-01-01** rather than 2013-01-01. Promotional data was unrecorded in 2013 (0%) and only partially phased in during mid-2014; starting at 2015-01-01 ensures that all 16-to-28-day lags and rolling aggregates have complete promotional history, avoids artificial zero-sales biases from late-opening stores (e.g., Store 53 in May 2014, Store 20 in Feb 2015), and provides 2.5 years of clean, consistent post-inception data for gradient boosting.

---

## 5. Pilot Implementation & Rollout Plan

- **Pilot Scope:** 4-week A/B pilot across **10 stores** (5 treatment, 5 control) balanced across Store Types (A, B, C, D) and clusters.
- **Treatment Arm:** Replenishment quantities generated using LightGBM P50 predictions with dynamic safety buffers.
- **Control Arm:** Existing 7-day seasonal naive replenishment schedule.
- **Success Criteria:**
  1. $\ge 15\%$ reduction in recorded stockout incidents on top-20 revenue SKUs.
  2. Maintained or decreased end-of-cycle inventory write-offs.
  3. Planner override rate $< 15\%$ on suggested order quantities.

---

## 6. Scope & Roadmap: First-Ship vs. Later Iterations

| Feature Dimension | MVP / First-Ship (Current) | Phase 2 (Next Milestone) | Future Vision |
| :--- | :--- | :--- | :--- |
| **Forecast Horizon** | 16-day fixed multi-step horizon | Dynamic 1-30 day horizons | Real-time intra-day re-forecasting |
| **Model Architecture** | Global LightGBM Regressor | Multi-quantile P10/P50/P90 models | Hierarchical reconciliation (MinT) |
| **Promotions** | Promo item counts & active flags | Econometric uplift curve by discount % | Joint pricing & promo optimizer |
| **Inventory Logic** | Heuristic $(R, S)$ simulation | Cost-optimal safety stock sizing | Multi-echelon DC-to-store optimization |
| **Interface** | CLI pipeline & executive reports | Interactive Streamlit Scenario Planner | ERP / SAP automated order webhook |
