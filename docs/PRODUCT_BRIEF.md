# Product Brief: ML Demand Forecasting & Replenishment System

**Product:** Corporación Favorita Demand Planner
**Role:** AI/ML Product Manager & Senior Product Analyst
**Target Organization:** Corporación Favorita (Ecuadorian Grocery Chain)
**Status:** Validated Pipeline & Re-frozen Replenishment Benchmark

---

## 1. Problem Statement & Business Opportunity

Grocery supply chains operate under an unforgiving trade-off: over-replenishing inventory triggers margin-eroding spoilage and capital holding costs, while under-replenishing leads to phantom stockouts, lost sales, and degraded customer loyalty.

Across **54 supermarket stores** and **33 product families** in Ecuador (3,000,888 historical daily POS records), store inventory replenishment has historically relied on static heuristic ordering rules—predominantly 7-day seasonal naive repeating and 28-day moving averages.

Under walk-forward evaluation, static heuristics prove inadequate:
* **7-Day Seasonal Naive:** **16.33% Mean WAPE** across 5 backtest folds; **20.67% WAPE** on the final 16-day holdout.
* **Selected LightGBM Model:** **13.56% Mean WAPE** across 5 backtest folds (**+2.77 pts FVA**, 16.96% relative error reduction); **15.16% WAPE** on the untouched 16-day holdout (2017-07-31 to 2017-08-15, **13,319,179.78 units sold**, **+5.51 pts FVA**, 26.66% relative error reduction).

### Business Impact: Working Capital Optimization
Under periodic $(R, S)$ inventory control with out-of-sample error safety stock:
* **Chain-Level Inventory Reduction:** **-12.27% to -14.51%** average inventory reduction compared to Seasonal Naive at a matched 99.0% fill rate.
  * **$L = 3$ days:** **-12.27%** inventory [95% Series Bootstrap CI: -14.59% to -10.05%; 95% Time-Block Bootstrap CI: -19.97% to -2.25%].
  * **$L = 7$ days:** **-14.51%** inventory [95% Series Bootstrap CI: -17.53% to -11.37%; 95% Time-Block Bootstrap CI: -21.22% to -7.31%].
  * Both series and time-block 95% bootstrap confidence intervals strictly exclude 0 across both operational lead times.

---

## 2. Target Personas & Key Decisions Supported

| Persona | Primary Workflow | Key Operational Decision Supported |
| :--- | :--- | :--- |
| **Category Demand Planner** | Weekly replenishment review | Sets order-up-to level $S = \hat{D}_{R+L} + z \cdot \sigma_{\text{oos}}$ by store-family. |
| **Store Operations Manager** | Daily receiving & stockroom staging | Anticipates high-volume delivery days and balances floor shelf labor. |
| **Category Merchant** | Promotional campaign design | Sizes supplier buy-in orders using empirical promo lift (+171.2% chain average). |

---

## 3. Value Driver Metric Tree

```mermaid
graph TD
    A["North Star: Working Capital & Customer Service Level"] --> B["Working Capital Freed (-12.3% to -14.5% Inventory)"]
    A --> C["On-Shelf Fill Rate (Matched 99.0% Target)"]
    B --> D["Safety Stock Buffer Compression (z * sigma_oos)"]
    C --> D
    D --> E["Residual Forecast Error Reduction"]
    E --> F["5-Fold Backtest WAPE: 13.56% (vs 16.33% Naive)"]
    E --> G["Holdout WAPE: 15.16% (vs 20.67% Naive)"]
    F --> H["High-Velocity Staples (Grocery I, Bread/Bakery)"]
    F --> I["Perishable Departments (Poultry, Deli, Meats)"]
```

---

## 4. Pre-Specified Primary Family Results

To prevent false discovery from multiple comparisons, one primary comparison was pre-specified per family: **Target Fill Rate = 99.0%, LightGBM Point vs Seasonal Naive, evaluated at both $L=3$ and $L=7$**, with statistical significance requiring both 95% Series and Time-Block CIs to exclude 0 under Holm-Bonferroni correction ($\alpha = 0.05$ across 13 families).

* **$L=3$ Primary Win:**
  * `LAWN AND GARDEN`: **-32.25%** inventory (absolute diff: **-2,780.66 units**, Series CI: [-41.50%, -23.58%], Time CI: [-45.14%, -20.01%], $p < 0.001$).
* **$L=7$ Primary Wins (6 families):**
  1. `GROCERY I`: **-19.57%** ($p < 0.001$, Series CI: [-26.68%, -12.58%], Time CI: [-30.60%, -7.83%])
  2. `BREAD/BAKERY`: **-10.02%** ($p < 0.001$, Series CI: [-14.58%, -5.14%], Time CI: [-13.48%, -5.24%])
  3. `POULTRY`: **-14.75%** ($p < 0.001$, Series CI: [-21.47%, -8.68%], Time CI: [-22.29%, -6.45%])
  4. `DELI`: **-22.73%** ($p < 0.001$, Series CI: [-26.00%, -18.61%], Time CI: [-31.03%, -13.70%])
  5. `LAWN AND GARDEN`: **-38.02%** ($p < 0.001$, absolute diff: **-3,652.14 units**, Series CI: [-44.13%, -30.82%], Time CI: [-48.07%, -24.10%])
  6. `MEATS`: **-9.71%** ($p = 0.004 \le 0.00625$, Series CI: [-17.07%, -4.18%], Time CI: [-16.66%, -2.63%])
* **Boundary & Exploratory Cases:**
  * `BABY CARE`: Status is *target above frontier maximum* (frontier max fill rate is 92.65% for Point, 94.38% for Seasonal Naive due to low-volume sparsity).
  * `BOOKS`: Reachable at 99.0% fill rate; $L=3$ uses -13.23% (-26.46 units, Series CI: [-23.59%, +16.50%]), $L=7$ uses +18.44% (+30.19 units, Series CI: [-21.00%, +47.65%]). Both differences are within noise (CIs cross 0).

---

## 5. Critical Assumptions & Methodological Limitations

1. **Out-of-Sample Safety Stock Construction:** Safety stock residual $\sigma_{\text{oos}}$ is strictly computed from out-of-sample forecast errors originating prior to review date $t$.
2. **Demand Proxy and Censoring Caveat:** Point-of-sale records capture fulfilled transactions, not unconstrained consumer demand. When a stockout occurs, unobserved demand is censored, which can bias naive baselines downward during replenishment recovery.
3. **Observational Promotions:** Promotional tracking commenced on 2014-04-01; promotional lift reflects observational historical co-occurrence rather than randomized pricing trials.
4. **Institutional Calendar Shocks:** December 25 is a deterministic chain-wide closure (sales = 0). Bi-weekly paydays (15th and month-end) create recurring non-linear demand spikes.

---

## 6. 5-Store Pilot Implementation Protocol

* **Design:** 6-week randomized controlled field experiment across **10 matched stores** (5 treatment, 5 control).
* **Stratification:** Matched pairs balanced across Store Types (A, B, C, D) and regional clusters in Pichincha and Guayas.
* **Treatment Arm:** Weekly replenishment quantities generated via LightGBM Point forecasts with out-of-sample safety stock ($R=7, L=3$).
* **Control Arm:** Existing 7-day seasonal naive heuristic replenishment schedule.
* **Pilot Success Metrics:**
  1. $\ge 10\%$ reduction in average working capital tied to stock on shelf.
  2. Maintained customer on-shelf availability ($\ge 99.0\%$ measured fill rate).
  3. $\le 15\%$ planner manual override rate on recommended replenishment orders.

---

## 7. Product Scope: What to Ship First vs. Build Later

| Capability Dimension | What to Ship First (MVP / v1.0) | What to Build Later (Phase 2 & Roadmap) |
| :--- | :--- | :--- |
| **Forecasting Engine** | Global LightGBM point model with recursive multi-step lags | Direct multi-quantile forecasting (P10/P50/P90) & deep learning (TFT) |
| **Replenishment Logic** | Periodic $(R, S)$ order-up-to policy with out-of-sample safety stock | Multi-echelon distribution center to store joint optimization |
| **Service Level Targets** | Frontier service-level tuning grid ($z \in [-1.5, 2.5]$) | Category-specific dynamic service-level optimization based on margins |
| **User Interface** | Interactive Streamlit executive dashboard & SQL analytical exports | Enterprise ERP / SAP automated purchase order write-back integration |
| **Causal Controls** | Store-family fixed effects log-linear regression for promotional lift | Real-time cross-elasticity pricing engine and markdown planner |
