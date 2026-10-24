# Operational Business Insights & SQL Analytics

This document connects the production SQL queries in `sql/` to concrete inventory, merchandising, and replenishment decisions for Corporación Favorita store managers and category buyers.

---

## Executive Summary: Replenishment Benchmark

Under rigorous walk-forward backtesting with out-of-sample residual safety stock and Rule 2 boundary extension, the machine learning replenishment policy achieves:

* **Chain-Level Inventory Reduction:** **-12.27% to -14.51%** average inventory reduction compared to Seasonal Naive at a matched 99.0% target fill rate.
  * **$L=3$ days:** **-12.27%** inventory [95% Series Bootstrap CI: -14.59% to -10.05%; 95% Time-Block Bootstrap CI: -19.97% to -2.25%].
  * **$L=7$ days:** **-14.51%** inventory [95% Series Bootstrap CI: -17.53% to -11.37%; 95% Time-Block Bootstrap CI: -21.22% to -7.31%].
  * Both series and time-block 95% bootstrap confidence intervals strictly exclude 0 across both operational lead times.

* **Primary Pre-Specified Family Results (Matched 99.0% Fill Rate, Holm-Bonferroni FWER $\alpha = 0.05$):**
  * **$L=3$ Primary Win:**
    * `LAWN AND GARDEN`: **-32.25%** inventory (absolute diff: **-2,780.66 units**, Series CI: [-41.50%, -23.58%], Time CI: [-45.14%, -20.01%], $p < 0.001$).
  * **$L=7$ Primary Wins (6 families):**
    1. `GROCERY I`: **-19.57%** ($p < 0.001$, Series CI: [-26.68%, -12.58%], Time CI: [-30.60%, -7.83%])
    2. `BREAD/BAKERY`: **-10.02%** ($p < 0.001$, Series CI: [-14.58%, -5.14%], Time CI: [-13.48%, -5.24%])
    3. `POULTRY`: **-14.75%** ($p < 0.001$, Series CI: [-21.47%, -8.68%], Time CI: [-22.29%, -6.45%])
    4. `DELI`: **-22.73%** ($p < 0.001$, Series CI: [-26.00%, -18.61%], Time CI: [-31.03%, -13.70%])
    5. `LAWN AND GARDEN`: **-38.02%** ($p < 0.001$, absolute diff: **-3,652.14 units**, Series CI: [-44.13%, -30.82%], Time CI: [-48.07%, -24.10%])
    6. `MEATS`: **-9.71%** ($p = 0.004 \le 0.00625$, Series CI: [-17.07%, -4.18%], Time CI: [-16.66%, -2.63%])
  * **Reachability & Boundary Cases:**
    * `BABY CARE`: Status is *target above frontier maximum* (maximum reachable fill rate is 92.65% for LightGBM Point and 94.38% for Seasonal Naive due to low-volume sparsity).
    * `BOOKS`: Reachable at 99.0% fill rate; $L=3$ Point uses -13.23% (-26.46 units, Series CI: [-23.59%, +16.50%]), $L=7$ Point uses +18.44% (+30.19 units, Series CI: [-21.00%, +47.65%]). Both differences are within noise (CIs cross 0).

* **Operational Limitations & Assumptions:**
  1. *Out-of-sample safety stock construction:* Forecast uncertainty $\sigma$ is strictly derived from historical out-of-sample forecast errors originating prior to review date $t$.
  2. *Demand proxy and censoring caveat:* Point-of-sale data records completed transactions, not true customer demand. When stockouts occur, unfulfilled demand is censored (recorded as zero sales), causing naive models to systematically under-forecast demand during replenishment recovery.

---

## 1. Promotional Elasticity & Sales Lift (`sql/promo_elasticity.sql`)

### Operational Decision Driven
* **Promotional Stock Allocation & Vendor Co-Op Negotiation:** Identifies store-family pairs exhibiting high empirical sales responsiveness during promotional periods to guide safety stock adjustments, while flagging low-elasticity categories where price promotions dilute gross margin without driving volume.

### Query Logic
* Evaluates store-family sales post **2014-04-01** (when promotional tracking commenced across Corporación Favorita).
* Partitions days into promotional (`onpromotion > 0`) vs baseline non-promotional days (`onpromotion = 0`).
* Computes empirical daily sales velocity lift:
  $$\text{Lift \%} = \frac{\bar{S}_{\text{promo}} - \bar{S}_{\text{base}}}{\bar{S}_{\text{base}}} \times 100$$
* Enforces minimum support thresholds ($\ge 30$ baseline days, $\ge 10$ promotional days) to suppress sparse outlier ratios.

### Exact Computed Query Findings (`sql/promo_elasticity.sql`)
* **Evaluated Series:** 1,412 store-family pairs meet the minimum support thresholds.
* **Mean Promotional Day Share:** 35.13% of days in evaluated series have active promotions.
* **Empirical Lift Distribution:**
  * Median empirical sales lift across store-families is **+42.34%** (mean: **+769.73%**).
  * Outlier: Store 52 `BEVERAGES` exhibits an empirical lift of **+125,576.31%** (117 promo days vs 1,113 non-promo days, reflecting Store 52 opening on 2017-04-20 and pre-opening zero-sales days).
* **Controlled Econometric Association (`metrics.json` `promo_lift_analysis`):**
  * When evaluated with store-family and year-month fixed effects over 33,264 series-month observations, the overall sales association with promotion is **+107.68%** (95% CI: [+72.29%, +155.26%]).
  * By-family ranked associations: `GROCERY I` (+1592.95%), `SCHOOL AND OFFICE SUPPLIES` (+602.95%), `BEVERAGES` (+365.68%), `CLEANING` (+197.12%), `DAIRY` (+133.52%), `PRODUCE` (+111.60%).

---

## 2. Zero-Sales-Day Risk Proxy (`sql/zero_sales_day_proxy.sql`)

### Operational Decision Driven
* **On-Shelf Availability Audits & Phantom Inventory Detection:** Flags fast-moving store departments experiencing intermittent zero-sales streaks.
* **Proxy Warning:** Because on-hand inventory balances are unrecorded in the dataset, zero-sales days serve strictly as an empirical PROXY for potential stockouts or delayed openings, not confirmed out-of-stock events.

### Query Logic
* Measures zero-sales frequency against mean daily volume across 54 stores and 33 families for series with $\ge 60$ days.
* Categorizes series into 4 empirical proxy segments:
  1. `HIGH_VELOCITY_ELEVATED_ZERO_SALES_DAYS_PROXY`: Mean sales $\ge 50$ units/day and $\ge 10\%$ zero-sales days.
  2. `MODERATE_ZERO_SALES_DAYS_PROXY`: Mean sales $\ge 15$ units/day and $\ge 25\%$ zero-sales days.
  3. `SLOW_MOVING_INTERMITTENT_TAIL`: Mean sales $< 5$ units/day and $\ge 50\%$ zero-sales days.
  4. `LOW_ZERO_SALES_DAYS`: Baseline operational continuity.

### Exact Computed Query Findings (`sql/zero_sales_day_proxy.sql`)
* **Evaluated Series:** 1,782 series meet the 60-day filter.
* **Tier Distribution:**
  * `LOW_ZERO_SALES_DAYS`: **1,121 series** (62.91%)
  * `SLOW_MOVING_INTERMITTENT_TAIL`: **370 series** (20.76%)
  * `HIGH_VELOCITY_ELEVATED_ZERO_SALES_DAYS_PROXY`: **195 series** (10.94%)
  * `MODERATE_ZERO_SALES_DAYS_PROXY`: **96 series** (5.39%)
* **Top Velocity Elevated Zero-Sales Series:**
  1. Store 8 `PRODUCE`: Mean daily sales **3,059.52 units/day**, **266 zero-sales days** out of 1,684 total days (**15.80%** zero-sales rate).
  2. Store 20 `GROCERY I`: Mean daily sales **2,841.46 units/day**, **773 zero-sales days** out of 1,684 total days (**45.90%** zero-sales rate, driven by pre-opening / relocation periods).
  3. Store 24 `PRODUCE`: Mean daily sales **2,266.02 units/day**, **224 zero-sales days** out of 1,684 total days (**13.30%** zero-sales rate).

---

## 3. Weekly Multi-Source Analytical View (`sql/weekly_store_family.sql`)

### Operational Decision Driven
* **Integrated S&OP (Sales and Operations Planning) Planning:** Consolidates store POS sales, customer transaction counts, macroeconomic oil shocks, and civic holiday disruptions into a weekly grain for cross-functional review.

### Query Logic
* Aggregates daily observations into Monday-aligned calendar weeks (`date_trunc('week', date)`).
* Joins store transaction counts, national oil prices (WTI crude benchmark), and holiday calendars (tracking transferred vs non-transferred civic holidays).

### Exact Computed Query Findings (`sql/weekly_store_family.sql`)
* **Record Count:** Exactly **431,244 store-family-week rows** generated across 54 stores and 33 families.
* **Date Span:** Spans **2012-12-31 to 2017-08-14** (242 continuous calendar weeks).
* **Total Sales Volume:** Aggregates **1,073,644,970.12 total units** sold across the entire chain history.

---

## 4. Perishable Demand Volatility Proxy (`sql/perishable_volatility_proxy.sql`)

### Operational Decision Driven
* **Perishable Buffer Sizing & Clearance Strategy:** Identifies fresh food categories where high demand volatility (coefficient of variation $\text{CV} = \sigma / \mu$) elevates the risk of stock imbalance.
* **Proxy Warning:** Physical spoilage, scrap, and shrinkage logs are unrecorded in the dataset. Demand CV serves as an empirical proxy for inventory volatility exposure, not an audit of confirmed physical shrinkage.

### Query Logic
* Isolates perishable departments (`DAIRY`, `MEATS`, `POULTRY`, `PRODUCE`, `SEAFOOD`, `BREAD/BAKERY`, `DELI`, `EGGS`).
* Computes demand coefficient of variation:
  $$\text{CV} = \frac{\sigma_{\text{daily}}}{\mu_{\text{daily}}}$$
* Computes composite volatility exposure index: $\mu \times \text{CV} = \sigma_{\text{daily}}$.
* Segments series into 3 volatility tiers:
  1. `HIGH_VOLATILITY_PERISHABLE_PROXY`: Perishable with $\text{CV} \ge 0.75$.
  2. `MODERATE_VOLATILITY_PERISHABLE_PROXY`: Perishable with $\text{CV} < 0.75$.
  3. `NON_PERISHABLE_OR_LOW_VOLATILITY`: Shelf-stable categories.

### Exact Computed Query Findings (`sql/perishable_volatility_proxy.sql`)
* **Evaluated Series:** 1,483 series with $\ge 60$ total days and mean daily sales $\ge 1.0$ unit.
* **Tier Distribution:**
  * `NON_PERISHABLE_OR_LOW_VOLATILITY`: **1,056 series** (71.21%)
  * `MODERATE_VOLATILITY_PERISHABLE_PROXY`: **297 series** (20.03%)
  * `HIGH_VOLATILITY_PERISHABLE_PROXY`: **130 series** (8.77%)
* **Highest Volatility Series:**
  * Store 39 `MEATS`: Mean daily sales **239.97 units/day**, daily sales standard deviation **2,222.50 units**, resulting in $\text{CV} = \mathbf{9.2616}$ and volatility exposure index of **2,222.50**.
