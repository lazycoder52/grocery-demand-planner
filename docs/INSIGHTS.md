# Operational Business Insights & SQL Analytics

This document connects the SQL analytical queries in `sql/` to concrete inventory, merchandising, and replenishment decisions for Corporación Favorita store managers and category buyers.

---

## Summary: Replenishment Benchmark

Under rigorous walk-forward backtesting with out-of-sample residual safety stock and Rule 2 boundary extension, the machine learning replenishment policy achieves:

* **Chain-Level Inventory Reduction:** **-12.27% to -14.51%** average inventory reduction compared to Seasonal Naive at a matched 99.0% target fill rate.
  * **$L=3$ days:** **-12.27%** inventory [95% Series Bootstrap CI: -14.59% to -10.05%; 95% Time-Block Bootstrap CI: -19.97% to -2.25%].
  * **$L=7$ days:** **-14.51%** inventory [95% Series Bootstrap CI: -17.53% to -11.37%; 95% Time-Block Bootstrap CI: -21.22% to -7.31%].
  * Both series and time-block 95% bootstrap confidence intervals strictly exclude 0 across both operational lead times.
* **Why Compare at Matched Fill Rate?** Evaluating policies at an identical safety multiplier $z$ conflates forecast accuracy with customer service: an over-forecasting model holds more inventory while achieving higher fill, whereas an under-forecaster appears lean while inducing stockouts. Matching policies at target on-shelf availability (e.g., 99.0% fill rate) controls for service level, providing an apples-to-apples working capital comparison.
* **Lead-Time Error Autocorrelation:** The empirical ratio of multi-day window-sum error standard deviation to theoretical independent $\sigma_{\text{oos}} \sqrt{R+L}$ is **1.436 at $W=10$ days** ($L=3$) and **1.522 at $W=14$ days** ($L=7$).
* **Conservative Temporal Uncertainty:** At 99.5% target fill rate ($L=3$), LightGBM Point reduces inventory by **-10.13%** with Series CI [-12.65%, -6.99%], but the conservative Time-Block CI spans **[-16.28%, +0.88%]**, crossing 0 due to temporal cluster variance during holiday/payday weeks.

* **Primary Pre-Specified Family Results (Matched 99.0% Fill Rate, Holm-Bonferroni FWER $\alpha = 0.05$):**
  To prevent false discovery, one primary comparison was pre-specified per family across $m = 13$ hypotheses per lead time. A family is designated as a finding **only if its Holm-adjusted p-value is below 0.05 AND both Series and Time-Block bootstrap CIs strictly exclude 0**:
  * **$L=3$ Primary Win (1 family):**
    * `LAWN AND GARDEN`: **-32.25%** inventory (absolute diff: **-2,780.66 units**, Series CI: [-41.50%, -23.58%], Time CI: [-45.14%, -20.01%], $p_{\text{raw}} < 0.001$, $p_{\text{adj}} < 0.001$).
  * **$L=3$ Exploratory Rows (Not Significant after Holm adjustment):**
    * `GROCERY I`: **-18.51%** (Series CI: [-23.26%, -13.84%], Time CI: [-27.42%, -3.20%]). Although both CIs exclude 0, raw p-value is $p_{\text{raw}} = 0.016$, exceeding rank-5 Holm threshold ($\alpha_5 = 0.00556$, $p_{\text{adj}} = 0.144 \ge 0.05$); correctly labeled **Exploratory (Holm-adj p >= 0.05)**.
    * `DELI`: **-17.63%** ($p_{\text{raw}} = 0.008$, $\alpha_2 = 0.00417$, $p_{\text{adj}} = 0.096 \ge 0.05$).
    * `MEATS`: **-9.23%** ($p_{\text{raw}} = 0.008$, $\alpha_3 = 0.00455$, $p_{\text{adj}} = 0.096 \ge 0.05$).
    * `POULTRY`: **-9.91%** ($p_{\text{raw}} = 0.030$, $\alpha_6 = 0.00625$, $p_{\text{adj}} = 0.240 \ge 0.05$).
    * `BREAD/BAKERY`: **-7.52%** (Time CI: [-14.26%, +0.95%], crosses 0).
  * **$L=7$ Primary Wins (6 families):**
    1. `LAWN AND GARDEN`: **-38.02%** ($p_{\text{adj}} < 0.001$, absolute diff: **-3,652.14 units**, Series CI: [-44.13%, -30.82%], Time CI: [-48.07%, -24.10%])
    2. `DELI`: **-22.73%** ($p_{\text{adj}} < 0.001$, Series CI: [-26.00%, -18.61%], Time CI: [-31.03%, -13.70%])
    3. `GROCERY I`: **-19.57%** ($p_{\text{adj}} < 0.001$, Series CI: [-26.68%, -12.58%], Time CI: [-30.60%, -7.83%])
    4. `POULTRY`: **-14.75%** ($p_{\text{adj}} < 0.001$, Series CI: [-21.47%, -8.68%], Time CI: [-22.29%, -6.45%])
    5. `BREAD/BAKERY`: **-10.02%** ($p_{\text{adj}} < 0.001$, Series CI: [-14.58%, -5.14%], Time CI: [-13.48%, -5.24%])
    6. `MEATS`: **-9.71%** ($p_{\text{raw}} = 0.004$, $\alpha_6 = 0.00625$, $p_{\text{adj}} = 0.032 < 0.05$, Series CI: [-17.07%, -4.18%], Time CI: [-16.66%, -2.63%])
  * **Reachability & Boundary Cases:**
    * `DAIRY`: Exceeds 99% fill rate across all tested safety factors even at negative safety stock ($z=-1.5$, min fill rate: 99.38% at $L=3$, 99.16% at $L=7$) under the sales-as-demand proxy. Target 99.0% is *target below frontier minimum*.
    * `BABY CARE`: Status is *target above frontier maximum* (maximum reachable fill rate is 92.65% for LightGBM Point and 94.38% for Seasonal Naive due to low-volume sparsity).
    * `BOOKS`: Reachable at 99.0% fill rate; $L=3$ Point uses -13.23% (Series CI: [-23.59%, +16.50%]), $L=7$ Point uses +18.44% (Series CI: [-21.00%, +47.65%]). Both differences are within noise (CIs cross 0).

* **Operational Limitations & Assumptions:**
  1. *Out-of-sample safety stock construction:* Forecast uncertainty $\sigma$ is strictly derived from historical out-of-sample forecast errors originating prior to review date $t$.
  2. *Demand proxy and censoring caveat:* Point-of-sale data records completed transactions, not true customer demand. When stockouts occur, unfulfilled demand is censored (recorded as zero sales), causing naive models to systematically under-forecast demand during replenishment recovery.
  3. *Untraceable Figures Explanation:* Earlier exploratory runs reported -12.05% ($L=3$) and -14.54% ($L=7$) inventory reductions. These figures were generated using in-sample residual $\sigma$ before the simulator was frozen with strictly out-of-sample forecast error buffers. Because they cannot be reproduced from any commit, they are declared invalid and are not used anywhere.

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
* **Evaluated Series:** 1,395 store-family pairs meet the minimum support thresholds (excluding dates prior to each store's first positive-sales date and Dec 25 closures).
* **Promotional Day Share:** Mean promotional day share is **36.98%** (median: **32.36%**).
* **Empirical Lift Distribution:**
  * Median empirical sales lift across store-families is **+32.15%**. (Following conservative reporting standards, only medians are reported for lift to avoid skewness from promotional base distortions).
* **Controlled Econometric Association (`metrics.json` `promo_lift_analysis`):**
  * Evaluated excluding zero-sales days, pre-opening periods, and store closures (see Section 5 below for full methodology and exploratory family estimates).

---

## 2. Zero-Sales-Day Risk Proxy (`sql/zero_sales_day_proxy.sql`)

### Operational Decision Driven
* **On-Shelf Availability Audits & Unobserved Stockout Detection:** Flags fast-moving store departments experiencing intermittent zero-sales streaks.
* **Proxy Warning:** Because on-hand inventory balances are unrecorded in the dataset, zero-sales days serve strictly as an empirical PROXY for potential stockouts or delayed openings, not confirmed out-of-stock events.

### Query Logic
* Measures zero-sales frequency against mean daily volume across 54 stores and 33 families for series with $\ge 60$ days post store opening date (excluding Dec 25 closures).
* Categorizes series into 4 empirical proxy segments:
  1. `HIGH_VELOCITY_ELEVATED_ZERO_SALES_DAYS_PROXY`: Mean sales $\ge 50$ units/day and $\ge 10\%$ zero-sales days.
  2. `MODERATE_ZERO_SALES_DAYS_PROXY`: Mean sales $\ge 15$ units/day and $\ge 25\%$ zero-sales days.
  3. `SLOW_MOVING_INTERMITTENT_TAIL`: Mean sales $< 5$ units/day and $\ge 50\%$ zero-sales days.
  4. `LOW_ZERO_SALES_DAYS`: Baseline operational continuity.

### Exact Computed Query Findings (`sql/zero_sales_day_proxy.sql`)
* **Evaluated Series:** 1,782 series meet the 60-day filter.
* **Tier Distribution:**
  * `LOW_ZERO_SALES_DAYS`: **1,289 series** (72.33%)
  * `SLOW_MOVING_INTERMITTENT_TAIL`: **304 series** (17.06%)
  * `HIGH_VELOCITY_ELEVATED_ZERO_SALES_DAYS_PROXY`: **115 series** (6.45%)
  * `MODERATE_ZERO_SALES_DAYS_PROXY`: **74 series** (4.15%)
* **Top Velocity Elevated Zero-Sales Series:**
  1. Store 8 `PRODUCE`: Mean daily sales **3,061.34 units/day**, **265 zero-sales days** out of 1,683 total days (**15.75%** zero-sales rate).
  2. Store 24 `PRODUCE`: Mean daily sales **2,267.36 units/day**, **223 zero-sales days** out of 1,683 total days (**13.25%** zero-sales rate).
  3. Store 1 `PRODUCE`: Mean daily sales **1,846.59 units/day**, **174 zero-sales days** out of 1,683 total days (**10.34%** zero-sales rate).

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
* Filters to post-store opening dates and excludes Dec 25 closures.
* Computes demand coefficient of variation:
  $$\text{CV} = \frac{\sigma_{\text{daily}}}{\mu_{\text{daily}}}$$
* Computes composite volatility exposure index: $\mu \times \text{CV} = \sigma_{\text{daily}}$.
* Segments series into 3 volatility tiers:
  1. `HIGH_VOLATILITY_PERISHABLE_PROXY`: Perishable with $\text{CV} \ge 0.75$.
  2. `MODERATE_VOLATILITY_PERISHABLE_PROXY`: Perishable with $\text{CV} < 0.75$.
  3. `NON_PERISHABLE_OR_LOW_VOLATILITY`: Shelf-stable categories.

### Exact Computed Query Findings (`sql/perishable_volatility_proxy.sql`)
* **Evaluated Series:** 1,503 series with $\ge 60$ total days and mean daily sales $\ge 1.0$ unit.
* **Tier Distribution:**
  * `NON_PERISHABLE_OR_LOW_VOLATILITY`: **1,073 series** (71.39%)
  * `MODERATE_VOLATILITY_PERISHABLE_PROXY`: **349 series** (23.22%)
  * `HIGH_VOLATILITY_PERISHABLE_PROXY`: **81 series** (5.39%)
* **Highest Volatility Series & Anomaly Investigation:**
  * Store 39 `MEATS`: Mean daily sales **240.11 units/day**, daily sales standard deviation **2,223.16 units**, resulting in $\text{CV} = \mathbf{9.2585}$ and volatility exposure index of **2,223.16**.
  * **Investigation of Store 39 MEATS:** Deep-dive analysis confirms this extreme volatility is driven by an isolated **single-day data entry / recording anomaly** on **2016-10-07**, where recorded meat sales spiked to **89,576.36 units** (accounting for 22.2% of the store's entire 4.5-year cumulative meat sales in a single day; the 2nd highest day was 5,519 units). Store 39 has traded continuously since 2013-01-02 (the second day of the entire dataset), confirming that this is an anomalous recording spike, not a store opening effect.
