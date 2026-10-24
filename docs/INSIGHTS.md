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
* **Promotional Stock Allocation & Vendor Co-Op Negotiation:** Determines which store-family pairs require temporary order-up-to target increases during promotional campaigns to avoid stockouts, and flags low-elasticity categories where price promotions dilute gross margin without driving volume.

### Query Logic
* Restricts analysis to post **2014-04-01** (the exact date promotional tracking began across Favorita).
* Segregates days into promotional (`onpromotion > 0`) vs baseline non-promotional days.
* Computes empirical daily sales velocity lift:
  $$\text{Lift \%} = \frac{\bar{S}_{\text{promo}} - \bar{S}_{\text{base}}}{\bar{S}_{\text{base}}} \times 100$$
* Enforces minimum support thresholds ($\ge 30$ baseline days, $\ge 10$ promotional days) to suppress sparse outlier ratios.

### Key Business Takeaway
* Perishable fresh departments (e.g., `PRODUCE`, `GROCERY I`, `HOME CARE`) show promotional sales multipliers exceeding +500% to +1000%, demanding dynamic safety stock adjustments during planned flyer dates. Conversely, non-promotional staples with low lift should not receive promotional inventory build-up.

---

## 2. Stockout Risk & Censored Demand (`sql/stockout_risk.sql`)

### Operational Decision Driven
* **On-Shelf Availability (OSA) Audits & Phantom Inventory Detection:** Flags fast-moving store departments experiencing intermittent zero-sales streaks. In grocery retailing, high-velocity items with repeated zero-sales days represent stockouts or misplaced inventory rather than genuine zero consumer demand.

### Query Logic
* Computes overall zero-sales day frequency against mean daily volume across 54 stores and 33 families.
* Segments inventory into 4 operational risk tiers:
  1. `CRITICAL_HIGH_VELOCITY_STOCKOUT_RISK`: High mean volume ($\ge 50$ units/day) with $\ge 10\%$ zero-sales days. Requires immediate physical shelf recount and minimum display stock re-benchmarking.
  2. `MODERATE_STOCKOUT_RISK`: Moderate volume ($\ge 15$ units/day) with $\ge 25\%$ zero-sales days.
  3. `SLOW_MOVING_INTERMITTENT_TAIL`: Low volume ($< 5$ units/day) with $\ge 50\%$ zero-sales days.
  4. `STABLE_IN_STOCK`: Normal operational turnover.

### Key Business Takeaway
* Categorizing series into risk tiers prevents the forecaster from treating stockout-induced zero sales as natural demand drop-offs, directly addressing the demand censoring caveat.

---

## 3. Weekly Multi-Source Analytical View (`sql/weekly_store_family.sql`)

### Operational Decision Driven
* **Integrated S&OP (Sales and Operations Planning) Dashboard:** Unifies store POS sales, store traffic (transactions), macroeconomic oil shocks, and public holiday disruptions into a consolidated weekly grain for supply chain planning.

### Query Logic
* Groups daily data into calendar weeks (Monday-aligned `date_trunc('week', date)`).
* Enriches sales velocity with:
  * Store transactions (measuring customer footfall per store).
  * National oil price (WTI crude benchmark, Ecuador's primary macro driver).
  * Holiday event counts, distinguishing transferred from non-transferred civic holidays.
  * Store physical location metadata (city, state, store type, store cluster).

### Key Business Takeaway
* Disentangles whether store-level sales drops stem from chain-wide macro swings (oil price crashes), store-specific traffic declines, or holiday calendar shifts, enabling category managers to hold regional replenishment reviews.

---

## 4. Perishable Spoilage & Waste Proxy (`sql/shrinkage_or_waste_proxy.sql`)

### Operational Decision Driven
* **Perishable Write-Off Minimization & Dynamic Clearance:** Pinpoints fresh food categories with high demand volatility (coefficient of variation $\text{CV} = \sigma / \mu$) where supply-chain over-replenishment directly causes physical rotting, product shrinkage, and markdown write-offs.

### Query Logic
* Isolates perishable families (`DAIRY`, `MEATS`, `POULTRY`, `PRODUCE`, `SEAFOOD`, `BREAD/BAKERY`, `DELI`, `EGGS`) having strict shelf lives (1 to 7 days).
* Computes demand coefficient of variation:
  $$\text{CV} = \frac{\sigma_{\text{daily}}}{\mu_{\text{daily}}}$$
* Calculates a composite `spoilage_exposure_index = \mu \times \text{CV}`.
* Assigns risk tiers (`HIGH_SPOILAGE_RISK`, `MODERATE_SPOILAGE_RISK`, `LOW_NON_PERISHABLE`).

### Key Business Takeaway
* Highly volatile perishables cannot absorb standard fixed safety stocks without catastrophic shrinkage. For high-spoilage items, replenishment policies must enforce asymmetric loss functions (penalizing over-stocking more heavily than stockouts) or adopt shorter review frequencies ($R=1$ instead of $R=7$).
