# Data Card: [Store Sales - Time Series Forecasting (Kaggle; Corporación Favorita data)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

## 1. Overview & Dataset Description

This project utilizes the public [Store Sales - Time Series Forecasting (Kaggle; Corporación Favorita data)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset.
Corporación Favorita is one of Ecuador's largest supermarket chains.

- **Primary Source:** Kaggle / Corporación Favorita
- **Geographic Scope:** Ecuador (national chain across multiple cities and provinces)
- **Primary Granularity:** Daily unit sales by Store (`store_nbr`) and Product Family (`family`)
- **Total Stores:** 54
- **Total Product Families:** 33

## 2. Ingestion & Table Summary

| Table | Rows | Date Range | Primary Key / Grain |
| --- | --- | --- | --- |
| `train` | 3,000,888 | 2013-01-01 to 2017-08-15 | `id` (date x store_nbr x family) |
| `test` | 28,512 | 2017-08-16 to 2017-08-31 | `id` (16 days x 54 stores x 33 families) |
| `stores` | 54 | Static | `store_nbr` (city, state, type, cluster) |
| `oil` | 1,218 | 2013-01-01 to 2017-08-31 | `date` (WTI spot price) |
| `holidays_events` | 350 | 2012-03-02 to 2017-12-26 | `date`, `type`, `transferred` |
| `transactions` | 83,488 | 2013-01-01 to 2017-08-15 | `date` x `store_nbr` |
| `sample_submission` | 28,512 | 16-day evaluation | `id` |

## 3. Data Audit Findings

### 3.1 Missing Calendar Dates (Store Closures)
Across the 2013-01-01 to 2017-08-15 training period (3,000,888 rows),
exactly 4 calendar days are completely missing from `train.csv`:
`2013-12-25`, `2014-12-25`, `2015-12-25`, `2016-12-25`

**Reason:** In Ecuador, retail operations for Corporación Favorita close entirely on
Christmas Day (`December 25`). The Kaggle data omits these dates rather than providing zero sales.

### 3.2 Late-Opening Stores
Several stores opened well after January 2013:
- **Store 1:** First recorded positive sale on `2013-01-02`.
- **Store 2:** First recorded positive sale on `2013-01-02`.
- **Store 3:** First recorded positive sale on `2013-01-02`.
- **Store 4:** First recorded positive sale on `2013-01-02`.
- **Store 5:** First recorded positive sale on `2013-01-02`.
- **Store 6:** First recorded positive sale on `2013-01-02`.
- **Store 7:** First recorded positive sale on `2013-01-02`.
- **Store 8:** First recorded positive sale on `2013-01-02`.
- **Store 9:** First recorded positive sale on `2013-01-02`.
- **Store 10:** First recorded positive sale on `2013-01-02`.
- **Store 11:** First recorded positive sale on `2013-01-02`.
- **Store 12:** First recorded positive sale on `2013-01-02`.
- **Store 13:** First recorded positive sale on `2013-01-02`.
- **Store 14:** First recorded positive sale on `2013-01-02`.
- **Store 15:** First recorded positive sale on `2013-01-02`.
- **Store 16:** First recorded positive sale on `2013-01-02`.
- **Store 17:** First recorded positive sale on `2013-01-02`.
- **Store 18:** First recorded positive sale on `2013-01-02`.
- **Store 19:** First recorded positive sale on `2013-01-02`.
- **Store 20:** First recorded positive sale on `2015-02-13`.
- **Store 21:** First recorded positive sale on `2015-07-24`.
- **Store 22:** First recorded positive sale on `2015-10-09`.
- **Store 23:** First recorded positive sale on `2013-01-02`.
- **Store 24:** First recorded positive sale on `2013-01-02`.
- **Store 26:** First recorded positive sale on `2013-01-02`.
- **Store 27:** First recorded positive sale on `2013-01-02`.
- **Store 28:** First recorded positive sale on `2013-01-02`.
- **Store 29:** First recorded positive sale on `2015-03-20`.
- **Store 30:** First recorded positive sale on `2013-01-02`.
- **Store 31:** First recorded positive sale on `2013-01-02`.
- **Store 32:** First recorded positive sale on `2013-01-02`.
- **Store 33:** First recorded positive sale on `2013-01-02`.
- **Store 34:** First recorded positive sale on `2013-01-02`.
- **Store 35:** First recorded positive sale on `2013-01-02`.
- **Store 36:** First recorded positive sale on `2013-05-09`.
- **Store 37:** First recorded positive sale on `2013-01-02`.
- **Store 38:** First recorded positive sale on `2013-01-02`.
- **Store 39:** First recorded positive sale on `2013-01-02`.
- **Store 40:** First recorded positive sale on `2013-01-02`.
- **Store 41:** First recorded positive sale on `2013-01-02`.
- **Store 42:** First recorded positive sale on `2015-08-21`.
- **Store 43:** First recorded positive sale on `2013-01-02`.
- **Store 44:** First recorded positive sale on `2013-01-02`.
- **Store 45:** First recorded positive sale on `2013-01-02`.
- **Store 46:** First recorded positive sale on `2013-01-02`.
- **Store 47:** First recorded positive sale on `2013-01-02`.
- **Store 48:** First recorded positive sale on `2013-01-02`.
- **Store 49:** First recorded positive sale on `2013-01-02`.
- **Store 50:** First recorded positive sale on `2013-01-02`.
- **Store 51:** First recorded positive sale on `2013-01-02`.
- **Store 52:** First recorded positive sale on `2017-04-20`.
- **Store 53:** First recorded positive sale on `2014-05-29`.
- **Store 54:** First recorded positive sale on `2013-01-02`.

Prior to opening, records for these stores exist in `train.csv` with zero sales.

### 3.3 Intermittent Families & Zero-Sales Share
Sales intermittency varies dramatically across product categories:
- **Intermittent Families (>50% zero-sales days):** `BOOKS`, `BABY CARE`, `SCHOOL AND OFFICE SUPPLIES`, `HOME APPLIANCES`, `LADIESWEAR`, `MAGAZINES`, `PET SUPPLIES`
- **Fast-Moving Grocery Staples (<10% zero-sales days):** `DAIRY`, `DELI`, `MEATS`,
  `BREAD/BAKERY`, `EGGS`, `PRODUCE`, `GROCERY I`.
  Their non-zero rate is ~92% (zero sales occur almost exclusively on closures or Sundays).

### 3.4 Promotional Tracking Evolution
The recording of promotional activity (`onpromotion` column) evolved over time:
- **2013:** 0% tracked (`onpromotion` = 0 for the entire year).
- **First Tracked Promotion Date:** `2014-04-01`.

| Year | Active Promo Rows | Promo Share (%) | Mean Promo Count |
| --- | --- | --- | --- |
| 2013 | 0 | 0.0% | 0.0 |
| 2014 | 64,979 | 10.02% | 0.71 |
| 2015 | 135,302 | 20.86% | 1.68 |
| 2016 | 227,981 | 35.05% | 5.4 |
| 2017 | 183,067 | 45.26% | 6.8 |

### 3.5 Oil Price Series & Missing Values
- Total oil price observations: 1,218
- Weekday missing price observations: 43 (3.53%)
- In addition, weekends are omitted from the price table.

### 3.6 Holiday Structure & Transferred Observances
- Total holiday / event entries: 350
- Transferred holidays: 12 holidays had `transferred = True`,
  accompanied by a corresponding `Transfer` entry on the actual observed date.
- Major Events: Includes the April 16, 2016 Manabi Earthquake (`Terremoto Manabi`),
  which triggered massive emergency relief procurement and distribution spikes.

### 3.7 Sales Distribution & Extreme Outliers
- **Median (P50):** 11.0 units
- **90th Percentile (P90):** 867.0 units
- **99th Percentile (P99):** 5,507.0 units
- **99.9th Percentile (P99.9):** 12,076.23 units
- **Max Single Record:** 124,717.0 units (relief distribution during earthquake aftermath)

### 3.8 Store 39 MEATS Recording Anomaly & Model Sensitivity
- **Single-Day Spike:** On `2016-10-07`, recorded meat sales for Store 39 spiked to **89,576.36 units** (accounting for 22.2% of the store's entire 4.5-year cumulative meat sales in a single day, compared to its non-spike median of 136.94 units and 2nd highest day of 5,519 units). Store 39 has traded continuously since 2013-01-02, confirming this is an isolated data entry / recording anomaly rather than a store opening effect.
- **Model Sensitivity Analysis:**
  - *Holdout (2017-07-31 to 2017-08-15):* Replacing the 2016-10-07 spike with the non-spike series median (136.94 units) shifts Global Holdout WAPE from **15.29%** (15.2888%) with the spike to **15.15%** (15.1547%) without the spike ($-0.13$ percentage points), and Store 39 MEATS holdout WAPE from **17.12%** (17.1183%) with the spike to **17.26%** (17.2644%) without the spike ($+0.15$ percentage points, holdout only).
  - *5-Fold Rolling Cross-Validation:* LightGBM 5-fold mean WAPE is **13.49%** (13.4880%) with the spike vs. **13.54%** (13.5368%) without the spike ($+0.05$ percentage points). Seasonal Naive (7-day) 5-fold mean WAPE is unchanged at **16.33%** (16.3282%) with and without the spike ($+0.00$ percentage points), because the October 2016 spike is more than 215 days prior to the first CV fold origin (2017-05-11).
  - *Headline Policy:* All reported headline metrics in the repository are unchanged and retain the spike as an audited historical dataset artifact.

## 4. Cleaning Decisions & Modeling Assumptions

1. **Sales Are Not True Demand (Censoring):**
   Sales observations represent fulfilled demand. When a store stocks out, recorded sales
   drop to zero regardless of customer intent. Simulated replenishment and fill rates
   derived from sales data must be interpreted as optimistic lower bounds on true demand variance.

2. **Observational Nature of Promotions & Assortment Whitespace:**
   Promotional assignments are not randomized A/B tests; marketing teams selectively discount
   high-velocity items or distressed inventory. All uplift and assortment-gap models are
   observational associations, not experimental treatment effects.

3. **Information Leakage Prevention (Forecast Origin Cutoff):**
   Same-day transactions and same-day oil prices are NEVER permitted as features for predicting
   same-day sales. At forecast origin $t_0$, only strictly historical signals ($t \le t_0$)
   and known calendar/promotional schedules ($t > t_0$) may be utilized. Furthermore, neither
   oil prices nor transactions are used as features in the LightGBM forecasting model: same-day
   transactions are simultaneous with sales (leakage), and oil prices introduce external forecasting
   dependencies without improving store-family walk-forward accuracy. Both tables are used
   exclusively in weekly analytical SQL views (`sql/weekly_store_family.sql`).

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
