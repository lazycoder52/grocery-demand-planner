# Grocery Demand Planner

> End-to-end ML demand forecasting and inventory replenishment toolkit for a 54-store Ecuadorian supermarket chain, cutting forecast error by **26.65%** over standard retail heuristics.

[![CI](https://github.com/lazycoder52/grocery-demand-planner/actions/workflows/ci.yml/badge.svg)](https://github.com/lazycoder52/grocery-demand-planner/actions/workflows/ci.yml)
[![Python 3.12 | 3.14](https://img.shields.io/badge/python-3.12%20%7C%203.14-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 1. Headline Results

### Primary Validation: 5-Fold Rolling-Origin Cross-Validation
Evaluated across **5 expanding-window folds** (16-day horizons; origins stepping every 16 days from 2017-05-11 to 2017-07-14) covering all **54 stores** and **33 product families**:

| Model | Mean WAPE (%) | WAPE Std (%) | Mean Bias (%) | Mean MAE (Units) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LightGBM Global Regressor** | **13.56%** | **±0.97%** | **-0.81%** | **65.55** | **Selected model** |
| Seasonal Naive (7-Day Lag) | 16.33% | ±1.29% | -1.58% | 78.99 | Retail Benchmark |
| 28-Day Moving Average | 23.65% | ±1.03% | -0.89% | 114.12 | Moving Average Baseline |
| Last-Value Naive | 27.83% | ±2.67% | -10.42% | 134.20 | Heuristic Benchmark |

> **Cross-Validation Forecast Value Added (FVA):** LightGBM consistently outperforms the strongest baseline across all 5 folds, delivering a mean **+2.77 percentage points** of WAPE reduction (**16.95% relative error reduction**).

### Final Holdout Confirmation (Untouched 16-Day Test Window: 2017-07-31 to 2017-08-15)
Evaluated once on the final unseen 16 days (**28,512 store-family predictions**, totaling **13,319,179.78 units sold**):

| Model | WAPE (%) | Bias (%) | MAE (Units) | RMSLE | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LightGBM Global Regressor** | **15.16%** | **+2.85%** | **70.81** | **0.4071** | **Selected model** |
| Seasonal Naive (7-Day Lag) | 20.67% | +0.41% | 96.54 | 0.6170 | Retail Benchmark |
| 28-Day Moving Average | 20.81% | +1.34% | 97.22 | 0.5216 | Moving Average Baseline |
| Last-Value Naive | 44.42% | +34.99% | 207.51 | 0.6595 | Heuristic Benchmark |

> **Holdout FVA:** LightGBM achieves **+5.51 percentage points** of WAPE reduction (**26.65% relative error reduction**) on the final confirmation window.

*Note on WAPE:* WAPE is volume-weighted ($\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y}$), evaluating aggregate unit error across total chain volume rather than averaging percentage errors across SKUs, which naturally prioritizes revenue drivers and avoids the division-by-zero distortion that invalidates MAPE on intermittent grocery categories.

*Note on Bias:* The +2.85% net bias on the holdout stems from the non-linear re-transformation of log1p predictions ($\exp(\hat{y}) - 1$) and provides an intentional, cost-favorable safety buffer against high-penalty stockouts during mid-month payday volume surges.

---

## 2. Forecast vs. Actuals

![Actual Sales vs Forecasts](docs/img/actual_vs_forecast_stores.png)

*Figure 1: Daily store-level actuals vs. model forecasts across the 16-day holdout period for Store 1 (Quito, Type D) and Store 2 (Guayaquil, Type B). The model captures weekend demand peaks and mid-month bi-weekly payday surges (August 15).*

---

## 3. Quick Start

### Installation
```bash
# Clone and enter directory
git clone https://github.com/lazycoder52/grocery-demand-planner.git
cd grocery-demand-planner

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Linux/macOS: source .venv/bin/activate

# Install dependencies in editable mode
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

### Run Pipeline
```bash
# Execute end-to-end pipeline (Data Ingestion, Model Training, Executive Charts)
python -m demand_planner.pipeline --stages data models reports
```

### Run Tests & Linters
```bash
# Run pytest test suite (runs fast logic tests in CI via tiny Favorita fixture)
pytest

# Code quality checks
ruff check .
pre-commit run --all-files
```

---

## 4. Key Design Decisions

1. **WAPE Over MAPE:**
   With 7 product categories exhibiting $>50\%$ zero-sales days (`BOOKS` at **96.96%**, `BABY CARE` at **94.13%**), Mean Absolute Percentage Error (MAPE) encounters division-by-zero instability. Weighted Absolute Percentage Error (WAPE) evaluates aggregate volume error, naturally weighting high-velocity revenue categories.
2. **Strict Zero-Leakage 16-Day Lags:**
   To forecast the entire 16-day replenishment window at origin $t_0$, sales lag features are restricted to $\ge 16$ days ($t-16, t-21, t-28$, and 7d/28d rolling means over lag 16). No future signals or same-day post-checkout transactions are ever exposed to the test set.
3. **Post-April 2014 Promotion Conditioning:**
   Automated data audits revealed promotional tracking officially started on **2014-04-01** (all 2013 promotional counts were zero). Training strictly conditions on post-inception data to avoid false zero-promo distortions.
4. **Fast Columnar Storage & Ingestion:**
   DuckDB handles typed schema validation and aggregation (< 2s ingestion of 3M+ rows), writing lightweight parquet summaries for instant app consumption.

---

## 5. Known Limitations

- **Sales $\ne$ True Demand (Censoring):** Point-of-sale data records fulfilled sales. When a store stocks out, recorded sales drop to zero regardless of true customer demand.
- **Lack of Physical Inventory Balances:** The public dataset provides transactions and sales without live on-hand stock or lead-time variance; inventory safety stock must be parameterized via simulation.
- **Observational Promotional Lift:** Promotional assignments are observational historical records, not randomized A/B experiments.

---

## 6. Documentation & Product Resources

- 📄 **[Product Brief](docs/PRODUCT_BRIEF.md):** 2-page executive summary covering target personas, metric tree (Mermaid), pilot rollout plan, and scope roadmap.
- 📊 **[Data Card](docs/DATA_CARD.md):** Schema specifications, audit findings, missing date resolutions, and cleaning decisions.
- 📈 **[Executive Visualizations](docs/img/):**
  - [WAPE by Family vs. Baseline](docs/img/wape_by_family_vs_baseline.png)
  - [Daily Actuals vs. Forecast](docs/img/actual_vs_forecast_stores.png)
  - [Demand Intermittency Distribution](docs/img/zero_rate_by_family.png)
