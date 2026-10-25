# Outputs Directory

This directory stores generated artifacts, forecasts, and simulation results.

## Regeneration Commands

Large prediction parquet files (>1 MB, such as `holdout_predictions.parquet` and `oof_and_holdout_predictions.parquet`) are excluded from git tracking to prevent repository bloat, enforce the 5 MB CI limit on `outputs/`, and maintain fast CI performance. Only small summary aggregates (`*_summary.parquet`) are committed.

To regenerate all out-of-fold predictions, holdout predictions, and replenishment simulation files:

```bash
# Regenerate feature engineering, LightGBM models, and out-of-fold predictions
python -m demand_planner.pipeline --stages models

# Regenerate replenishment policy simulations and trade-off visualizations
python -m demand_planner.pipeline --stages simulation
```

## Generated Artifacts

- `oof_and_holdout_predictions.parquet`: Full 171,072-row matrix containing point (`pred_lgbm`, `pred_snaive`) and quantile (`pred_p10`, `pred_p50`, `pred_p90`) forecasts across all 5 rolling-origin CV folds and the holdout period.
- `holdout_predictions.parquet`: 28,512-row table of predictions for the final 16-day confirmation holdout window (`2017-07-31` to `2017-08-15`).
- `sim_policy_summary.parquet`: Tracked aggregate table of fill rate, inventory, stockout rate, and service level across replenishment policies.
