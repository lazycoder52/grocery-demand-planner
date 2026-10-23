"""Rolling-origin backtests, metrics, and heuristic baselines."""

from demand_planner.evaluation.backtest import (
    FoldSplit,
    assert_no_temporal_leakage,
    evaluate_baselines_on_split,
    generate_rolling_folds,
    run_baseline_benchmarks,
)
from demand_planner.evaluation.baselines import (
    predict_moving_average,
    predict_naive,
    predict_seasonal_naive,
)
from demand_planner.evaluation.metrics import (
    calc_bias,
    calc_mae,
    calc_rmsle,
    calc_wape,
    evaluate_predictions,
)

__all__ = [
    "FoldSplit",
    "assert_no_temporal_leakage",
    "calc_bias",
    "calc_mae",
    "calc_rmsle",
    "calc_wape",
    "evaluate_baselines_on_split",
    "evaluate_predictions",
    "generate_rolling_folds",
    "predict_moving_average",
    "predict_naive",
    "predict_seasonal_naive",
    "run_baseline_benchmarks",
]
