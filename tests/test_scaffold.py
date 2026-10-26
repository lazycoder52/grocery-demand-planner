"""Smoke tests for project scaffold, dependencies, and test fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from demand_planner.config import CONFIG_DIR, ROOT, load_config, rel
from demand_planner.pipeline import main


def test_root_contains_pyproject():
    assert (ROOT / "pyproject.toml").exists()
    assert CONFIG_DIR.is_dir()


def test_model_config_has_seed_and_horizon():
    cfg = load_config("model")
    assert isinstance(cfg["seed"], int)
    assert cfg["evaluation"]["horizon_days"] == 16
    assert cfg["evaluation"]["n_folds"] >= 5


def test_sim_config_matches_plan():
    cfg = load_config("sim_config")
    assert cfg["review_period_days"] == 7
    assert cfg["lead_times_days"] == [3, 7]
    assert cfg["service_levels"] == [0.90, 0.95, 0.98]
    assert cfg["target_fill_rates"] == [98.5, 99.0, 99.5]


def test_rel_returns_repo_relative_posix_path():
    assert rel(ROOT / "config" / "model.yaml") == "config/model.yaml"


def test_pipeline_main_runs():
    assert main(["--list"]) == 0


def test_pipeline_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "demand_planner.pipeline", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "demand planning pipeline" in result.stdout


def test_core_dependencies_importable():
    """Verify in CI and locally that all required packages install cleanly and are importable."""
    from importlib.metadata import version

    import duckdb
    import lightgbm
    import pre_commit  # noqa: F401
    import streamlit

    assert duckdb.__version__
    assert lightgbm.__version__
    assert streamlit.__version__
    assert version("pre-commit")


def test_tiny_favorita_fixture_characteristics(
    tiny_favorita_dir: Path, tiny_favorita_files: dict[str, Path]
):
    """Verify tiny_favorita fixture matches all required Favorita structures and quirks."""
    stores_df = pd.read_csv(tiny_favorita_files["stores"])
    train_df = pd.read_csv(tiny_favorita_files["train"])
    holidays_df = pd.read_csv(tiny_favorita_files["holidays"])
    oil_df = pd.read_csv(tiny_favorita_files["oil"])
    trans_df = pd.read_csv(tiny_favorita_files["transactions"])
    test_df = pd.read_csv(tiny_favorita_files["test"])

    # 1. 5 stores and 6 families
    assert len(stores_df) == 5
    assert set(train_df["store_nbr"].unique()) == {1, 2, 3, 4, 5}
    assert len(train_df["family"].unique()) == 6

    # 2. ~400 days in train (excluding Dec 25)
    unique_dates = train_df["date"].unique()
    assert 390 <= len(unique_dates) <= 415

    # 3. Dec 25 missing (Christmas store closure)
    assert not any("-12-25" in d for d in unique_dates)
    assert not any("-12-25" in d for d in trans_df["date"].unique())

    # 4. Late opening store (store 5 has 0 sales before its opening date)
    store_5 = train_df[train_df["store_nbr"] == 5]
    first_date = store_5["date"].min()
    assert (store_5[store_5["date"] == first_date]["sales"] == 0.0).all()
    # But later in the series, it has positive sales
    last_date = store_5["date"].max()
    assert (store_5[store_5["date"] == last_date]["sales"] > 0.0).any()

    # 5. Promotions present and non-zero
    assert (train_df["onpromotion"] > 0).any()

    # 6. Transferred holiday present
    transferred = holidays_df[holidays_df["transferred"] == True]  # noqa: E712
    assert len(transferred) >= 1
    transfer_type = holidays_df[holidays_df["type"] == "Transfer"]
    assert len(transfer_type) >= 1

    # 7. Oil prices have values and NaNs (weekends)
    assert oil_df["dcoilwtico"].notna().any()
    assert oil_df["dcoilwtico"].isna().any()

    # 8. Test set has 16 days
    assert len(test_df["date"].unique()) == 16
