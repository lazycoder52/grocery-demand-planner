"""Smoke tests for the project scaffold."""

from __future__ import annotations

import subprocess
import sys

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
    assert cfg["lead_times_days"] == [3, 7, 14]
    assert cfg["service_levels"] == [0.90, 0.95, 0.98]


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
