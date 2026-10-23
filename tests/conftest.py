"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.tiny_favorita import generate_tiny_favorita

from demand_planner.config import DATA_RAW

RAW_FILES = (
    "train.csv",
    "test.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
)


def raw_data_available() -> bool:
    """Return True if the real Kaggle dataset files are present in data/raw."""
    return all((DATA_RAW / f).exists() for f in RAW_FILES)


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``data`` when the raw Kaggle files are absent (e.g. in CI)."""
    if raw_data_available():
        return
    skip = pytest.mark.skip(reason="raw Kaggle files not present in data/raw")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_favorita_dir(tmp_path_factory) -> Path:
    """Session-scoped fixture generating a tiny Favorita dataset for fast CI logic tests."""
    out_dir = tmp_path_factory.mktemp("tiny_favorita")
    generate_tiny_favorita(out_dir)
    return out_dir


@pytest.fixture(scope="session")
def tiny_favorita_files(tiny_favorita_dir: Path) -> dict[str, Path]:
    """Dictionary of paths to each generated synthetic CSV in tiny_favorita_dir."""
    return {
        "stores": tiny_favorita_dir / "stores.csv",
        "holidays": tiny_favorita_dir / "holidays_events.csv",
        "oil": tiny_favorita_dir / "oil.csv",
        "transactions": tiny_favorita_dir / "transactions.csv",
        "train": tiny_favorita_dir / "train.csv",
        "test": tiny_favorita_dir / "test.csv",
        "sample_submission": tiny_favorita_dir / "sample_submission.csv",
    }
