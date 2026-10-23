"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

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
    return all((DATA_RAW / f).exists() for f in RAW_FILES)


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``data`` when the raw Kaggle files are absent (e.g. in CI)."""
    if raw_data_available():
        return
    skip = pytest.mark.skip(reason="raw Kaggle files not present in data/raw")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
