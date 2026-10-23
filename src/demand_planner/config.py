"""Project paths and YAML configuration loading.

All paths are resolved relative to the repository root so that no absolute local
path is ever written to a committed file.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml


def _find_root() -> Path:
    env = os.environ.get("DEMAND_PLANNER_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "config").is_dir():
            return parent
    return Path.cwd().resolve()


ROOT = _find_root()
CONFIG_DIR = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
OUTPUTS = ROOT / "outputs"
METRICS_DIR = ROOT / "metrics"
METRICS_FILE = METRICS_DIR / "metrics.json"
DOCS = ROOT / "docs"
SQL_DIR = ROOT / "sql"


@cache
def load_config(name: str) -> dict[str, Any]:
    """Load ``config/<name>.yaml`` as a dict."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def rel(path: Path) -> str:
    """Return ``path`` relative to the repository root, with forward slashes."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name
