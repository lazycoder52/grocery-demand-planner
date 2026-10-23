"""Pipeline entry point: ``python -m demand_planner.pipeline``.

Each stage is a callable registered in ``STAGES``. Stages are added by later
pull requests; running with no arguments executes every registered stage in
order.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable

log = logging.getLogger("demand_planner")

StageFn = Callable[[], None]

# Ordered registry of pipeline stages: name -> callable.
STAGES: dict[str, StageFn] = {}


def _register_data_stage() -> None:
    from demand_planner.data import (
        audit_dataset,
        generate_data_card,
        ingest_csvs,
        update_metrics_json,
    )

    def stage_data() -> None:
        con = ingest_csvs()
        audit = audit_dataset(con)
        update_metrics_json(audit)
        generate_data_card(audit)

    STAGES["data"] = stage_data


_register_data_stage()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m demand_planner.pipeline",
        description="Run the grocery demand planning pipeline.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES) or None,
        help="Subset of stages to run (default: all, in order).",
    )
    parser.add_argument("--list", action="store_true", help="List stages and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.list:
        for name in STAGES:
            print(name)
        return 0
    selected = args.stages or list(STAGES)
    if not selected:
        log.info("No stages registered yet.")
        return 0
    for name in selected:
        start = time.perf_counter()
        log.info("stage %s: start", name)
        STAGES[name]()
        log.info("stage %s: done in %.1fs", name, time.perf_counter() - start)
    return 0
