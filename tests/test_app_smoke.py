"""Headless smoke tests for Streamlit application (app.py)."""

from __future__ import annotations

import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_smoke_default_page():
    """Verify that app.py loads the default page without unhandled exceptions."""
    t0 = time.time()
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    elapsed = time.time() - t0

    # Ensure fast execution (< 10 seconds total in test environment)
    assert elapsed < 10.0, f"App startup exceeded 10 seconds: {elapsed:.2f}s"
    assert not at.exception, f"App raised exceptions: {at.exception}"
    assert len(at.title) > 0


def test_app_smoke_navigation_pages():
    """Verify that all 4 navigation pages render cleanly without exceptions."""
    pages = [
        "Executive Summary",
        "Forecast Explorer",
        "Replenishment Simulator",
        "Business Insights & SQL",
    ]

    for page_name in pages:
        at = AppTest.from_file(APP_PATH, default_timeout=15).run()
        # Radio button navigation
        radio = at.sidebar.radio[0]
        radio.set_value(page_name).run()

        assert not at.exception, f"Page '{page_name}' raised exceptions: {at.exception}"
        assert len(at.title) > 0


def test_app_does_not_reference_raw_data_or_ingest():
    """Verify app.py source code contains no imports of ingest or references to data/raw."""
    src = APP_PATH.read_text(encoding="utf-8")
    assert "data/raw" not in src
    assert "ingest_csvs" not in src
    assert "DATA_RAW" not in src
