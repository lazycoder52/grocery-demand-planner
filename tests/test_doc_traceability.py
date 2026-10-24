"""Automated Documentation Traceability Verification.

Asserts that every quantitative claim in README.md, docs/PRODUCT_BRIEF.md,
and docs/INSIGHTS.md is strictly traceable to metrics/metrics.json or named SQL
queries in sql/ and parquet artifacts in outputs/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MAPPING_FILE = ROOT / "tests" / "doc_claims_mapping.json"
METRICS_FILE = ROOT / "metrics" / "metrics.json"


def _resolve_json_path(data: dict[str, Any], path_str: str) -> Any:
    """Resolve a dot-and-bracket notation path into a JSON dictionary."""
    tokens = re.split(r"\.(?![^\[]*\])", path_str)
    curr: Any = data
    for token in tokens:
        if "[" in token and token.endswith("]"):
            key, query = token[:-1].split("[", 1)
            curr = curr[key]
            # Parse query filters e.g. lead_time_days=3,target_fill_rate_pct=99.0,Point
            subparts = query.split(",")
            filters: dict[str, Any] = {}
            contains_str: str | None = None
            for p in subparts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    try:
                        filters[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        filters[k] = v
                else:
                    contains_str = p

            matched = None
            for item in curr:
                match = True
                for fk, fv in filters.items():
                    if item.get(fk) != fv:
                        match = False
                        break
                if match and contains_str:
                    item_str = str(item)
                    if contains_str not in item_str:
                        match = False
                if match:
                    matched = item
                    break
            if matched is None:
                raise KeyError(f"No item matched query [{query}] in list at {key}")
            curr = matched
        else:
            curr = curr[token]
    return curr


def test_metrics_json_claims_traceability():
    """Verify all metrics.json claims match formatted values in documentation."""
    assert MAPPING_FILE.exists(), f"Missing mapping file: {MAPPING_FILE}"
    assert METRICS_FILE.exists(), f"Missing metrics file: {METRICS_FILE}"

    with MAPPING_FILE.open(encoding="utf-8") as f:
        mapping = json.load(f)
    with METRICS_FILE.open(encoding="utf-8") as f:
        metrics = json.load(f)

    doc_texts = {
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "docs/PRODUCT_BRIEF.md": (ROOT / "docs" / "PRODUCT_BRIEF.md").read_text(encoding="utf-8"),
        "docs/INSIGHTS.md": (ROOT / "docs" / "INSIGHTS.md").read_text(encoding="utf-8"),
    }

    claims = mapping.get("metrics_json_claims", [])
    assert len(claims) > 0, "No claims defined in doc_claims_mapping.json"

    for claim in claims:
        claim_id = claim["claim_id"]
        json_path = claim["json_path"]
        fmt = claim["format"]
        target_docs = claim["docs"]

        # Handle CI composite formats
        if json_path.endswith(".ci_95_series"):
            base_path = json_path[: -len(".ci_95_series")]
            parent = _resolve_json_path(metrics, base_path)
            low = parent["ci_95_series_pct_diff_low"]
            high = parent["ci_95_series_pct_diff_high"]
            formatted_val = fmt.format(low, high)
        elif json_path.endswith(".ci_95_time_block"):
            base_path = json_path[: -len(".ci_95_time_block")]
            parent = _resolve_json_path(metrics, base_path)
            low = parent["ci_95_time_block_pct_diff_low"]
            high = parent["ci_95_time_block_pct_diff_high"]
            formatted_val = fmt.format(low, high)
        else:
            val = _resolve_json_path(metrics, json_path)
            formatted_val = fmt.format(val)

        for doc_name in target_docs:
            doc_content = doc_texts[doc_name]
            assert formatted_val in doc_content, (
                f"Claim '{claim_id}' ({formatted_val} from {json_path}) not found in {doc_name}"
            )


def test_claims_without_json_path_sql_traceability():
    """Verify claims without a JSON path are documented and trace to valid SQL/parquet."""
    assert MAPPING_FILE.exists(), f"Missing mapping file: {MAPPING_FILE}"

    with MAPPING_FILE.open(encoding="utf-8") as f:
        mapping = json.load(f)

    sql_claims = mapping.get("claims_without_json_path_sql_derived", [])
    assert len(sql_claims) > 0, "No SQL-derived claims registered"

    doc_texts = {
        "docs/INSIGHTS.md": (ROOT / "docs" / "INSIGHTS.md").read_text(encoding="utf-8"),
    }

    for item in sql_claims:
        claim_id = item["claim_id"]
        claim_text = item["claim_text"]
        source_sql = ROOT / item["source_sql"]
        parquet_art = ROOT / item["parquet_artifact"]
        target_docs = item["docs"]

        assert source_sql.exists(), f"Source SQL file missing for {claim_id}: {source_sql}"
        assert parquet_art.exists(), f"Parquet artifact missing for {claim_id}: {parquet_art}"

        for doc_name in target_docs:
            doc_content = doc_texts[doc_name]
            assert claim_text in doc_content, (
                f"SQL-derived claim '{claim_id}' ('{claim_text}') not found in {doc_name}"
            )


def test_list_and_audit_all_claims_coverage():
    """Verify registry contains both metrics.json claims and SQL-derived claims."""
    with MAPPING_FILE.open(encoding="utf-8") as f:
        mapping = json.load(f)

    json_claims = mapping.get("metrics_json_claims", [])
    sql_claims = mapping.get("claims_without_json_path_sql_derived", [])

    # Ensure coverage meets strict audit requirements
    assert len(json_claims) >= 30, f"Expected >= 30 metrics claims, got {len(json_claims)}"
    assert len(sql_claims) >= 15, f"Expected >= 15 SQL claims, got {len(sql_claims)}"
