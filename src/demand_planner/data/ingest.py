"""DuckDB ingestion module for raw Corporación Favorita CSV files."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from demand_planner.config import DATA_INTERIM, DATA_RAW

log = logging.getLogger("demand_planner.data.ingest")

TABLE_SCHEMAS = {
    "stores": """
        CREATE OR REPLACE TABLE stores (
            store_nbr INTEGER PRIMARY KEY,
            city VARCHAR NOT NULL,
            state VARCHAR NOT NULL,
            type VARCHAR NOT NULL,
            cluster INTEGER NOT NULL
        )
    """,
    "oil": """
        CREATE OR REPLACE TABLE oil (
            date DATE PRIMARY KEY,
            dcoilwtico DOUBLE
        )
    """,
    "holidays_events": """
        CREATE OR REPLACE TABLE holidays_events (
            date DATE NOT NULL,
            type VARCHAR NOT NULL,
            locale VARCHAR NOT NULL,
            locale_name VARCHAR NOT NULL,
            description VARCHAR NOT NULL,
            transferred BOOLEAN NOT NULL
        )
    """,
    "transactions": """
        CREATE OR REPLACE TABLE transactions (
            date DATE NOT NULL,
            store_nbr INTEGER NOT NULL,
            transactions INTEGER NOT NULL,
            PRIMARY KEY (date, store_nbr)
        )
    """,
    "train": """
        CREATE OR REPLACE TABLE train (
            id BIGINT PRIMARY KEY,
            date DATE NOT NULL,
            store_nbr INTEGER NOT NULL,
            family VARCHAR NOT NULL,
            sales DOUBLE NOT NULL,
            onpromotion INTEGER NOT NULL
        )
    """,
    "test": """
        CREATE OR REPLACE TABLE test (
            id BIGINT PRIMARY KEY,
            date DATE NOT NULL,
            store_nbr INTEGER NOT NULL,
            family VARCHAR NOT NULL,
            onpromotion INTEGER NOT NULL
        )
    """,
    "sample_submission": """
        CREATE OR REPLACE TABLE sample_submission (
            id BIGINT PRIMARY KEY,
            sales DOUBLE NOT NULL
        )
    """,
}


def get_duckdb_connection(
    db_path: Path | str | None = None,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Create or connect to a DuckDB database. Defaults to an in-memory database."""
    if db_path is None:
        return duckdb.connect(":memory:")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def ingest_csvs(
    raw_dir: Path | str = DATA_RAW,
    con: duckdb.DuckDBPyConnection | None = None,
    db_path: Path | str | None = None,
) -> duckdb.DuckDBPyConnection:
    """Ingest raw CSV files from ``raw_dir`` into DuckDB tables.

    If ``con`` is None, creates a connection to ``db_path`` (or in-memory if db_path is None).
    Returns the connection with all available tables populated.
    """
    raw_path = Path(raw_dir)
    if con is None:
        con = get_duckdb_connection(db_path)

    for table_name, create_sql in TABLE_SCHEMAS.items():
        csv_file = raw_path / f"{table_name}.csv"
        if not csv_file.exists():
            log.warning("CSV file not found, skipping table %s: %s", table_name, csv_file)
            continue

        con.execute(create_sql)
        csv_str = str(csv_file).replace("\\", "/")
        insert_sql = f"""
            INSERT INTO {table_name}
            SELECT * FROM read_csv('{csv_str}', header=True, auto_detect=True)
        """
        con.execute(insert_sql)
        row_count = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
        log.info("Ingested table %s: %d rows from %s", table_name, row_count, csv_file.name)

    return con


def get_default_db(raw_dir: Path | str = DATA_RAW) -> duckdb.DuckDBPyConnection:
    """Get or build the default interim DuckDB database (data/interim/favorita.duckdb)."""
    db_file = DATA_INTERIM / "favorita.duckdb"
    return ingest_csvs(raw_dir=raw_dir, db_path=db_file)
