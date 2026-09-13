"""CLI for PostgreSQL raw schema initialization, ingestion, and validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.ingestion.raw_postgres import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    cleanup_failed_load_tables,
    connect_from_env,
    discover_logical_tables,
    filter_tables,
    ingest_tables,
    initialize_schemas,
    reconcile_table,
    reset_raw_ingestion_state,
    table_is_unlogged,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("init-schemas", "list-tables", "ingest", "validate", "reset", "cleanup-loads"),
        help="Operation to run.",
    )
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--table", action="append", dest="tables", help="Logical table to process.")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = discover_logical_tables(args.raw_dir)

    if args.command == "list-tables":
        for table in tables:
            sources = ", ".join(source.path.name for source in table.sources)
            print(f"{table.name}: {table.source_row_count:,} rows from {sources}")
        return

    conn = connect_from_env()
    try:
        initialize_schemas(conn)

        selected = filter_tables(tables, args.tables)
        if args.command == "init-schemas":
            print("Initialized schemas: raw, staging, features, analytics, monitoring")
            return

        if args.command == "reset":
            reset_raw_ingestion_state(conn, logical_tables=args.tables)
            target = ", ".join(args.tables) if args.tables else "all raw ingestion state"
            print(f"Reset {target}")
            return

        if args.command == "cleanup-loads":
            dropped = cleanup_failed_load_tables(conn)
            print(f"Dropped {len(dropped)} failed load/old table(s)")
            return

        if args.command == "ingest":
            run_id, results = ingest_tables(conn, selected, batch_size=args.batch_size)
            print(f"run_id={run_id}")
            for result in results:
                unlogged = table_is_unlogged(conn, result.logical_table)
                print(
                    f"raw.{result.logical_table}: "
                    f"{result.loaded_row_count:,}/{result.source_row_count:,} rows loaded; "
                    f"unlogged={str(unlogged).lower()}"
                )
            return

        if args.command == "validate":
            for table in selected:
                result = reconcile_table(conn, table)
                status = "ok" if result.ok else "failed"
                unlogged = table_is_unlogged(conn, result.logical_table)
                print(
                    f"raw.{result.logical_table}: {status}; "
                    f"source={result.source_row_count:,}; loaded={result.loaded_row_count:,}; "
                    f"columns={len(result.actual_columns)}/{len(result.expected_columns)}; "
                    f"unlogged={str(unlogged).lower()}"
                )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
