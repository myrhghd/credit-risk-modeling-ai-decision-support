"""PostgreSQL raw ingestion for Home Credit Parquet sources."""

from __future__ import annotations

import csv
import io
import os
import re
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

RAW_SCHEMAS = ("raw", "staging", "features", "analytics", "monitoring")
RUN_TABLE = "raw.ingestion_run"
SOURCE_TABLE = "raw.ingestion_source_file"
TABLE_TABLE = "raw.ingestion_logical_table"
DEFAULT_BATCH_SIZE = 50_000
LOAD_TABLE_PREFIX = "__load_"
OLD_TABLE_PREFIX = "__old_"


@dataclass(frozen=True)
class SourceFile:
    path: Path
    logical_table: str
    source_rows: int
    schema: pa.Schema


@dataclass(frozen=True)
class LogicalTable:
    name: str
    sources: tuple[SourceFile, ...]
    schema: pa.Schema

    @property
    def source_row_count(self) -> int:
        return sum(source.source_rows for source in self.sources)

    @property
    def column_names(self) -> list[str]:
        return self.schema.names


@dataclass(frozen=True)
class ReconciliationResult:
    logical_table: str
    source_row_count: int
    loaded_row_count: int
    expected_columns: tuple[str, ...]
    actual_columns: tuple[str, ...]
    key_columns: tuple[str, ...]

    @property
    def row_counts_match(self) -> bool:
        return self.source_row_count == self.loaded_row_count

    @property
    def columns_match(self) -> bool:
        return self.expected_columns == self.actual_columns

    @property
    def key_columns_exist(self) -> bool:
        return all(column in self.actual_columns for column in self.key_columns)

    @property
    def ok(self) -> bool:
        return self.row_counts_match and self.columns_match and self.key_columns_exist


class IngestionError(RuntimeError):
    """Raised when raw ingestion cannot continue safely."""


def logical_table_name(file_name: str) -> str:
    """Map a Home Credit training Parquet file to its consolidated raw table name."""
    stem = Path(file_name).stem
    name = stem.removeprefix("train_")
    return re.sub(r"_(\d+)_(\d+)$", r"_\1", name)


def discover_logical_tables(raw_dir: Path) -> list[LogicalTable]:
    paths = sorted(raw_dir.glob("train_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No training Parquet files found in {raw_dir}")

    sources_by_table: dict[str, list[SourceFile]] = defaultdict(list)
    for path in paths:
        metadata = pq.read_metadata(path)
        schema = metadata.schema.to_arrow_schema()
        table_name = logical_table_name(path.name)
        sources_by_table[table_name].append(
            SourceFile(path, table_name, metadata.num_rows, schema)
        )

    logical_tables: list[LogicalTable] = []
    for table_name, sources in sorted(sources_by_table.items()):
        schema = compatible_schema(sources)
        logical_tables.append(LogicalTable(table_name, tuple(sources), schema))
    return logical_tables


def compatible_schema(sources: Sequence[SourceFile]) -> pa.Schema:
    """Return a shared Arrow schema or raise a clear incompatibility error."""
    if not sources:
        raise IngestionError("Cannot validate an empty source group.")

    reference = sources[0].schema
    for source in sources[1:]:
        if source.schema.names != reference.names:
            raise IngestionError(
                "Partition schema mismatch for "
                f"{source.logical_table}: {sources[0].path.name} columns "
                f"{reference.names} differ from {source.path.name} columns {source.schema.names}"
            )
        for left, right in zip(reference, source.schema, strict=True):
            if left.type != right.type:
                raise IngestionError(
                    "Partition schema mismatch for "
                    f"{source.logical_table}: column {left.name!r} has type "
                    f"{left.type} in {sources[0].path.name} but {right.type} in {source.path.name}"
                )
    return reference


def arrow_type_to_postgres(arrow_type: pa.DataType) -> str:
    """Map an Arrow data type to a PostgreSQL column type."""
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_int8(arrow_type) or pa.types.is_int16(arrow_type):
        return "smallint"
    if pa.types.is_int32(arrow_type):
        return "integer"
    if pa.types.is_int64(arrow_type):
        return "bigint"
    if pa.types.is_uint8(arrow_type) or pa.types.is_uint16(arrow_type):
        return "integer"
    if pa.types.is_uint32(arrow_type):
        return "bigint"
    if pa.types.is_uint64(arrow_type):
        return "numeric(20,0)"
    if pa.types.is_float32(arrow_type):
        return "real"
    if pa.types.is_float64(arrow_type):
        return "double precision"
    if pa.types.is_decimal(arrow_type):
        return f"numeric({arrow_type.precision},{arrow_type.scale})"
    if pa.types.is_date(arrow_type):
        return "date"
    if pa.types.is_timestamp(arrow_type):
        return "timestamp with time zone" if arrow_type.tz else "timestamp without time zone"
    if pa.types.is_time(arrow_type):
        return "time"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "text"
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return "bytea"
    raise IngestionError(f"No PostgreSQL type mapping for Arrow type {arrow_type}")


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def qualified_raw_table(table_name: str) -> str:
    return f"raw.{quote_ident(table_name)}"


def create_table_ddl(table: LogicalTable, table_name: str | None = None) -> str:
    columns = [
        f"{quote_ident(field.name)} {arrow_type_to_postgres(field.type)}"
        for field in table.schema
    ]
    column_sql = ",\n  ".join(columns)
    target_name = table.name if table_name is None else table_name
    return f"CREATE UNLOGGED TABLE {qualified_raw_table(target_name)} (\n  {column_sql}\n);"


def load_table_name(logical_table: str, run_id: uuid.UUID) -> str:
    suffix = run_id.hex[:12]
    return f"{LOAD_TABLE_PREFIX}{logical_table}_{suffix}"


def old_table_name(logical_table: str, run_id: uuid.UUID) -> str:
    suffix = run_id.hex[:12]
    return f"{OLD_TABLE_PREFIX}{logical_table}_{suffix}"


def is_managed_raw_table_name(table_name: str) -> bool:
    return table_name.startswith((LOAD_TABLE_PREFIX, OLD_TABLE_PREFIX))


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def connect_from_env() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("psycopg is required for PostgreSQL ingestion.") from exc

    load_env_file()
    if os.getenv("DATABASE_URL"):
        return psycopg.connect(os.environ["DATABASE_URL"])

    kwargs = {
        "host": os.getenv("POSTGRES_HOST", os.getenv("PGHOST", "localhost")),
        "port": os.getenv("POSTGRES_PORT", os.getenv("PGPORT", "5432")),
        "dbname": os.getenv("POSTGRES_DB", os.getenv("PGDATABASE", "credit_risk")),
        "user": os.getenv("POSTGRES_USER", os.getenv("PGUSER", "credit_risk")),
        "password": os.getenv("POSTGRES_PASSWORD", os.getenv("PGPASSWORD")),
    }
    return psycopg.connect(**{key: value for key, value in kwargs.items() if value})


def initialize_schemas(conn: Any) -> None:
    with conn.cursor() as cur:
        for schema in RAW_SCHEMAS:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)};")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
                run_id uuid PRIMARY KEY,
                started_at timestamptz NOT NULL,
                finished_at timestamptz,
                run_status text NOT NULL,
                message text
            );

            CREATE TABLE IF NOT EXISTS {TABLE_TABLE} (
                run_id uuid NOT NULL REFERENCES {RUN_TABLE}(run_id),
                logical_table text NOT NULL,
                load_table text NOT NULL,
                source_row_count bigint NOT NULL,
                loaded_row_count bigint NOT NULL DEFAULT 0,
                logical_table_status text NOT NULL,
                started_at timestamptz NOT NULL,
                finished_at timestamptz,
                message text,
                PRIMARY KEY (run_id, logical_table)
            );

            CREATE TABLE IF NOT EXISTS {SOURCE_TABLE} (
                run_id uuid NOT NULL REFERENCES {RUN_TABLE}(run_id),
                logical_table text NOT NULL,
                source_file text NOT NULL,
                source_row_count bigint NOT NULL,
                loaded_row_count bigint NOT NULL DEFAULT 0,
                source_file_status text NOT NULL,
                started_at timestamptz NOT NULL,
                finished_at timestamptz,
                message text,
                PRIMARY KEY (run_id, logical_table, source_file),
                FOREIGN KEY (run_id, logical_table)
                    REFERENCES {TABLE_TABLE}(run_id, logical_table)
            );
            """
        )
    conn.commit()


def create_ingestion_run(conn: Any) -> uuid.UUID:
    run_id = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {RUN_TABLE} (run_id, started_at, run_status)
            VALUES (%s, %s, 'running');
            """,
            (run_id, datetime.now(UTC)),
        )
    conn.commit()
    return run_id


def finish_ingestion_run(conn: Any, run_id: uuid.UUID, status: str, message: str | None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {RUN_TABLE}
            SET finished_at = %s, run_status = %s, message = %s
            WHERE run_id = %s;
            """,
            (datetime.now(UTC), status, message, run_id),
        )
    conn.commit()


def ingest_tables(
    conn: Any,
    tables: Sequence[LogicalTable],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[uuid.UUID, list[ReconciliationResult]]:
    run_id = create_ingestion_run(conn)
    results: list[ReconciliationResult] = []
    try:
        for table in tables:
            results.append(ingest_logical_table(conn, table, run_id=run_id, batch_size=batch_size))
    except Exception as exc:
        finish_ingestion_run(conn, run_id, "failed", str(exc))
        raise
    finish_ingestion_run(conn, run_id, "success", None)
    return run_id, results


def ingest_logical_table(
    conn: Any,
    table: LogicalTable,
    *,
    run_id: uuid.UUID | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ReconciliationResult:
    active_run_id = run_id or create_ingestion_run(conn)
    load_name = load_table_name(table.name, active_run_id)
    started_at = datetime.now(UTC)

    try:
        create_load_table(conn, table, load_name)
        register_logical_table(conn, active_run_id, table, load_name, started_at)
        loaded_by_source: dict[str, int] = {}
        for source in table.sources:
            source_started_at = datetime.now(UTC)
            register_source_file(conn, active_run_id, table.name, source, source_started_at)
            loaded_rows = copy_parquet_to_postgres(
                conn,
                table,
                source.path,
                load_name,
                batch_size=batch_size,
            )
            if loaded_rows != source.source_rows:
                raise IngestionError(
                    f"Loaded {loaded_rows} rows from {source.path.name}, "
                    f"expected {source.source_rows}"
                )
            loaded_by_source[source.path.name] = loaded_rows
            finish_source_file(
                conn,
                active_run_id,
                table.name,
                source.path.name,
                loaded_rows,
                "success",
                None,
            )

        result = reconcile_table(conn, table, table_name=load_name)
        if not result.ok:
            raise IngestionError(
                f"Raw ingestion reconciliation failed for raw.{table.name}: {result}"
            )
        promote_load_table(conn, table.name, load_name, active_run_id)
        finish_logical_table(
            conn,
            active_run_id,
            table.name,
            result.loaded_row_count,
            "success",
            None,
        )
        if run_id is None:
            finish_ingestion_run(conn, active_run_id, "success", None)
        return result
    except Exception as exc:
        conn.rollback()
        mark_incomplete_sources_failed(conn, active_run_id, table.name, str(exc))
        mark_logical_table_failed(conn, active_run_id, table.name, str(exc))
        drop_table_if_exists(conn, load_name)
        if run_id is None:
            finish_ingestion_run(conn, active_run_id, "failed", str(exc))
        raise


def create_load_table(conn: Any, table: LogicalTable, load_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {qualified_raw_table(load_name)};")
        cur.execute(create_table_ddl(table, load_name))
    conn.commit()


def register_logical_table(
    conn: Any,
    run_id: uuid.UUID,
    table: LogicalTable,
    load_name: str,
    started_at: datetime,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {TABLE_TABLE}
                (run_id, logical_table, load_table, source_row_count,
                 logical_table_status, started_at)
            VALUES (%s, %s, %s, %s, 'running', %s);
            """,
            (run_id, table.name, load_name, table.source_row_count, started_at),
        )
    conn.commit()


def register_source_file(
    conn: Any,
    run_id: uuid.UUID,
    logical_table: str,
    source: SourceFile,
    started_at: datetime,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {SOURCE_TABLE}
                (run_id, logical_table, source_file, source_row_count,
                 source_file_status, started_at)
            VALUES (%s, %s, %s, %s, 'running', %s);
            """,
            (run_id, logical_table, source.path.name, source.source_rows, started_at),
        )
    conn.commit()


def finish_source_file(
    conn: Any,
    run_id: uuid.UUID,
    logical_table: str,
    source_file: str,
    loaded_rows: int,
    status: str,
    message: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {SOURCE_TABLE}
            SET loaded_row_count = %s,
                source_file_status = %s,
                finished_at = %s,
                message = %s
            WHERE run_id = %s AND logical_table = %s AND source_file = %s;
            """,
            (loaded_rows, status, datetime.now(UTC), message, run_id, logical_table, source_file),
        )
    conn.commit()


def finish_logical_table(
    conn: Any,
    run_id: uuid.UUID,
    logical_table: str,
    loaded_rows: int,
    status: str,
    message: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {TABLE_TABLE}
            SET loaded_row_count = %s,
                logical_table_status = %s,
                finished_at = %s,
                message = %s
            WHERE run_id = %s AND logical_table = %s;
            """,
            (loaded_rows, status, datetime.now(UTC), message, run_id, logical_table),
        )
    conn.commit()


def mark_incomplete_sources_failed(
    conn: Any, run_id: uuid.UUID, logical_table: str, message: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {SOURCE_TABLE}
            SET source_file_status = 'failed',
                finished_at = COALESCE(finished_at, %s),
                message = COALESCE(message, %s)
            WHERE run_id = %s
              AND logical_table = %s
              AND source_file_status <> 'success';
            """,
            (datetime.now(UTC), message, run_id, logical_table),
        )
    conn.commit()


def mark_logical_table_failed(
    conn: Any, run_id: uuid.UUID, logical_table: str, message: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {TABLE_TABLE}
            SET logical_table_status = 'failed',
                finished_at = COALESCE(finished_at, %s),
                message = COALESCE(message, %s)
            WHERE run_id = %s AND logical_table = %s;
            """,
            (datetime.now(UTC), message, run_id, logical_table),
        )
    conn.commit()


def promote_load_table(conn: Any, logical_table: str, load_name: str, run_id: uuid.UUID) -> None:
    old_name = old_table_name(logical_table, run_id)
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT to_regclass(%s) IS NOT NULL;
                """,
                (f"raw.{logical_table}",),
            )
            permanent_exists = bool(cur.fetchone()[0])
            if permanent_exists:
                cur.execute(
                    f"""
                    ALTER TABLE {qualified_raw_table(logical_table)}
                    RENAME TO {quote_ident(old_name)};
                    """
                )
            cur.execute(
                f"""
                ALTER TABLE {qualified_raw_table(load_name)}
                RENAME TO {quote_ident(logical_table)};
                """
            )
            if permanent_exists:
                cur.execute(f"DROP TABLE {qualified_raw_table(old_name)};")


def copy_parquet_to_postgres(
    conn: Any,
    table: LogicalTable,
    path: Path,
    target_table_name: str,
    *,
    batch_size: int,
) -> int:
    columns_sql = ", ".join(quote_ident(name) for name in table.column_names)
    copy_sql = (
        f"COPY {qualified_raw_table(target_table_name)} ({columns_sql}) "
        "FROM STDIN WITH (FORMAT csv, NULL '')"
    )
    loaded_rows = 0
    parquet_file = pq.ParquetFile(path)
    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                write_batch_as_copy_csv(copy, batch, table.column_names)
                loaded_rows += batch.num_rows
    conn.commit()
    return loaded_rows


def write_batch_as_copy_csv(copy: Any, batch: pa.RecordBatch, column_names: Sequence[str]) -> None:
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, lineterminator="\n")
    columns = [
        batch.column(batch.schema.get_field_index(name)).to_pylist()
        for name in column_names
    ]
    for row_values in zip(*columns, strict=True):
        writer.writerow([format_copy_value(value) for value in row_values])
    copy.write(csv_buffer.getvalue())


def format_copy_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return str(value)
    return value


def reconcile_table(
    conn: Any, table: LogicalTable, table_name: str | None = None
) -> ReconciliationResult:
    target_name = table.name if table_name is None else table_name
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {qualified_raw_table(target_name)};")
        loaded_count = int(cur.fetchone()[0])
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'raw' AND table_name = %s
            ORDER BY ordinal_position;
            """,
            (target_name,),
        )
        actual_columns = tuple(row[0] for row in cur.fetchall())

    expected_columns = tuple(table.column_names)
    return ReconciliationResult(
        logical_table=table.name,
        source_row_count=table.source_row_count,
        loaded_row_count=loaded_count,
        expected_columns=expected_columns,
        actual_columns=actual_columns,
        key_columns=tuple(
            column
            for column in ("case_id", "num_group1", "num_group2")
            if column in expected_columns
        ),
    )


def reset_raw_ingestion_state(conn: Any, *, logical_tables: Sequence[str] | None = None) -> None:
    """Explicitly remove raw data/load tables and ingestion metadata."""
    with conn.cursor() as cur:
        if logical_tables:
            for table_name in logical_tables:
                cur.execute(f"DROP TABLE IF EXISTS {qualified_raw_table(table_name)};")
            cur.execute("SELECT to_regclass('raw.ingestion_log') IS NOT NULL;")
            if cur.fetchone()[0]:
                cur.execute(
                    "DELETE FROM raw.ingestion_log WHERE logical_table = ANY(%s);",
                    (list(logical_tables),),
                )
            cur.execute(
                f"DELETE FROM {SOURCE_TABLE} WHERE logical_table = ANY(%s);",
                (list(logical_tables),),
            )
            cur.execute(
                f"DELETE FROM {TABLE_TABLE} WHERE logical_table = ANY(%s);",
                (list(logical_tables),),
            )
            cur.execute(
                f"""
                UPDATE {RUN_TABLE}
                SET message = COALESCE(message, '') || ' reset-table'
                WHERE run_id NOT IN (SELECT DISTINCT run_id FROM {TABLE_TABLE});
                """
            )
        else:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'raw'
                  AND table_type = 'BASE TABLE'
                  AND table_name NOT IN (
                    'ingestion_run',
                    'ingestion_logical_table',
                    'ingestion_source_file',
                    'ingestion_log'
                  );
                """
            )
            for (table_name,) in cur.fetchall():
                cur.execute(f"DROP TABLE IF EXISTS {qualified_raw_table(table_name)};")
            cur.execute(f"TRUNCATE {SOURCE_TABLE}, {TABLE_TABLE}, {RUN_TABLE} CASCADE;")
            cur.execute("SELECT to_regclass('raw.ingestion_log') IS NOT NULL;")
            if cur.fetchone()[0]:
                cur.execute("TRUNCATE raw.ingestion_log;")
    conn.commit()


def cleanup_failed_load_tables(conn: Any) -> list[str]:
    dropped: list[str] = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'raw'
              AND table_type = 'BASE TABLE'
              AND (table_name LIKE '\\_\\_load\\_%' OR table_name LIKE '\\_\\_old\\_%');
            """
        )
        for (table_name,) in cur.fetchall():
            cur.execute(f"DROP TABLE IF EXISTS {qualified_raw_table(table_name)};")
            dropped.append(table_name)
    conn.commit()
    return dropped


def table_is_unlogged(conn: Any, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT relpersistence = 'u'
            FROM pg_class
            JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
            WHERE pg_namespace.nspname = 'raw' AND pg_class.relname = %s;
            """,
            (table_name,),
        )
        row = cur.fetchone()
    return bool(row and row[0])


def drop_table_if_exists(conn: Any, table_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {qualified_raw_table(table_name)};")
    conn.commit()


def filter_tables(
    tables: Iterable[LogicalTable], names: Sequence[str] | None
) -> list[LogicalTable]:
    selected = list(tables)
    if not names:
        return selected
    wanted = set(names)
    missing = wanted - {table.name for table in selected}
    if missing:
        raise IngestionError(f"Unknown logical table(s): {', '.join(sorted(missing))}")
    return [table for table in selected if table.name in wanted]
