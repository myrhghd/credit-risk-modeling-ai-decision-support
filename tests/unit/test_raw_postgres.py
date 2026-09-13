from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from credit_risk.ingestion.raw_postgres import (
    IngestionError,
    LogicalTable,
    ReconciliationResult,
    SourceFile,
    arrow_type_to_postgres,
    compatible_schema,
    create_table_ddl,
    discover_logical_tables,
    initialize_schemas,
    is_managed_raw_table_name,
    load_table_name,
    logical_table_name,
    promote_load_table,
)


def _write_parquet(path: Path, payload: dict[str, list[object]]) -> None:
    table = pa.Table.from_pydict(payload)
    pq.write_table(table, path)


def test_logical_table_name_removes_train_prefix_and_partition_suffix() -> None:
    assert logical_table_name("train_base.parquet") == "base"
    assert logical_table_name("train_other_1.parquet") == "other_1"
    assert logical_table_name("train_static_0_1.parquet") == "static_0"
    assert logical_table_name("train_credit_bureau_a_1_3.parquet") == "credit_bureau_a_1"
    assert logical_table_name("train_credit_bureau_a_2_10.parquet") == "credit_bureau_a_2"


def test_discover_logical_tables_groups_compatible_partitions(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_parquet(raw_dir / "train_static_0_0.parquet", {"case_id": [1], "x": [1.0]})
    _write_parquet(raw_dir / "train_static_0_1.parquet", {"case_id": [2, 3], "x": [2.0, 3.0]})
    _write_parquet(raw_dir / "train_other_1.parquet", {"case_id": [1], "value": ["a"]})

    tables = discover_logical_tables(raw_dir)
    by_name = {table.name: table for table in tables}

    assert sorted(by_name) == ["other_1", "static_0"]
    assert by_name["static_0"].source_row_count == 3
    assert [source.path.name for source in by_name["static_0"].sources] == [
        "train_static_0_0.parquet",
        "train_static_0_1.parquet",
    ]


def test_compatible_schema_rejects_column_mismatch() -> None:
    left = SourceFile(Path("train_x_0.parquet"), "x", 1, pa.schema([("case_id", pa.int64())]))
    right = SourceFile(
        Path("train_x_1.parquet"),
        "x",
        1,
        pa.schema([("case_id", pa.int64()), ("extra", pa.string())]),
    )

    with pytest.raises(IngestionError, match="columns"):
        compatible_schema([left, right])


def test_compatible_schema_rejects_type_mismatch() -> None:
    left = SourceFile(Path("train_x_0.parquet"), "x", 1, pa.schema([("case_id", pa.int64())]))
    right = SourceFile(Path("train_x_1.parquet"), "x", 1, pa.schema([("case_id", pa.string())]))

    with pytest.raises(IngestionError, match="case_id"):
        compatible_schema([left, right])


@pytest.mark.parametrize(
    ("arrow_type", "postgres_type"),
    [
        (pa.bool_(), "boolean"),
        (pa.int16(), "smallint"),
        (pa.int32(), "integer"),
        (pa.int64(), "bigint"),
        (pa.float32(), "real"),
        (pa.float64(), "double precision"),
        (pa.decimal128(12, 2), "numeric(12,2)"),
        (pa.date32(), "date"),
        (pa.timestamp("us"), "timestamp without time zone"),
        (pa.string(), "text"),
    ],
)
def test_arrow_type_to_postgres(arrow_type: pa.DataType, postgres_type: str) -> None:
    assert arrow_type_to_postgres(arrow_type) == postgres_type


def test_reconciliation_result_flags_mismatches() -> None:
    table = LogicalTable(
        "sample",
        (
            SourceFile(
                Path("train_sample.parquet"),
                "sample",
                2,
                pa.schema([("case_id", pa.int64())]),
            ),
        ),
        pa.schema([("case_id", pa.int64())]),
    )
    assert table.source_row_count == 2

    ok = ReconciliationResult("sample", 2, 2, ("case_id",), ("case_id",), ("case_id",))
    bad_rows = ReconciliationResult("sample", 2, 1, ("case_id",), ("case_id",), ("case_id",))
    bad_columns = ReconciliationResult("sample", 2, 2, ("case_id",), ("id",), ("case_id",))

    assert ok.ok
    assert not bad_rows.ok
    assert not bad_columns.ok


def test_unlogged_table_ddl() -> None:
    table = LogicalTable(
        "sample",
        (),
        pa.schema([("case_id", pa.int64()), ("flag", pa.bool_())]),
    )

    ddl = create_table_ddl(table, "__load_sample_abc")

    assert ddl.startswith('CREATE UNLOGGED TABLE raw."__load_sample_abc"')
    assert '"case_id" bigint' in ddl
    assert '"flag" boolean' in ddl


def test_load_table_names_are_managed_and_unique() -> None:
    import uuid

    first = load_table_name("other_1", uuid.UUID("11111111-1111-1111-1111-111111111111"))
    second = load_table_name("other_1", uuid.UUID("22222222-2222-2222-2222-222222222222"))

    assert first == "__load_other_1_111111111111"
    assert second == "__load_other_1_222222222222"
    assert first != second
    assert is_managed_raw_table_name(first)
    assert not is_managed_raw_table_name("other_1")


class _FakeCursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self.conn = conn

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: object | None = None) -> None:
        self.conn.statements.append((sql, params))

    def fetchone(self) -> tuple[bool]:
        return (self.conn.permanent_exists,)


class _FakeTransaction:
    def __enter__(self) -> "_FakeTransaction":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeConn:
    def __init__(self, *, permanent_exists: bool = True) -> None:
        self.permanent_exists = permanent_exists
        self.statements: list[tuple[str, object | None]] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    def commit(self) -> None:
        return None


def test_atomic_promotion_renames_old_table_then_load_table() -> None:
    import uuid

    conn = _FakeConn(permanent_exists=True)

    promote_load_table(
        conn,
        "other_1",
        "__load_other_1_111111111111",
        uuid.UUID("11111111-1111-1111-1111-111111111111"),
    )

    sql = "\n".join(statement for statement, _ in conn.statements)
    assert 'ALTER TABLE raw."other_1"' in sql
    assert 'RENAME TO "__old_other_1_111111111111"' in sql
    assert 'ALTER TABLE raw."__load_other_1_111111111111"' in sql
    assert 'DROP TABLE raw."__old_other_1_111111111111"' in sql


def test_initialize_schemas_declares_run_table_source_table_and_uniqueness() -> None:
    conn = _FakeConn()

    initialize_schemas(conn)

    sql = "\n".join(statement for statement, _ in conn.statements)
    assert "CREATE TABLE IF NOT EXISTS raw.ingestion_run" in sql
    assert "CREATE TABLE IF NOT EXISTS raw.ingestion_logical_table" in sql
    assert "CREATE TABLE IF NOT EXISTS raw.ingestion_source_file" in sql
    assert "PRIMARY KEY (run_id, logical_table, source_file)" in sql
    assert "logical_table_status text NOT NULL" in sql
    assert "source_file_status text NOT NULL" in sql


def test_logical_table_completion_requires_full_reconciliation() -> None:
    missing_key = ReconciliationResult(
        "sample",
        2,
        2,
        ("case_id",),
        ("id",),
        ("case_id",),
    )

    assert not missing_key.ok
    assert not missing_key.key_columns_exist
