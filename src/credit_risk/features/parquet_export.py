"""Disk-efficient applicant feature export to compressed Parquet."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from credit_risk.features.applicant_features import (
    STATIC_TABLES,
    FeatureSpec,
    build_repeated_cte,
    sanitize_feature_part,
)
from credit_risk.features.feature_selection import load_registry
from credit_risk.ingestion.raw_postgres import connect_from_env, quote_ident

BASE_COLUMNS = ("case_id", "target", "date_decision", "WEEK_NUM", "MONTH")
FAMILY_SLUGS: Mapping[str, str] = {
    "static applicant/bureau snapshots": "static_applicant_bureau",
    "previous applications": "previous_applications",
    "credit bureau histories": "credit_bureau_histories",
    "person/participant histories": "person_participant_histories",
    "debit/deposit/other histories": "debit_deposit_other_histories",
    "tax registry histories": "tax_registry_histories",
}
DEFAULT_EXPORT_DIR = Path("data/processed/feature_families")
DEFAULT_METADATA_PATH = Path("artifacts/features/parquet_export_manifest.json")
DEFAULT_COMPRESSION = "zstd"
TEMP_SUFFIX = ".tmp"


@dataclass(frozen=True)
class FamilyExportResult:
    family: str
    row_count: int
    column_count: int
    output_path: str
    file_size_bytes: int
    completion_status: str
    timestamp: str
    message: str | None = None


def load_kept_specs(
    registry_path: Path = Path("artifacts/features/applicant_feature_registry.json"),
    selection_manifest_path: Path = Path(
        "artifacts/features/applicant_feature_selection_manifest.json"
    ),
) -> list[FeatureSpec]:
    registry = {spec.feature_name: spec for spec in load_registry(registry_path)}
    decisions = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    kept_names = [item["feature_name"] for item in decisions if item["decision"] == "keep"]
    return [registry[name] for name in sorted(kept_names)]


def load_source_catalog_columns(
    path: Path = Path("artifacts/source_catalog/feature_catalog.json"),
) -> dict[tuple[str, str], Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (column["logical_table"], column["column_name"]): column
        for column in payload["columns"]
    }


def group_specs_by_family(specs: Sequence[FeatureSpec]) -> dict[str, list[FeatureSpec]]:
    grouped: dict[str, list[FeatureSpec]] = defaultdict(list)
    for spec in specs:
        grouped[spec.feature_family].append(spec)
    return {
        family: sorted(items, key=lambda spec: spec.feature_name)
        for family, items in grouped.items()
    }


def family_slug(family: str) -> str:
    return FAMILY_SLUGS[family]


def base_query(smoke_limit: int | None = None) -> str:
    limit_sql = f"\nLIMIT {smoke_limit}" if smoke_limit is not None else ""
    return f"""
SELECT case_id, target, date_decision, "WEEK_NUM", "MONTH"
FROM raw.base
ORDER BY case_id{limit_sql};
"""


def build_family_query(
    family: str,
    specs: Sequence[FeatureSpec],
    *,
    smoke_limit: int | None = None,
) -> str:
    specs_by_table: dict[str, list[FeatureSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_table[spec.source_table].append(spec)

    base_limit = f"\n    LIMIT {smoke_limit}" if smoke_limit is not None else ""
    ctes = [
        f"""base_cases AS (
    SELECT case_id, date_decision
    FROM raw.base
    ORDER BY case_id{base_limit}
)"""
    ]
    joins = []
    select_columns = ["b.case_id"]
    for table_name in sorted(specs_by_table):
        table_specs = specs_by_table[table_name]
        alias = sanitize_feature_part(table_name)
        source_sql = f"raw.{quote_ident(table_name)}"
        if smoke_limit is not None:
            source_sql = f"(SELECT * FROM raw.{quote_ident(table_name)} LIMIT {smoke_limit * 20})"
        if table_name in STATIC_TABLES:
            cte_alias = f"{alias}_one"
            ctes.append(build_static_cte(table_name, cte_alias, table_specs, source_sql))
            joins.append(f"LEFT JOIN {quote_ident(cte_alias)} USING (case_id)")
            select_columns.extend(
                f"{quote_ident(cte_alias)}.{quote_ident(spec.feature_name)}" for spec in table_specs
            )
        else:
            ctes.append(build_repeated_cte(table_name, alias, table_specs, source_sql))
            joins.append(f"LEFT JOIN {quote_ident(alias)} USING (case_id)")
            select_columns.extend(
                f"{quote_ident(alias)}.{quote_ident(spec.feature_name)}" for spec in table_specs
            )

    cte_sql = ",\n".join(ctes)
    select_sql = ",\n    ".join(select_columns)
    join_sql = "\n".join(joins)
    return f"""WITH
{cte_sql}
SELECT
    {select_sql}
FROM base_cases b
{join_sql}
ORDER BY b.case_id;
"""


def build_static_cte(
    table_name: str,
    alias: str,
    specs: Sequence[FeatureSpec],
    source_sql: str,
) -> str:
    expressions = ",\n        ".join(
        f"{spec.sql_expression} AS {quote_ident(spec.feature_name)}" for spec in specs
    )
    return f"""{quote_ident(alias)} AS (
    SELECT DISTINCT ON ({quote_ident(table_name)}.case_id)
        {quote_ident(table_name)}.case_id,
        {expressions}
    FROM {source_sql} AS {quote_ident(table_name)}
    JOIN base_cases b USING (case_id)
    ORDER BY {quote_ident(table_name)}.case_id
)"""


def arrow_type_from_postgres_type(postgres_type: str) -> pa.DataType:
    normalized = postgres_type.lower()
    if normalized in {"smallint", "integer", "bigint"}:
        return pa.int64()
    if normalized in {"real", "double precision"} or normalized.startswith("numeric"):
        return pa.float64()
    if normalized == "boolean":
        return pa.bool_()
    return pa.string()


def base_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            ("case_id", pa.int64()),
            ("target", pa.int64()),
            ("date_decision", pa.string()),
            ("WEEK_NUM", pa.int64()),
            ("MONTH", pa.int64()),
        ]
    )


def family_arrow_schema(
    specs: Sequence[FeatureSpec],
    source_catalog: Mapping[tuple[str, str], Mapping[str, Any]],
) -> pa.Schema:
    fields = [pa.field("case_id", pa.int64())]
    for spec in sorted(specs, key=lambda item: item.feature_name):
        if spec.transformation in {
            "count",
            "min",
            "mean",
            "max",
            "std",
            "sum",
            "n_distinct",
            "min_days_before_decision",
            "mean_days_before_decision",
            "max_days_before_decision",
        }:
            fields.append(pa.field(spec.feature_name, pa.float64()))
        elif spec.transformation == "is_missing":
            fields.append(pa.field(spec.feature_name, pa.int64()))
        else:
            source = source_catalog.get((spec.source_table, spec.source_columns[0]), {})
            fields.append(
                pa.field(
                    spec.feature_name,
                    arrow_type_from_postgres_type(str(source.get("postgres_type", "text"))),
                )
            )
    return pa.schema(fields)


def export_query_to_parquet(
    conn: Any,
    query: str,
    output_path: Path,
    *,
    schema: pa.Schema,
    batch_size: int = 50_000,
    compression: str = DEFAULT_COMPRESSION,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + TEMP_SUFFIX)
    row_count = 0
    try:
        if temp_path.exists():
            temp_path.unlink()
        writer = pq.ParquetWriter(temp_path, schema, compression=compression)
        with conn.cursor(name=f"feature_export_{datetime.now(UTC).timestamp():.0f}") as cur:
            cur.execute(query)
            column_names = [column.name for column in cur.description]
            if column_names != schema.names:
                raise ValueError(
                    f"Query columns {column_names} do not match expected schema {schema.names}"
                )
            while rows := cur.fetchmany(batch_size):
                writer.write_table(rows_to_table(rows, schema))
                row_count += len(rows)
        writer.close()
        writer = None
        temp_path.replace(output_path)
    finally:
        if "writer" in locals() and writer is not None:
            writer.close()
        if temp_path.exists():
            temp_path.unlink()
    return row_count, len(schema)


def rows_to_table(rows: Sequence[Sequence[Any]], schema: pa.Schema) -> pa.Table:
    arrays = []
    for index, field in enumerate(schema):
        arrays.append(pa.array([row[index] for row in rows], type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def write_row_batches_to_parquet(
    row_batches: Sequence[Sequence[Sequence[Any]]],
    schema: pa.Schema,
    output_path: Path,
    *,
    compression: str = DEFAULT_COMPRESSION,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + TEMP_SUFFIX)
    row_count = 0
    writer = None
    try:
        if temp_path.exists():
            temp_path.unlink()
        writer = pq.ParquetWriter(temp_path, schema, compression=compression)
        for rows in row_batches:
            writer.write_table(rows_to_table(rows, schema))
            row_count += len(rows)
        writer.close()
        writer = None
        temp_path.replace(output_path)
    finally:
        if writer is not None:
            writer.close()
        if temp_path.exists():
            temp_path.unlink()
    return row_count, len(schema)


def output_path_for_family(output_dir: Path, family: str, *, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return output_dir / f"{family_slug(family)}{suffix}.parquet"


def read_export_manifest(path: Path = DEFAULT_METADATA_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_export_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")


def completed_families(manifest_rows: Sequence[Mapping[str, Any]], *, smoke: bool) -> set[str]:
    return {
        str(row["family"])
        for row in manifest_rows
        if row["completion_status"] == "success" and bool(row.get("smoke")) == smoke
    }


def export_feature_families(
    *,
    output_dir: Path = DEFAULT_EXPORT_DIR,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    smoke_limit: int | None = 1_000,
    batch_size: int = 50_000,
    compression: str = DEFAULT_COMPRESSION,
    force: bool = False,
    families: Sequence[str] | None = None,
) -> dict[str, Any]:
    specs = load_kept_specs()
    source_catalog = load_source_catalog_columns()
    grouped = group_specs_by_family(specs)
    if families:
        requested = set(families)
        missing = requested - set(grouped)
        if missing:
            raise ValueError(f"Unknown feature family: {', '.join(sorted(missing))}")
        grouped = {family: items for family, items in grouped.items() if family in requested}
    smoke = smoke_limit is not None
    manifest_rows = read_export_manifest(metadata_path)
    done = set() if force else completed_families(manifest_rows, smoke=smoke)
    results: list[FamilyExportResult] = []

    conn = connect_from_env()
    try:
        base_output = output_dir / ("base_labels_smoke.parquet" if smoke else "base_labels.parquet")
        if force or not base_output.exists():
            rows, columns = export_query_to_parquet(
                conn,
                base_query(smoke_limit),
                base_output,
                schema=base_arrow_schema(),
                batch_size=batch_size,
                compression=compression,
            )
            results.append(
                FamilyExportResult(
                    "base labels",
                    rows,
                    columns,
                    str(base_output),
                    base_output.stat().st_size,
                    "success",
                    datetime.now(UTC).isoformat(),
                )
            )
        for family in sorted(grouped):
            if family in done:
                continue
            output_path = output_path_for_family(output_dir, family, smoke=smoke)
            if output_path.exists() and not force:
                done.add(family)
                continue
            query = build_family_query(family, grouped[family], smoke_limit=smoke_limit)
            rows, columns = export_query_to_parquet(
                conn,
                query,
                output_path,
                schema=family_arrow_schema(grouped[family], source_catalog),
                batch_size=batch_size,
                compression=compression,
            )
            results.append(
                FamilyExportResult(
                    family,
                    rows,
                    columns,
                    str(output_path),
                    output_path.stat().st_size,
                    "success",
                    datetime.now(UTC).isoformat(),
                )
            )
    finally:
        conn.close()

    existing = [row for row in manifest_rows if not (force and bool(row.get("smoke")) == smoke)]
    new_rows = [{**asdict(result), "smoke": smoke} for result in results]
    write_export_manifest(metadata_path, [*existing, *new_rows])
    return {
        "feature_counts_by_family": dict(Counter(spec.feature_family for spec in specs)),
        "exports": new_rows,
        "skipped_families": sorted(done),
    }


def join_family_parquets(
    *,
    output_dir: Path = DEFAULT_EXPORT_DIR,
    output_path: Path = Path("data/processed/modeling_features_smoke.parquet"),
    smoke: bool = True,
    compression: str = DEFAULT_COMPRESSION,
) -> dict[str, Any]:
    base_path = output_dir / ("base_labels_smoke.parquet" if smoke else "base_labels.parquet")
    frame = pl.scan_parquet(base_path)
    for family in FAMILY_SLUGS:
        family_path = output_path_for_family(output_dir, family, smoke=smoke)
        if family_path.exists():
            frame = frame.join(pl.scan_parquet(family_path), on="case_id", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.sink_parquet(output_path, compression=compression)
    metadata = pq.read_metadata(output_path)
    return {
        "output_path": str(output_path),
        "row_count": metadata.num_rows,
        "column_count": metadata.num_columns,
        "file_size_bytes": output_path.stat().st_size,
    }


def validate_joined_parquet(
    output_path: Path,
    manifest_path: Path = Path("artifacts/features/applicant_feature_selection_manifest.json"),
) -> dict[str, Any]:
    metadata = pq.read_metadata(output_path)
    schema_names = metadata.schema.to_arrow_schema().names
    predictor_columns = [name for name in schema_names if name not in BASE_COLUMNS]
    kept_features = [
        item["feature_name"]
        for item in json.loads(manifest_path.read_text(encoding="utf-8"))
        if item["decision"] == "keep"
    ]
    missing = sorted(set(kept_features) - set(predictor_columns))
    extras = sorted(set(predictor_columns) - set(kept_features))
    duplicate_columns = [
        name for name, count in Counter(schema_names).items() if count > 1
    ]
    case_stats = (
        pl.scan_parquet(output_path)
        .select(
            pl.len().alias("row_count"),
            pl.col("case_id").n_unique().alias("distinct_case_id_count"),
        )
        .collect()
    )
    row_count = int(case_stats["row_count"][0])
    distinct_case_id_count = int(case_stats["distinct_case_id_count"][0])
    return {
        "row_count": row_count,
        "column_count": metadata.num_columns,
        "distinct_case_id_count": distinct_case_id_count,
        "one_row_per_case": row_count == distinct_case_id_count,
        "required_columns_present": all(column in schema_names for column in BASE_COLUMNS),
        "target_not_in_predictors": "target" not in predictor_columns,
        "duplicate_predictor_names": duplicate_columns,
        "missing_manifest_features": missing,
        "extra_predictor_columns": extras,
        "lineage_matches_manifest": not missing and not extras,
    }
