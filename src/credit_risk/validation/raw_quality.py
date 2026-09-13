"""Raw-to-staging relationship profiling and data-quality checks."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from credit_risk.ingestion.raw_postgres import (
    LogicalTable,
    connect_from_env,
    discover_logical_tables,
    quote_ident,
)

QUALITY_PASS = "PASS"
QUALITY_WARNING = "WARNING"
QUALITY_FAIL = "FAIL"
EXACT_DUPLICATE_ROW_THRESHOLD = 10_000_000


@dataclass(frozen=True)
class QualityCheck:
    check_name: str
    status: str
    table_name: str
    message: str


@dataclass(frozen=True)
class RelationshipProfile:
    table_name: str
    logical_depth: str
    row_count: int
    distinct_case_id_count: int
    base_case_coverage_pct: float
    null_case_id_count: int
    orphan_case_id_count: int
    min_rows_per_case: int | None
    median_rows_per_case: float | None
    mean_rows_per_case: float | None
    p95_rows_per_case: float | None
    max_rows_per_case: int | None
    duplicate_key_count: int | None
    duplicate_key_exact: bool
    has_num_group1: bool
    has_num_group2: bool
    null_num_group1_count: int | None
    null_num_group2_count: int | None
    negative_num_group1_count: int | None
    negative_num_group2_count: int | None


@dataclass(frozen=True)
class BaseProfile:
    total_cases: int
    row_count: int
    target_null_count: int
    target_0_count: int
    target_1_count: int
    target_other_count: int
    default_rate: float
    date_decision_min: str
    date_decision_max: str
    week_num_min: int
    week_num_max: int
    week_num_cardinality: int
    month_min: int | None
    month_max: int | None
    month_cardinality: int | None
    case_id_unique: bool
    duplicate_case_id_count: int
    null_case_id_count: int


@dataclass(frozen=True)
class FeatureDefinitionCoverage:
    definition_count: int
    source_column_count: int
    represented_source_column_count: int
    missing_description_count: int
    unused_definition_count: int
    columns_without_descriptions: tuple[str, ...]
    definitions_not_in_training_tables: tuple[str, ...]


def logical_depth(table_name: str) -> str:
    if table_name == "base":
        return "base"
    if table_name.endswith("_0"):
        return "depth 0"
    if table_name.endswith("_1"):
        return "depth 1"
    if table_name.endswith("_2"):
        return "depth 2"
    return "unknown"


def expected_grain_columns(column_names: Sequence[str]) -> tuple[str, ...]:
    grain = ["case_id"]
    if "num_group1" in column_names:
        grain.append("num_group1")
    if "num_group2" in column_names:
        grain.append("num_group2")
    return tuple(grain)


def classify_quality(status_inputs: Sequence[str]) -> str:
    if QUALITY_FAIL in status_inputs:
        return QUALITY_FAIL
    if QUALITY_WARNING in status_inputs:
        return QUALITY_WARNING
    return QUALITY_PASS


def required_column_check(
    table_name: str, columns: Sequence[str], required: Sequence[str]
) -> QualityCheck:
    missing = sorted(set(required) - set(columns))
    if missing:
        return QualityCheck(
            "required_columns",
            QUALITY_FAIL,
            table_name,
            f"Missing required columns: {', '.join(missing)}",
        )
    return QualityCheck("required_columns", QUALITY_PASS, table_name, "Required columns present.")


def null_key_check(table_name: str, null_count: int, key_name: str = "case_id") -> QualityCheck:
    if null_count:
        return QualityCheck(
            "null_keys",
            QUALITY_FAIL,
            table_name,
            f"{null_count:,} rows have null {key_name}.",
        )
    return QualityCheck("null_keys", QUALITY_PASS, table_name, f"No null {key_name} values.")


def duplicate_key_check(
    table_name: str, duplicate_count: int | None, grain: Sequence[str]
) -> QualityCheck:
    if duplicate_count is None:
        return QualityCheck(
            "duplicate_expected_grain",
            QUALITY_WARNING,
            table_name,
            f"Exact duplicate check deferred for grain ({', '.join(grain)}).",
        )
    if duplicate_count:
        return QualityCheck(
            "duplicate_expected_grain",
            QUALITY_WARNING,
            table_name,
            f"{duplicate_count:,} duplicate rows at grain ({', '.join(grain)}).",
        )
    return QualityCheck(
        "duplicate_expected_grain",
        QUALITY_PASS,
        table_name,
        f"No duplicates at grain ({', '.join(grain)}).",
    )


def orphan_case_check(table_name: str, orphan_count: int) -> QualityCheck:
    if orphan_count:
        return QualityCheck(
            "orphan_case_ids",
            QUALITY_FAIL,
            table_name,
            f"{orphan_count:,} distinct case_id values are not present in raw.base.",
        )
    return QualityCheck("orphan_case_ids", QUALITY_PASS, table_name, "No orphan case_id values.")


def target_validity_check(profile: BaseProfile) -> QualityCheck:
    if profile.target_null_count or profile.target_other_count:
        return QualityCheck(
            "target_validity",
            QUALITY_FAIL,
            "base",
            "Target contains nulls or non-binary values.",
        )
    return QualityCheck("target_validity", QUALITY_PASS, "base", "Target is binary and non-null.")


def group_identifier_check(table_name: str, profile: RelationshipProfile) -> QualityCheck:
    invalid = []
    if profile.negative_num_group1_count:
        invalid.append(f"num_group1 negatives={profile.negative_num_group1_count:,}")
    if profile.negative_num_group2_count:
        invalid.append(f"num_group2 negatives={profile.negative_num_group2_count:,}")
    if invalid:
        return QualityCheck(
            "group_identifier_validity",
            QUALITY_FAIL,
            table_name,
            "; ".join(invalid),
        )
    return QualityCheck(
        "group_identifier_validity",
        QUALITY_PASS,
        table_name,
        "No negative group identifiers detected.",
    )


def source_to_staging_view_check(
    table_name: str, source_row_count: int, staging_row_count: int
) -> QualityCheck:
    if source_row_count != staging_row_count:
        return QualityCheck(
            "source_to_staging_row_preservation",
            QUALITY_FAIL,
            table_name,
            f"raw rows={source_row_count:,}; staging rows={staging_row_count:,}.",
        )
    return QualityCheck(
        "source_to_staging_row_preservation",
        QUALITY_PASS,
        table_name,
        "Staging view preserves raw row count.",
    )


def fetch_database_size(conn: Any) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()));")
        return str(cur.fetchone()[0])


def apply_staging_sql(conn: Any, sql_path: Path = Path("sql/staging/001_raw_views.sql")) -> None:
    with conn.cursor() as cur:
        cur.execute(sql_path.read_text(encoding="utf-8"))
    conn.commit()


def profile_base(conn: Any) -> BaseProfile:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                count(*)::bigint,
                count(DISTINCT case_id)::bigint,
                count(*) FILTER (WHERE case_id IS NULL)::bigint,
                (count(*) - count(DISTINCT case_id))::bigint,
                count(*) FILTER (WHERE target IS NULL)::bigint,
                count(*) FILTER (WHERE target = 0)::bigint,
                count(*) FILTER (WHERE target = 1)::bigint,
                count(*) FILTER (WHERE target IS NOT NULL AND target NOT IN (0, 1))::bigint,
                min(date_decision)::text,
                max(date_decision)::text,
                min("WEEK_NUM")::integer,
                max("WEEK_NUM")::integer,
                count(DISTINCT "WEEK_NUM")::integer
            FROM raw.base;
            """
        )
        row = cur.fetchone()
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'raw'
                  AND table_name = 'base'
                  AND column_name = 'MONTH'
            );
            """
        )
        has_month = bool(cur.fetchone()[0])
        month_stats = (None, None, None)
        if has_month:
            cur.execute(
                """
                SELECT min("MONTH")::integer, max("MONTH")::integer,
                       count(DISTINCT "MONTH")::integer
                FROM raw.base;
                """
            )
            month_stats = cur.fetchone()

    row_count = int(row[0])
    target_1_count = int(row[6])
    target_non_null = int(row[5]) + int(row[6])
    default_rate = target_1_count / target_non_null if target_non_null else 0.0
    return BaseProfile(
        total_cases=int(row[1]),
        row_count=row_count,
        null_case_id_count=int(row[2]),
        duplicate_case_id_count=int(row[3]),
        target_null_count=int(row[4]),
        target_0_count=int(row[5]),
        target_1_count=int(row[6]),
        target_other_count=int(row[7]),
        default_rate=default_rate,
        date_decision_min=str(row[8]),
        date_decision_max=str(row[9]),
        week_num_min=int(row[10]),
        week_num_max=int(row[11]),
        week_num_cardinality=int(row[12]),
        month_min=month_stats[0],
        month_max=month_stats[1],
        month_cardinality=month_stats[2],
        case_id_unique=int(row[0]) == int(row[1]) and int(row[2]) == 0,
    )


def profile_relationship(
    conn: Any, table: LogicalTable, base_case_count: int
) -> RelationshipProfile:
    table_sql = f"raw.{quote_ident(table.name)}"
    if table.source_row_count > EXACT_DUPLICATE_ROW_THRESHOLD:
        return profile_large_relationship_from_parquet(conn, table, base_case_count)

    grain = expected_grain_columns(table.column_names)
    has_num_group1 = "num_group1" in table.column_names
    has_num_group2 = "num_group2" in table.column_names
    duplicate_key_count = None
    duplicate_key_exact = table.source_row_count <= EXACT_DUPLICATE_ROW_THRESHOLD

    group_parts = [
        "count(*) FILTER (WHERE case_id IS NULL)::bigint AS null_case_id_count",
        "count(DISTINCT case_id)::bigint AS distinct_case_id_count",
    ]
    if has_num_group1:
        group_parts.extend(
            [
                "count(*) FILTER (WHERE num_group1 IS NULL)::bigint AS null_num_group1_count",
                "count(*) FILTER (WHERE num_group1 < 0)::bigint AS negative_num_group1_count",
            ]
        )
    else:
        group_parts.extend(
            [
                "NULL::bigint AS null_num_group1_count",
                "NULL::bigint AS negative_num_group1_count",
            ]
        )
    if has_num_group2:
        group_parts.extend(
            [
                "count(*) FILTER (WHERE num_group2 IS NULL)::bigint AS null_num_group2_count",
                "count(*) FILTER (WHERE num_group2 < 0)::bigint AS negative_num_group2_count",
            ]
        )
    else:
        group_parts.extend(
            [
                "NULL::bigint AS null_num_group2_count",
                "NULL::bigint AS negative_num_group2_count",
            ]
        )

    with conn.cursor() as cur:
        cur.execute(
            f"""
            WITH base_cases AS (
                SELECT case_id FROM raw.base WHERE case_id IS NOT NULL
            ),
            per_case AS (
                SELECT case_id, count(*)::bigint AS rows_per_case
                FROM {table_sql}
                WHERE case_id IS NOT NULL
                GROUP BY case_id
            ),
            orphan_cases AS (
                SELECT count(*)::bigint AS orphan_case_id_count
                FROM per_case
                LEFT JOIN base_cases USING (case_id)
                WHERE base_cases.case_id IS NULL
            ),
            table_stats AS (
                SELECT
                    count(*)::bigint AS row_count,
                    {", ".join(group_parts)}
                FROM {table_sql}
            ),
            row_distribution AS (
                SELECT
                    min(rows_per_case)::bigint AS min_rows_per_case,
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY rows_per_case)::float
                        AS median_rows_per_case,
                    avg(rows_per_case)::float AS mean_rows_per_case,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY rows_per_case)::float
                        AS p95_rows_per_case,
                    max(rows_per_case)::bigint AS max_rows_per_case
                FROM per_case
            )
            SELECT
                table_stats.row_count,
                table_stats.distinct_case_id_count,
                table_stats.null_case_id_count,
                table_stats.null_num_group1_count,
                table_stats.negative_num_group1_count,
                table_stats.null_num_group2_count,
                table_stats.negative_num_group2_count,
                orphan_cases.orphan_case_id_count,
                row_distribution.min_rows_per_case,
                row_distribution.median_rows_per_case,
                row_distribution.mean_rows_per_case,
                row_distribution.p95_rows_per_case,
                row_distribution.max_rows_per_case
            FROM table_stats, orphan_cases, row_distribution;
            """
        )
        row = cur.fetchone()
        if duplicate_key_exact:
            grain_expr = (
                quote_ident(grain[0])
                if len(grain) == 1
                else f"({', '.join(quote_ident(column) for column in grain)})"
            )
            cur.execute(
                f"""
                SELECT (count(*) - count(DISTINCT {grain_expr}))::bigint
                FROM {table_sql};
                """
            )
            duplicate_key_count = int(cur.fetchone()[0])

    distinct_case_id_count = int(row[1])
    coverage = distinct_case_id_count / base_case_count * 100 if base_case_count else 0.0
    return RelationshipProfile(
        table_name=table.name,
        logical_depth=logical_depth(table.name),
        row_count=int(row[0]),
        distinct_case_id_count=distinct_case_id_count,
        base_case_coverage_pct=coverage,
        null_case_id_count=int(row[2]),
        null_num_group1_count=None if row[3] is None else int(row[3]),
        negative_num_group1_count=None if row[4] is None else int(row[4]),
        null_num_group2_count=None if row[5] is None else int(row[5]),
        negative_num_group2_count=None if row[6] is None else int(row[6]),
        orphan_case_id_count=int(row[7]),
        duplicate_key_count=duplicate_key_count,
        duplicate_key_exact=duplicate_key_exact,
        min_rows_per_case=None if row[8] is None else int(row[8]),
        median_rows_per_case=None if row[9] is None else float(row[9]),
        mean_rows_per_case=None if row[10] is None else float(row[10]),
        p95_rows_per_case=None if row[11] is None else float(row[11]),
        max_rows_per_case=None if row[12] is None else int(row[12]),
        has_num_group1=has_num_group1,
        has_num_group2=has_num_group2,
    )


def profile_large_relationship_from_parquet(
    conn: Any, table: LogicalTable, base_case_count: int
) -> RelationshipProfile:
    base_case_ids = fetch_base_case_ids(conn)
    pg_row_count = fetch_table_row_count(conn, table.name)
    if pg_row_count != table.source_row_count:
        raise RuntimeError(
            f"raw.{table.name} row count {pg_row_count:,} does not match "
            f"Parquet source row count {table.source_row_count:,}"
        )

    has_num_group1 = "num_group1" in table.column_names
    has_num_group2 = "num_group2" in table.column_names
    selected_columns = ["case_id"]
    if has_num_group1:
        selected_columns.append("num_group1")
    if has_num_group2:
        selected_columns.append("num_group2")

    per_case: dict[int, int] = {}
    null_case_id_count = 0
    null_num_group1_count = 0 if has_num_group1 else None
    null_num_group2_count = 0 if has_num_group2 else None
    negative_num_group1_count = 0 if has_num_group1 else None
    negative_num_group2_count = 0 if has_num_group2 else None

    for source in table.sources:
        parquet_file = pq.ParquetFile(source.path)
        for batch in parquet_file.iter_batches(batch_size=250_000, columns=selected_columns):
            columns = {
                name: batch.column(batch.schema.get_field_index(name)).to_pylist()
                for name in selected_columns
            }
            case_ids = columns["case_id"]
            group1 = columns.get("num_group1")
            group2 = columns.get("num_group2")
            for index, case_id in enumerate(case_ids):
                if case_id is None:
                    null_case_id_count += 1
                else:
                    per_case[int(case_id)] = per_case.get(int(case_id), 0) + 1
                if group1 is not None:
                    value = group1[index]
                    if value is None:
                        null_num_group1_count += 1
                    elif value < 0:
                        negative_num_group1_count += 1
                if group2 is not None:
                    value = group2[index]
                    if value is None:
                        null_num_group2_count += 1
                    elif value < 0:
                        negative_num_group2_count += 1

    row_counts = sorted(per_case.values())
    distinct_case_id_count = len(per_case)
    coverage = distinct_case_id_count / base_case_count * 100 if base_case_count else 0.0
    orphan_case_id_count = sum(1 for case_id in per_case if case_id not in base_case_ids)

    return RelationshipProfile(
        table_name=table.name,
        logical_depth=logical_depth(table.name),
        row_count=pg_row_count,
        distinct_case_id_count=distinct_case_id_count,
        base_case_coverage_pct=coverage,
        null_case_id_count=null_case_id_count,
        null_num_group1_count=null_num_group1_count,
        negative_num_group1_count=negative_num_group1_count,
        null_num_group2_count=null_num_group2_count,
        negative_num_group2_count=negative_num_group2_count,
        orphan_case_id_count=orphan_case_id_count,
        duplicate_key_count=None,
        duplicate_key_exact=False,
        min_rows_per_case=row_counts[0] if row_counts else None,
        median_rows_per_case=percentile_sorted(row_counts, 0.5),
        mean_rows_per_case=sum(row_counts) / len(row_counts) if row_counts else None,
        p95_rows_per_case=percentile_sorted(row_counts, 0.95),
        max_rows_per_case=row_counts[-1] if row_counts else None,
        has_num_group1=has_num_group1,
        has_num_group2=has_num_group2,
    )


def fetch_base_case_ids(conn: Any) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT case_id FROM raw.base WHERE case_id IS NOT NULL;")
        return {int(row[0]) for row in cur.fetchall()}


def fetch_table_row_count(conn: Any, table_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT loaded_row_count
            FROM raw.ingestion_logical_table
            WHERE logical_table = %s
              AND logical_table_status = 'success'
              AND loaded_row_count = source_row_count
            ORDER BY finished_at DESC NULLS LAST, started_at DESC
            LIMIT 1;
            """,
            (table_name,),
        )
        row = cur.fetchone()
        if row is not None:
            return int(row[0])
        cur.execute(f"SELECT count(*) FROM raw.{quote_ident(table_name)};")
        return int(cur.fetchone()[0])


def percentile_sorted(values: Sequence[int], percentile: float) -> float | None:
    if not values:
        return None
    index = (len(values) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def profile_feature_definitions(
    feature_definitions_path: Path,
    tables: Sequence[LogicalTable],
) -> FeatureDefinitionCoverage:
    frame = pd.read_csv(feature_definitions_path)
    feature_column = next(
        column
        for column in ("Variable", "variable", "feature", "feature_name")
        if column in frame.columns
    )
    definitions = set(frame[feature_column].dropna().astype(str))
    source_columns = sorted({column for table in tables for column in table.column_names})
    represented = sorted(set(source_columns) & definitions)
    missing = sorted(set(source_columns) - definitions)
    unused = sorted(definitions - set(source_columns))
    return FeatureDefinitionCoverage(
        definition_count=len(definitions),
        source_column_count=len(source_columns),
        represented_source_column_count=len(represented),
        missing_description_count=len(missing),
        unused_definition_count=len(unused),
        columns_without_descriptions=tuple(missing),
        definitions_not_in_training_tables=tuple(unused),
    )


def build_quality_checks(
    tables: Sequence[LogicalTable],
    base_profile: BaseProfile,
    relationships: Sequence[RelationshipProfile],
    staging_base_row_count: int,
) -> list[QualityCheck]:
    by_name = {table.name: table for table in tables}
    checks = [
        required_column_check(
            "base",
            by_name["base"].column_names,
            ("case_id", "target", "date_decision", "WEEK_NUM"),
        ),
        null_key_check("base", base_profile.null_case_id_count),
        duplicate_key_check("base", base_profile.duplicate_case_id_count, ("case_id",)),
        target_validity_check(base_profile),
        source_to_staging_view_check("base", base_profile.row_count, staging_base_row_count),
    ]
    for profile in relationships:
        table = by_name[profile.table_name]
        checks.append(required_column_check(table.name, table.column_names, ("case_id",)))
        checks.append(null_key_check(table.name, profile.null_case_id_count))
        checks.append(orphan_case_check(table.name, profile.orphan_case_id_count))
        checks.append(
            duplicate_key_check(
                table.name,
                profile.duplicate_key_count,
                expected_grain_columns(table.column_names),
            )
        )
        checks.append(group_identifier_check(table.name, profile))
    return checks


def quality_status_summary(checks: Iterable[QualityCheck]) -> dict[str, int]:
    summary = {QUALITY_PASS: 0, QUALITY_WARNING: 0, QUALITY_FAIL: 0}
    for check in checks:
        summary[check.status] += 1
    return summary


def get_staging_base_row_count(conn: Any) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM staging.base;")
        return int(cur.fetchone()[0])


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=to_jsonable), encoding="utf-8")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown_summary(
    path: Path,
    base_profile: BaseProfile,
    relationships: Sequence[RelationshipProfile],
    feature_coverage: FeatureDefinitionCoverage,
    checks: Sequence[QualityCheck],
    database_size_before: str,
    database_size_after: str,
) -> None:
    relationship_rows = [
        (
            item.table_name,
            item.logical_depth,
            f"{item.row_count:,}",
            f"{item.distinct_case_id_count:,}",
            f"{item.base_case_coverage_pct:.2f}%",
            item.null_case_id_count,
            item.orphan_case_id_count,
            item.duplicate_key_count if item.duplicate_key_exact else "deferred",
            item.max_rows_per_case,
        )
        for item in relationships
    ]
    check_rows = [
        (check.status, check.table_name, check.check_name, check.message) for check in checks
    ]
    content = f"""# Raw Relationship And Data-Quality Summary

## Database Size

- Before: {database_size_before}
- After: {database_size_after}

## Base

- Total rows/cases: {base_profile.row_count:,} / {base_profile.total_cases:,}
- Default rate: {base_profile.default_rate:.6f}
- Target counts: 0={base_profile.target_0_count:,}, 1={base_profile.target_1_count:,}
- `date_decision`: {base_profile.date_decision_min} to {base_profile.date_decision_max}
- `WEEK_NUM`: {base_profile.week_num_min} to {base_profile.week_num_max}
  ({base_profile.week_num_cardinality} distinct)
- `MONTH`: {base_profile.month_min} to {base_profile.month_max}
  ({base_profile.month_cardinality} distinct)
- Duplicate `case_id` rows: {base_profile.duplicate_case_id_count:,}
- Null `case_id` rows: {base_profile.null_case_id_count:,}

## Relationships

{markdown_table(
    (
        "table",
        "depth",
        "rows",
        "cases",
        "base coverage",
        "null case_id",
        "orphans",
        "duplicate grain rows",
        "max rows/case",
    ),
    relationship_rows,
)}

## Feature Definitions

- Source columns: {feature_coverage.source_column_count:,}
- Represented by definitions: {feature_coverage.represented_source_column_count:,}
- Columns without descriptions: {feature_coverage.missing_description_count:,}
- Definitions not in ingested training tables: {feature_coverage.unused_definition_count:,}

## Quality Checks

{markdown_table(("status", "table", "check", "message"), check_rows)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_profile(
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("artifacts/raw_quality"),
) -> dict[str, Any]:
    tables = discover_logical_tables(raw_dir)
    conn = connect_from_env()
    try:
        database_size_before = fetch_database_size(conn)
        apply_staging_sql(conn)
        base_profile = profile_base(conn)
        relationships = []
        for table in tables:
            print(f"profiling raw.{table.name} ({table.source_row_count:,} rows)", flush=True)
            relationships.append(profile_relationship(conn, table, base_profile.total_cases))
        feature_coverage = profile_feature_definitions(raw_dir / "feature_definitions.csv", tables)
        checks = build_quality_checks(
            tables,
            base_profile,
            relationships,
            get_staging_base_row_count(conn),
        )
        database_size_after = fetch_database_size(conn)
    finally:
        conn.close()

    payload = {
        "database_size_before": database_size_before,
        "database_size_after": database_size_after,
        "base_profile": asdict(base_profile),
        "relationships": [asdict(item) for item in relationships],
        "feature_definition_coverage": asdict(feature_coverage),
        "quality_checks": [asdict(check) for check in checks],
        "quality_status_summary": quality_status_summary(checks),
    }
    write_json(output_dir / "relationship_inventory.json", payload)
    write_markdown_summary(
        output_dir / "relationship_summary.md",
        base_profile,
        relationships,
        feature_coverage,
        checks,
        database_size_before,
        database_size_after,
    )
    return payload
