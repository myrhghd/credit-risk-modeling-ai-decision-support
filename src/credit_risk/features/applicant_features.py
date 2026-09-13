"""First-layer one-row-per-case applicant feature generation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from credit_risk.ingestion.raw_postgres import connect_from_env, quote_ident

EXCLUDED_LEAKAGE_FIELDS = {
    ("base", "target"),
    ("applprev_1", "approvaldate_319D"),
    ("static_0", "lastapprdate_640D"),
    ("static_cb_0", "riskassesment_302T"),
}
KEY_FIELDS = {"case_id", "num_group1", "num_group2"}
BASE_TEMPORAL_FIELDS = ("target", "date_decision", "WEEK_NUM", "MONTH")
STATIC_TABLES = {"static_0", "static_cb_0"}
FAMILY_GROUPS: Mapping[str, str] = {
    "static": "static applicant/bureau snapshots",
    "static_credit_bureau": "static applicant/bureau snapshots",
    "previous_application": "previous applications",
    "credit_bureau_a": "credit bureau histories",
    "credit_bureau_b": "credit bureau histories",
    "person": "person/participant histories",
    "debit_card": "debit/deposit/other histories",
    "deposit": "debit/deposit/other histories",
    "other": "debit/deposit/other histories",
    "tax_registry_a": "tax registry histories",
    "tax_registry_b": "tax registry histories",
    "tax_registry_c": "tax registry histories",
}
NUMERIC_TYPES = {
    "smallint",
    "integer",
    "bigint",
    "real",
    "double precision",
    "numeric",
}


@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    source_table: str
    source_columns: tuple[str, ...]
    transformation: str
    feature_family: str
    description: str
    leakage_review_status: str
    sql_expression: str


def load_source_catalog(
    path: Path = Path("artifacts/source_catalog/feature_catalog.json"),
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_feature_part(value: str) -> str:
    cleaned = []
    for char in value.lower():
        cleaned.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(cleaned).split("_") if part)


def deterministic_feature_name(source_table: str, source_column: str, transformation: str) -> str:
    return "__".join(
        (
            sanitize_feature_part(source_table),
            sanitize_feature_part(source_column),
            sanitize_feature_part(transformation),
        )
    )


def family_group(logical_family: str) -> str:
    return FAMILY_GROUPS.get(logical_family, "other histories")


def is_numeric_type(postgres_type: str) -> bool:
    normalized = postgres_type.lower()
    return normalized in NUMERIC_TYPES or normalized.startswith("numeric")


def is_date_like(column: Mapping[str, Any]) -> bool:
    return column["semantic_category"] == "date/days" or column["postgres_type"] == "date"


def is_categorical(column: Mapping[str, Any]) -> bool:
    return column["postgres_type"] == "text" or column["semantic_category"] == "categorical status"


def eligible_predictor(column: Mapping[str, Any]) -> bool:
    return (
        column["column_name"] not in KEY_FIELDS
        and not column["is_target"]
        and not column["is_metadata_field"]
        and (column["logical_table"], column["column_name"]) not in EXCLUDED_LEAKAGE_FIELDS
    )


def build_feature_registry(catalog: Mapping[str, Any]) -> list[FeatureSpec]:
    columns_by_table: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    family_by_table = {
        table["logical_table"]: table["logical_family"] for table in catalog["tables"]
    }
    for column in catalog["columns"]:
        columns_by_table[column["logical_table"]].append(column)

    specs: list[FeatureSpec] = []
    for table_name in sorted(columns_by_table):
        if table_name == "base":
            continue
        table_columns = columns_by_table[table_name]
        feature_family = family_group(family_by_table[table_name])
        if table_name in STATIC_TABLES:
            specs.extend(build_static_specs(table_name, table_columns, feature_family))
        else:
            specs.extend(build_repeated_specs(table_name, table_columns, feature_family))
    return sorted(specs, key=lambda spec: spec.feature_name)


def build_static_specs(
    table_name: str,
    columns: Sequence[Mapping[str, Any]],
    feature_family: str,
) -> list[FeatureSpec]:
    specs = []
    for column in columns:
        if not eligible_predictor(column):
            continue
        column_name = column["column_name"]
        feature_name = deterministic_feature_name(table_name, column_name, "direct")
        specs.append(
            FeatureSpec(
                feature_name=feature_name,
                source_table=table_name,
                source_columns=(column_name,),
                transformation="direct",
                feature_family=feature_family,
                description=column["description"] or "Direct static predictor; meaning ambiguous.",
                leakage_review_status="approved",
                sql_expression=f"{quote_ident(table_name)}.{quote_ident(column_name)}",
            )
        )
        if column["semantic_category"] in {"amount", "date/days", "payment"}:
            null_feature = deterministic_feature_name(table_name, column_name, "is_missing")
            specs.append(
                FeatureSpec(
                    feature_name=null_feature,
                    source_table=table_name,
                    source_columns=(column_name,),
                    transformation="is_missing",
                    feature_family=feature_family,
                    description=f"Missingness indicator for {column_name}.",
                    leakage_review_status="approved",
                    sql_expression=(
                        f"({quote_ident(table_name)}.{quote_ident(column_name)} IS NULL)::integer"
                    ),
                )
            )
    return specs


def build_repeated_specs(
    table_name: str,
    columns: Sequence[Mapping[str, Any]],
    feature_family: str,
) -> list[FeatureSpec]:
    specs = [
        FeatureSpec(
            feature_name=deterministic_feature_name(table_name, "records", "count"),
            source_table=table_name,
            source_columns=("case_id",),
            transformation="count",
            feature_family=feature_family,
            description=f"Number of {table_name} records linked to the case.",
            leakage_review_status="approved",
            sql_expression="count(*)::double precision",
        )
    ]
    for column in columns:
        if not eligible_predictor(column):
            continue
        column_name = column["column_name"]
        quoted = f"r.{quote_ident(column_name)}"
        if is_numeric_type(column["postgres_type"]):
            for transform, expression in numeric_aggregations(quoted, column["semantic_category"]):
                specs.append(
                    FeatureSpec(
                        feature_name=deterministic_feature_name(table_name, column_name, transform),
                        source_table=table_name,
                        source_columns=(column_name,),
                        transformation=transform,
                        feature_family=feature_family,
                        description=f"{transform} of {column_name}: {column['description']}",
                        leakage_review_status="approved",
                        sql_expression=expression,
                    )
                )
        elif is_date_like(column):
            for transform, expression in date_aggregations(quoted):
                specs.append(
                    FeatureSpec(
                        feature_name=deterministic_feature_name(table_name, column_name, transform),
                        source_table=table_name,
                        source_columns=(column_name,),
                        transformation=transform,
                        feature_family=feature_family,
                        description=f"{transform} of {column_name}: {column['description']}",
                        leakage_review_status="approved",
                        sql_expression=expression,
                    )
                )
        elif is_categorical(column):
            specs.append(
                FeatureSpec(
                    feature_name=deterministic_feature_name(table_name, column_name, "n_distinct"),
                    source_table=table_name,
                    source_columns=(column_name,),
                    transformation="n_distinct",
                    feature_family=feature_family,
                    description=(
                        f"Distinct observed values for {column_name}: {column['description']}"
                    ),
                    leakage_review_status="approved",
                    sql_expression=f"count(DISTINCT {quoted})::double precision",
                )
            )
    return specs


def numeric_aggregations(quoted_column: str, semantic_category: str) -> tuple[tuple[str, str], ...]:
    aggregations = [
        ("min", f"min({quoted_column})::double precision"),
        ("mean", f"avg({quoted_column})::double precision"),
        ("max", f"max({quoted_column})::double precision"),
        ("std", f"stddev_samp({quoted_column})::double precision"),
    ]
    if semantic_category in {"amount", "payment", "count"}:
        aggregations.append(("sum", f"sum({quoted_column})::double precision"))
    return tuple(aggregations)


def date_aggregations(quoted_column: str) -> tuple[tuple[str, str], ...]:
    date_expression = f"NULLIF({quoted_column}::text, '')::date"
    return (
        (
            "min_days_before_decision",
            f"min(b.date_decision::date - {date_expression})::double precision",
        ),
        (
            "mean_days_before_decision",
            f"avg(b.date_decision::date - {date_expression})::double precision",
        ),
        (
            "max_days_before_decision",
            f"max(b.date_decision::date - {date_expression})::double precision",
        ),
    )


def build_feature_sql(
    specs: Sequence[FeatureSpec],
    *,
    output_table: str,
    smoke_limit: int | None = None,
) -> str:
    specs_by_table: dict[str, list[FeatureSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_table[spec.source_table].append(spec)

    base_limit = f"\n    LIMIT {smoke_limit}" if smoke_limit is not None else ""
    ctes = [
        f"""base_cases AS (
    SELECT case_id, target, date_decision, "WEEK_NUM", "MONTH"
    FROM raw.base
    ORDER BY case_id{base_limit}
)"""
    ]
    join_clauses = []
    select_columns = [
        "b.case_id",
        "b.target",
        "b.date_decision",
        'b."WEEK_NUM"',
        'b."MONTH"',
    ]
    for table_name in sorted(specs_by_table):
        alias = sanitize_feature_part(table_name)
        table_specs = specs_by_table[table_name]
        source_sql = f"raw.{quote_ident(table_name)}"
        if smoke_limit is not None:
            source_sql = f"(SELECT * FROM raw.{quote_ident(table_name)} LIMIT {smoke_limit * 20})"
        if table_name in STATIC_TABLES:
            join_clauses.append(
                "LEFT JOIN "
                f"{source_sql} AS {quote_ident(table_name)} USING (case_id)"
            )
            select_columns.extend(
                f"{spec.sql_expression} AS {quote_ident(spec.feature_name)}" for spec in table_specs
            )
        else:
            ctes.append(build_repeated_cte(table_name, alias, table_specs, source_sql))
            join_clauses.append(f"LEFT JOIN {quote_ident(alias)} USING (case_id)")
            select_columns.extend(
                f"{quote_ident(alias)}.{quote_ident(spec.feature_name)}" for spec in table_specs
            )

    cte_sql = ",\n".join(ctes)
    select_sql = ",\n    ".join(select_columns)
    join_sql = "\n".join(join_clauses)

    return f"""CREATE SCHEMA IF NOT EXISTS features;

DROP TABLE IF EXISTS features.{quote_ident(output_table)};

CREATE TABLE features.{quote_ident(output_table)} AS
WITH
{cte_sql}
SELECT
    {select_sql}
FROM base_cases b
{join_sql};
"""


def build_repeated_cte(
    table_name: str, alias: str, specs: Sequence[FeatureSpec], source_sql: str
) -> str:
    expressions = ",\n        ".join(
        f"{spec.sql_expression} AS {quote_ident(spec.feature_name)}" for spec in specs
    )
    return f"""{quote_ident(alias)} AS (
    SELECT
        r.case_id,
        {expressions}
    FROM {source_sql} r
    JOIN base_cases b USING (case_id)
    GROUP BY r.case_id
)"""


def write_registry(path: Path, specs: Sequence[FeatureSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(spec) for spec in specs], indent=2),
        encoding="utf-8",
    )


def write_sql(path: Path, sql: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sql, encoding="utf-8")


def execute_sql(conn: Any, sql: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def table_shape(conn: Any, table_name: str) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM features.{quote_ident(table_name)};")
        rows = int(cur.fetchone()[0])
        cur.execute(
            """
            SELECT count(*)
            FROM information_schema.columns
            WHERE table_schema = 'features' AND table_name = %s;
            """,
            (table_name,),
        )
        columns = int(cur.fetchone()[0])
    return rows, columns


def estimate_materialized_size_bytes(case_count: int, column_count: int) -> int:
    return int(case_count * column_count * 8 * 1.35)


def feature_counts_by_family(specs: Sequence[FeatureSpec]) -> dict[str, int]:
    return dict(Counter(spec.feature_family for spec in specs))


def build_applicant_feature_layer(
    *,
    catalog_path: Path = Path("artifacts/source_catalog/feature_catalog.json"),
    registry_path: Path = Path("artifacts/features/applicant_feature_registry.json"),
    sql_path: Path = Path("sql/features/001_applicant_features.sql"),
    output_table: str = "applicant_features_smoke",
    smoke_limit: int | None = 1_000,
    execute: bool = False,
) -> dict[str, Any]:
    catalog = load_source_catalog(catalog_path)
    specs = build_feature_registry(catalog)
    sql = build_feature_sql(specs, output_table=output_table, smoke_limit=smoke_limit)
    write_registry(registry_path, specs)
    write_sql(sql_path, sql)

    shape = None
    if execute:
        conn = connect_from_env()
        try:
            execute_sql(conn, sql)
            shape = table_shape(conn, output_table)
        finally:
            conn.close()

    case_count = smoke_limit or next(
        table["row_count"] for table in catalog["tables"] if table["logical_table"] == "base"
    )
    return {
        "feature_count": len(specs),
        "feature_counts_by_family": feature_counts_by_family(specs),
        "excluded_leakage_fields": sorted(
            f"{table}.{column}" for table, column in EXCLUDED_LEAKAGE_FIELDS
        ),
        "output_table": f"features.{output_table}",
        "shape": shape,
        "storage_estimate_bytes": estimate_materialized_size_bytes(case_count, len(specs) + 5),
    }
