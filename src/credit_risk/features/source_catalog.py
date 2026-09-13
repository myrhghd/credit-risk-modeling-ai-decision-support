"""Modeling-oriented source and feature catalog for raw Home Credit data."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk.ingestion.dataset_inventory import classify_table
from credit_risk.ingestion.raw_postgres import (
    LogicalTable,
    connect_from_env,
    discover_logical_tables,
    quote_ident,
)

KEY_COLUMNS = {"case_id", "num_group1", "num_group2"}
METADATA_COLUMNS = {"WEEK_NUM", "MONTH"}
TARGET_COLUMNS = {"target"}
DATE_COLUMNS = {"date_decision"}

SEMANTIC_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("amount", ("amount", "annuity", "income", "price", "payment", "paid", "debt")),
    ("date/days", ("date", "days", "dpd", "past due", "month", "week")),
    ("categorical status", ("status", "type", "category", "role", "education", "marital")),
    ("count", ("number of", "count", "times")),
    ("demographic", ("person", "birth", "employment", "address", "district", "gender")),
    ("payment", ("payment", "installment", "paid", "dpd", "past due")),
    ("application", ("application", "applicant", "request", "decision")),
    ("bureau/credit history", ("credit bureau", "bureau", "contract", "loan", "debt")),
    ("collateral/property", ("collateral", "property", "real estate")),
    ("tax/registry", ("tax", "registry", "deduction")),
)

FAMILY_BUSINESS_SUMMARIES: Mapping[str, str] = {
    "base": (
        "Application outcome anchor containing one row per case, the binary target, "
        "decision date, and time index."
    ),
    "static": (
        "Application-level/static applicant fields available at or near the decision snapshot."
    ),
    "static_credit_bureau": (
        "Static credit-bureau-derived fields available at the decision snapshot."
    ),
    "previous_application": "Previous Home Credit application or contract history.",
    "credit_bureau_a": "External credit bureau history from source A.",
    "credit_bureau_b": "External credit bureau history from source B.",
    "debit_card": "Debit card history linked to cases.",
    "deposit": "Deposit/account history linked to cases.",
    "other": "Other case-linked historical signals.",
    "person": "Person/application participant information.",
    "tax_registry_a": "Tax registry source A fields linked to cases.",
    "tax_registry_b": "Tax registry source B fields linked to cases.",
    "tax_registry_c": "Tax registry source C fields linked to cases.",
}


@dataclass(frozen=True)
class CatalogColumn:
    logical_table: str
    column_name: str
    description: str | None
    postgres_type: str
    source_type: str
    table_depth: str
    is_key_or_grouping_field: bool
    is_target: bool
    is_metadata_field: bool
    semantic_category: str
    static_or_repeated: str
    representative_values: tuple[str, ...]
    modeling_role: str
    leakage_candidate: bool
    notes: str


@dataclass(frozen=True)
class TableCatalogSummary:
    logical_table: str
    logical_family: str
    table_depth: str
    row_count: int
    column_count: int
    business_information: str
    historical_grain: str
    key_or_metadata_columns: tuple[str, ...]
    potentially_useful_columns: tuple[str, ...]
    ambiguous_columns: tuple[str, ...]
    leakage_candidates: tuple[str, ...]
    feature_engineering_categories: tuple[str, ...]


def read_feature_definitions(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    feature_column = next(
        column
        for column in ("Variable", "variable", "feature", "feature_name")
        if column in frame.columns
    )
    description_column = next(
        column for column in ("Description", "description") if column in frame.columns
    )
    return {
        str(row[feature_column]): str(row[description_column])
        for _, row in frame.dropna(subset=[feature_column]).iterrows()
        if pd.notna(row[description_column])
    }


def depth_label(table_name: str) -> str:
    if table_name == "base":
        return "base"
    if table_name.endswith("_0"):
        return "depth 0"
    if table_name.endswith("_1"):
        return "depth 1"
    if table_name.endswith("_2"):
        return "depth 2"
    return "unknown"


def family_name(table_name: str) -> str:
    family, _ = classify_table(f"train_{table_name}.parquet")
    return family


def static_or_repeated(table_name: str) -> str:
    depth = depth_label(table_name)
    if depth in {"base", "depth 0"}:
        return "static"
    if depth in {"depth 1", "depth 2"}:
        return "repeated/history-based"
    return "ambiguous"


def infer_semantic_category(column_name: str, description: str | None) -> str:
    if column_name in KEY_COLUMNS:
        return "key/grouping"
    if column_name in TARGET_COLUMNS:
        return "target"
    if column_name in METADATA_COLUMNS:
        return "time index"
    if column_name in DATE_COLUMNS or column_name.startswith("date_"):
        return "date/days"
    if not description:
        return "ambiguous"

    text = description.lower()
    matches = [
        category
        for category, tokens in SEMANTIC_RULES
        if any(token in text for token in tokens)
    ]
    if not matches:
        return "ambiguous"
    if "payment" in matches:
        return "payment"
    return matches[0]


def modeling_role(column_name: str, description: str | None) -> str:
    if column_name in KEY_COLUMNS:
        return "key/grouping"
    if column_name in METADATA_COLUMNS or column_name in DATE_COLUMNS:
        return "metadata/time"
    if column_name in TARGET_COLUMNS:
        return "target"
    if not description:
        return "candidate predictor; meaning ambiguous"
    return "candidate predictor"


def leakage_candidate(column_name: str, description: str | None) -> bool:
    if column_name == "target":
        return True
    if not description:
        return False
    text = f"{column_name} {description}".lower()
    return any(token in text for token in ("target", "default", "decision", "approval"))


def sample_values(conn: Any, table: LogicalTable, limit: int = 5) -> dict[str, tuple[str, ...]]:
    table_sql = f"raw.{quote_ident(table.name)}"
    columns_sql = ", ".join(quote_ident(column) for column in table.column_names)
    samples: dict[str, list[str]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute(f"SELECT {columns_sql} FROM {table_sql} LIMIT {limit};")
        rows = cur.fetchall()
    for row in rows:
        for column, value in zip(table.column_names, row, strict=True):
            if value is not None and len(samples[column]) < limit:
                samples[column].append(str(value))
    return {column: tuple(values) for column, values in samples.items()}


def postgres_column_types(conn: Any, table_names: Sequence[str]) -> dict[tuple[str, str], str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'raw'
              AND table_name = ANY(%s)
            ORDER BY table_name, ordinal_position;
            """,
            (list(table_names),),
        )
        return {(row[0], row[1]): row[2] for row in cur.fetchall()}


def build_catalog_columns(
    conn: Any,
    tables: Sequence[LogicalTable],
    definitions: Mapping[str, str],
) -> list[CatalogColumn]:
    pg_types = postgres_column_types(conn, [table.name for table in tables])
    catalog: list[CatalogColumn] = []
    for table in tables:
        table_samples = sample_values(conn, table)
        depth = depth_label(table.name)
        table_mode = static_or_repeated(table.name)
        for field in table.schema:
            description = definitions.get(field.name)
            is_key = field.name in KEY_COLUMNS
            is_target = field.name in TARGET_COLUMNS
            is_metadata = field.name in METADATA_COLUMNS or field.name in DATE_COLUMNS
            category = infer_semantic_category(field.name, description)
            ambiguous = category == "ambiguous"
            catalog.append(
                CatalogColumn(
                    logical_table=table.name,
                    column_name=field.name,
                    description=description,
                    postgres_type=pg_types.get((table.name, field.name), "unknown"),
                    source_type=str(field.type),
                    table_depth=depth,
                    is_key_or_grouping_field=is_key,
                    is_target=is_target,
                    is_metadata_field=is_metadata,
                    semantic_category=category,
                    static_or_repeated=table_mode,
                    representative_values=table_samples.get(field.name, ()),
                    modeling_role=modeling_role(field.name, description),
                    leakage_candidate=leakage_candidate(field.name, description),
                    notes="Meaning ambiguous from supplied definition." if ambiguous else "",
                )
            )
    return catalog


def summarize_table(
    table: LogicalTable, columns: Sequence[CatalogColumn]
) -> TableCatalogSummary:
    family = family_name(table.name)
    key_or_metadata = tuple(
        column.column_name
        for column in columns
        if column.is_key_or_grouping_field or column.is_metadata_field or column.is_target
    )
    useful = tuple(
        column.column_name
        for column in columns
        if column.modeling_role.startswith("candidate predictor")
    )
    ambiguous = tuple(
        column.column_name for column in columns if column.semantic_category == "ambiguous"
    )
    leakage = tuple(column.column_name for column in columns if column.leakage_candidate)
    categories = tuple(
        sorted(
            {
                column.semantic_category
                for column in columns
                if column.semantic_category not in {"key/grouping", "target", "time index"}
            }
        )
    )
    return TableCatalogSummary(
        logical_table=table.name,
        logical_family=family,
        table_depth=depth_label(table.name),
        row_count=table.source_row_count,
        column_count=len(table.column_names),
        business_information=FAMILY_BUSINESS_SUMMARIES.get(
            family, "Business meaning is ambiguous from table name and supplied definitions."
        ),
        historical_grain=historical_grain(table),
        key_or_metadata_columns=key_or_metadata,
        potentially_useful_columns=useful,
        ambiguous_columns=ambiguous,
        leakage_candidates=leakage,
        feature_engineering_categories=feature_engineering_categories(table.name, categories),
    )


def historical_grain(table: LogicalTable) -> str:
    grain = ["case_id"]
    if "num_group1" in table.column_names:
        grain.append("num_group1")
    if "num_group2" in table.column_names:
        grain.append("num_group2")
    if grain == ["case_id"]:
        return "one row per case or case-level snapshot"
    return "repeated records identified by " + ", ".join(grain)


def feature_engineering_categories(table_name: str, categories: Sequence[str]) -> tuple[str, ...]:
    suggestions = set()
    if static_or_repeated(table_name) == "static":
        suggestions.add("direct typed predictors and missingness indicators")
    else:
        suggestions.add("per-case counts and recency-aware aggregations")
        suggestions.add("min/mean/max/latest summaries for numeric history")
        suggestions.add("status/category frequency and latest-value summaries")
    if "payment" in categories:
        suggestions.add("delinquency/payment behavior summaries")
    if "date/days" in categories:
        suggestions.add("age/recency/duration transformations")
    if "amount" in categories:
        suggestions.add("amount totals, ratios, and utilization-style summaries")
    if "bureau/credit history" in categories:
        suggestions.add("credit-history burden and contract-status summaries")
    if "demographic" in categories:
        suggestions.add("stable applicant/person categorical encodings")
    return tuple(sorted(suggestions))


def feature_coverage(
    catalog_columns: Sequence[CatalogColumn],
    definitions: Mapping[str, str],
) -> dict[str, Any]:
    source_columns = {column.column_name for column in catalog_columns}
    described = {column.column_name for column in catalog_columns if column.description}
    return {
        "source_column_count": len(source_columns),
        "represented_source_column_count": len(described),
        "columns_without_descriptions": sorted(source_columns - described),
        "definitions_not_in_training_tables": sorted(set(definitions) - source_columns),
    }


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


def write_table_summary(
    path: Path, summaries: Sequence[TableCatalogSummary], coverage: Mapping[str, Any]
) -> None:
    rows = [
        (
            item.logical_table,
            item.table_depth,
            f"{item.row_count:,}",
            item.column_count,
            item.business_information,
            item.historical_grain,
            ", ".join(item.feature_engineering_categories),
        )
        for item in summaries
    ]
    represented_count = coverage["represented_source_column_count"]
    unused_count = len(coverage["definitions_not_in_training_tables"])
    content = f"""# Home Credit 2024 Source And Feature Catalog Summary

This catalog is modeling-oriented, not exhaustive profiling. It uses raw PostgreSQL
schemas, `feature_definitions.csv`, existing Parquet metadata, and small `LIMIT`
samples. It does not create indexes, physical staging copies, or engineered features.

## Feature Definition Coverage

- Unique raw source columns: {coverage["source_column_count"]:,}
- Raw columns represented in `feature_definitions.csv`: {represented_count:,}
- Raw columns without supplied descriptions: {len(coverage["columns_without_descriptions"]):,}
- Definitions not present in ingested training tables: {unused_count:,}

## Logical Tables

{markdown_table(
    (
        "table",
        "depth",
        "rows",
        "columns",
        "business information",
        "historical grain",
        "justified feature-engineering categories",
    ),
    rows,
)}

## Leakage Notes

`target` is the label and must be excluded from predictors. `date_decision`,
`WEEK_NUM`, and `MONTH` are temporal metadata suitable for validation design and
time-aware transformations, but not ordinary unconstrained predictors until the
temporal strategy is chosen. Other fields marked as leakage candidates are based
only on supplied descriptions containing decision/default/target language and need
manual review before modeling.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_source_catalog(
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("artifacts/source_catalog"),
) -> dict[str, Any]:
    tables = discover_logical_tables(raw_dir)
    definitions = read_feature_definitions(raw_dir / "feature_definitions.csv")
    conn = connect_from_env()
    try:
        columns = build_catalog_columns(conn, tables, definitions)
    finally:
        conn.close()

    by_table = defaultdict(list)
    for column in columns:
        by_table[column.logical_table].append(column)
    summaries = [summarize_table(table, by_table[table.name]) for table in tables]
    coverage = feature_coverage(columns, definitions)
    payload = {
        "columns": [asdict(column) for column in columns],
        "tables": [asdict(summary) for summary in summaries],
        "feature_definition_coverage": coverage,
        "recommendations": sorted(
            {
                category
                for summary in summaries
                for category in summary.feature_engineering_categories
            }
        ),
    }
    write_json(output_dir / "feature_catalog.json", payload)
    write_table_summary(output_dir / "table_family_summary.md", summaries, coverage)
    return payload
