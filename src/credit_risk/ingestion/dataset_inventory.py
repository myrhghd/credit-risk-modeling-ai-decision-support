"""Dataset inventory and lightweight profiling for raw Home Credit Parquet files."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments
    raise RuntimeError("pandas is required for dataset inventory profiling.") from exc

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments
    raise RuntimeError(
        "pyarrow is required to inspect Parquet metadata. Install project dependencies first."
    ) from exc


KEY_COLUMNS = ("case_id", "num_group1", "num_group2", "date_decision", "WEEK_NUM", "target")

FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("base", "base"),
    ("static_cb", "static_credit_bureau"),
    ("static", "static"),
    ("applprev", "previous_application"),
    ("credit_bureau_a", "credit_bureau_a"),
    ("credit_bureau_b", "credit_bureau_b"),
    ("debitcard", "debit_card"),
    ("deposit", "deposit"),
    ("person", "person"),
    ("other", "other"),
    ("tax_registry_a", "tax_registry_a"),
    ("tax_registry_b", "tax_registry_b"),
    ("tax_registry_c", "tax_registry_c"),
)


@dataclass(frozen=True)
class TableProfile:
    file_name: str
    file_size_bytes: int
    row_count: int
    column_count: int
    logical_family: str
    depth: int | None
    key_columns_present: list[str]
    schema: dict[str, str]


@dataclass(frozen=True)
class FeatureDefinitionProfile:
    file_name: str
    file_size_bytes: int
    row_count: int
    column_count: int
    columns: list[str]
    feature_name_column: str | None


def classify_table(file_name: str) -> tuple[str, int | None]:
    """Classify a Home Credit file name into a logical family and table depth."""
    stem = Path(file_name).stem
    name = stem.removeprefix("train_")

    depth_match = re.search(r"_(\d)(?:_\d+)?$", name)
    depth = int(depth_match.group(1)) if depth_match else None

    for marker, family in FAMILY_PATTERNS:
        if name == marker or name.startswith(f"{marker}_"):
            return family, depth

    return "unknown", depth


def profile_parquet_file(path: Path) -> TableProfile:
    metadata = pq.read_metadata(path)
    schema_arrow = metadata.schema.to_arrow_schema()
    schema = {field.name: str(field.type) for field in schema_arrow}
    key_columns_present = [column for column in KEY_COLUMNS if column in schema]
    logical_family, depth = classify_table(path.name)

    return TableProfile(
        file_name=path.name,
        file_size_bytes=path.stat().st_size,
        row_count=metadata.num_rows,
        column_count=metadata.num_columns,
        logical_family=logical_family,
        depth=depth,
        key_columns_present=key_columns_present,
        schema=schema,
    )


def profile_feature_definitions(path: Path) -> FeatureDefinitionProfile:
    frame = pd.read_csv(path)
    feature_name_column = next(
        (
            column
            for column in ("Variable", "variable", "feature", "feature_name")
            if column in frame.columns
        ),
        None,
    )

    return FeatureDefinitionProfile(
        file_name=path.name,
        file_size_bytes=path.stat().st_size,
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=list(frame.columns),
        feature_name_column=feature_name_column,
    )


def build_inventory(raw_dir: Path) -> dict[str, Any]:
    raw_dir = raw_dir.resolve()
    parquet_paths = sorted(raw_dir.glob("train_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No training Parquet files found in {raw_dir}")

    feature_definitions_path = raw_dir / "feature_definitions.csv"
    if not feature_definitions_path.exists():
        raise FileNotFoundError(f"Missing {feature_definitions_path}")

    tables = [profile_parquet_file(path) for path in parquet_paths]
    feature_definitions = profile_feature_definitions(feature_definitions_path)

    return {
        "raw_dir": str(raw_dir),
        "table_count": len(tables),
        "total_parquet_size_bytes": sum(table.file_size_bytes for table in tables),
        "total_rows": sum(table.row_count for table in tables),
        "tables": [asdict(table) for table in tables],
        "feature_definitions": asdict(feature_definitions),
    }


def write_inventory(inventory: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
