from pathlib import Path

import pandas as pd
import pyarrow as pa

from credit_risk.features.source_catalog import (
    feature_coverage,
    historical_grain,
    infer_semantic_category,
    read_feature_definitions,
    static_or_repeated,
    summarize_table,
)
from credit_risk.ingestion.raw_postgres import LogicalTable, SourceFile


def _table(name: str, columns: list[tuple[str, pa.DataType]]) -> LogicalTable:
    schema = pa.schema(columns)
    return LogicalTable(
        name,
        (SourceFile(Path(f"train_{name}.parquet"), name, 10, schema),),
        schema,
    )


def test_read_feature_definitions(tmp_path: Path) -> None:
    path = tmp_path / "feature_definitions.csv"
    pd.DataFrame(
        {
            "Variable": ["amt_income_total", "status_123L"],
            "Description": ["Income amount", "Status"],
        }
    ).to_csv(path, index=False)

    assert read_feature_definitions(path) == {
        "amt_income_total": "Income amount",
        "status_123L": "Status",
    }


def test_semantic_category_uses_description_and_preserves_ambiguity() -> None:
    assert infer_semantic_category("case_id", None) == "key/grouping"
    assert infer_semantic_category("target", None) == "target"
    assert infer_semantic_category("WEEK_NUM", None) == "time index"
    assert (
        infer_semantic_category("actualdpd_943P", "Days Past Due of previous contract")
        == "payment"
    )
    assert infer_semantic_category("annuity_780A", "Contract annuity amount") == "amount"
    assert infer_semantic_category("mystery_123X", None) == "ambiguous"


def test_static_or_repeated_and_historical_grain() -> None:
    assert static_or_repeated("base") == "static"
    assert static_or_repeated("static_0") == "static"
    assert static_or_repeated("applprev_1") == "repeated/history-based"

    assert historical_grain(_table("base", [("case_id", pa.int64())])) == (
        "one row per case or case-level snapshot"
    )
    assert historical_grain(
        _table("person_2", [("case_id", pa.int64()), ("num_group1", pa.int64())])
    ) == "repeated records identified by case_id, num_group1"


def test_feature_coverage_counts_unique_raw_columns() -> None:
    class Column:
        def __init__(self, column_name: str, description: str | None) -> None:
            self.column_name = column_name
            self.description = description

    coverage = feature_coverage(
        [Column("case_id", None), Column("amt", "Amount"), Column("amt", "Amount")],
        {"amt": "Amount", "unused": "Unused"},
    )

    assert coverage["source_column_count"] == 2
    assert coverage["represented_source_column_count"] == 1
    assert coverage["columns_without_descriptions"] == ["case_id"]
    assert coverage["definitions_not_in_training_tables"] == ["unused"]


def test_summarize_table_marks_keys_and_candidate_predictors() -> None:
    table = _table(
        "static_0",
        [("case_id", pa.int64()), ("target", pa.int64()), ("amt_income", pa.float64())],
    )

    class Column:
        logical_table = "static_0"

        def __init__(self, name: str, role: str, category: str, leakage: bool = False) -> None:
            self.column_name = name
            self.modeling_role = role
            self.semantic_category = category
            self.is_key_or_grouping_field = name == "case_id"
            self.is_metadata_field = False
            self.is_target = name == "target"
            self.leakage_candidate = leakage

    summary = summarize_table(
        table,
        [
            Column("case_id", "key/grouping", "key/grouping"),
            Column("target", "target", "target", True),
            Column("amt_income", "candidate predictor", "amount"),
        ],
    )

    assert "case_id" in summary.key_or_metadata_columns
    assert summary.potentially_useful_columns == ("amt_income",)
    assert summary.leakage_candidates == ("target",)
    assert (
        "direct typed predictors and missingness indicators"
        in summary.feature_engineering_categories
    )
