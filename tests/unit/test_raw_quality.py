from pathlib import Path

import pandas as pd
import pyarrow as pa

from credit_risk.ingestion.raw_postgres import LogicalTable, SourceFile
from credit_risk.validation.raw_quality import (
    BaseProfile,
    FeatureDefinitionCoverage,
    RelationshipProfile,
    build_quality_checks,
    duplicate_key_check,
    expected_grain_columns,
    logical_depth,
    null_key_check,
    orphan_case_check,
    percentile_sorted,
    profile_feature_definitions,
    quality_status_summary,
    required_column_check,
    source_to_staging_view_check,
    target_validity_check,
)


def _table(name: str, columns: list[tuple[str, pa.DataType]]) -> LogicalTable:
    schema = pa.schema(columns)
    return LogicalTable(
        name,
        (SourceFile(Path(f"train_{name}.parquet"), name, 1, schema),),
        schema,
    )


def test_logical_depth_and_expected_grain_columns() -> None:
    assert logical_depth("base") == "base"
    assert logical_depth("static_0") == "depth 0"
    assert logical_depth("person_1") == "depth 1"
    assert logical_depth("person_2") == "depth 2"
    assert expected_grain_columns(["case_id", "num_group1", "num_group2", "x"]) == (
        "case_id",
        "num_group1",
        "num_group2",
    )


def test_percentile_sorted_interpolates_without_extra_storage() -> None:
    assert percentile_sorted([], 0.5) is None
    assert percentile_sorted([1], 0.95) == 1
    assert percentile_sorted([1, 3], 0.5) == 2
    assert percentile_sorted([1, 2, 3, 4, 5], 0.95) == 4.8


def test_quality_check_classification_helpers() -> None:
    assert required_column_check("x", ["case_id"], ["case_id"]).status == "PASS"
    assert required_column_check("x", [], ["case_id"]).status == "FAIL"
    assert null_key_check("x", 0).status == "PASS"
    assert null_key_check("x", 1).status == "FAIL"
    assert orphan_case_check("x", 0).status == "PASS"
    assert orphan_case_check("x", 1).status == "FAIL"
    assert duplicate_key_check("x", 0, ("case_id",)).status == "PASS"
    assert duplicate_key_check("x", 1, ("case_id",)).status == "WARNING"
    assert duplicate_key_check("x", None, ("case_id",)).status == "WARNING"
    assert source_to_staging_view_check("base", 10, 10).status == "PASS"
    assert source_to_staging_view_check("base", 10, 9).status == "FAIL"


def test_target_validity_check_requires_binary_non_null_target() -> None:
    valid = BaseProfile(
        total_cases=2,
        row_count=2,
        target_null_count=0,
        target_0_count=1,
        target_1_count=1,
        target_other_count=0,
        default_rate=0.5,
        date_decision_min="2020-01-01",
        date_decision_max="2020-01-02",
        week_num_min=1,
        week_num_max=2,
        week_num_cardinality=2,
        month_min=None,
        month_max=None,
        month_cardinality=None,
        case_id_unique=True,
        duplicate_case_id_count=0,
        null_case_id_count=0,
    )
    invalid = BaseProfile(**{**valid.__dict__, "target_other_count": 1})

    assert target_validity_check(valid).status == "PASS"
    assert target_validity_check(invalid).status == "FAIL"


def test_feature_definition_coverage(tmp_path: Path) -> None:
    definitions_path = tmp_path / "feature_definitions.csv"
    pd.DataFrame(
        {
            "Variable": ["case_id", "target", "unused_feature"],
            "Description": ["id", "label", "not loaded"],
        }
    ).to_csv(definitions_path, index=False)
    tables = [
        _table("base", [("case_id", pa.int64()), ("target", pa.int64()), ("x", pa.string())])
    ]

    coverage = profile_feature_definitions(definitions_path, tables)

    assert isinstance(coverage, FeatureDefinitionCoverage)
    assert coverage.source_column_count == 3
    assert coverage.represented_source_column_count == 2
    assert coverage.columns_without_descriptions == ("x",)
    assert coverage.definitions_not_in_training_tables == ("unused_feature",)


def test_build_quality_checks_summarizes_pass_warning_fail() -> None:
    tables = [
        _table(
            "base",
            [
                ("case_id", pa.int64()),
                ("target", pa.int64()),
                ("date_decision", pa.date32()),
                ("WEEK_NUM", pa.int64()),
            ],
        ),
        _table("person_1", [("case_id", pa.int64()), ("num_group1", pa.int64())]),
    ]
    base_profile = BaseProfile(
        total_cases=2,
        row_count=2,
        target_null_count=0,
        target_0_count=1,
        target_1_count=1,
        target_other_count=0,
        default_rate=0.5,
        date_decision_min="2020-01-01",
        date_decision_max="2020-01-02",
        week_num_min=1,
        week_num_max=2,
        week_num_cardinality=2,
        month_min=None,
        month_max=None,
        month_cardinality=None,
        case_id_unique=True,
        duplicate_case_id_count=0,
        null_case_id_count=0,
    )
    relationship = RelationshipProfile(
        table_name="person_1",
        logical_depth="depth 1",
        row_count=3,
        distinct_case_id_count=2,
        base_case_coverage_pct=100.0,
        null_case_id_count=0,
        orphan_case_id_count=0,
        min_rows_per_case=1,
        median_rows_per_case=1.5,
        mean_rows_per_case=1.5,
        p95_rows_per_case=2.0,
        max_rows_per_case=2,
        duplicate_key_count=1,
        duplicate_key_exact=True,
        has_num_group1=True,
        has_num_group2=False,
        null_num_group1_count=0,
        null_num_group2_count=None,
        negative_num_group1_count=0,
        negative_num_group2_count=None,
    )

    checks = build_quality_checks(tables, base_profile, [relationship], 2)
    summary = quality_status_summary(checks)

    assert summary["PASS"] > 0
    assert summary["WARNING"] == 1
    assert summary["FAIL"] == 0
