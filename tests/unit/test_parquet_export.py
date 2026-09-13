import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from credit_risk.features.applicant_features import FeatureSpec
from credit_risk.features.parquet_export import (
    BASE_COLUMNS,
    build_family_query,
    completed_families,
    family_arrow_schema,
    group_specs_by_family,
    join_family_parquets,
    load_kept_specs,
    output_path_for_family,
    validate_joined_parquet,
    write_row_batches_to_parquet,
)


def _spec(name: str, family: str, table: str = "static_0") -> FeatureSpec:
    return FeatureSpec(
        feature_name=name,
        source_table=table,
        source_columns=("source_column",),
        transformation="direct",
        feature_family=family,
        description="Test feature.",
        leakage_review_status="approved",
        sql_expression=f'"{table}"."source_column"',
    )


def _write_parquet(path: Path, payload: dict[str, list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(payload), path)


def test_manifest_driven_family_selection(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    manifest_path = tmp_path / "manifest.json"
    registry_path.write_text(
        json.dumps(
            [
                {
                    "feature_name": "keep_me",
                    "source_table": "static_0",
                    "source_columns": ["x"],
                    "transformation": "direct",
                    "feature_family": "static applicant/bureau snapshots",
                    "description": "x",
                    "leakage_review_status": "approved",
                    "sql_expression": '"static_0"."x"',
                },
                {
                    "feature_name": "drop_me",
                    "source_table": "static_0",
                    "source_columns": ["target"],
                    "transformation": "direct",
                    "feature_family": "static applicant/bureau snapshots",
                    "description": "target",
                    "leakage_review_status": "excluded",
                    "sql_expression": '"static_0"."target"',
                },
            ]
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            [
                {"feature_name": "keep_me", "decision": "keep"},
                {"feature_name": "drop_me", "decision": "exclude"},
            ]
        ),
        encoding="utf-8",
    )

    specs = load_kept_specs(registry_path, manifest_path)

    assert [spec.feature_name for spec in specs] == ["keep_me"]


def test_grouping_and_family_query_are_deterministic_and_leakage_free() -> None:
    specs = [
        _spec("z_feature", "static applicant/bureau snapshots"),
        _spec("a_feature", "static applicant/bureau snapshots"),
    ]

    grouped = group_specs_by_family(specs)
    query = build_family_query(
        "static applicant/bureau snapshots",
        grouped["static applicant/bureau snapshots"],
        smoke_limit=10,
    )

    assert [spec.feature_name for spec in grouped["static applicant/bureau snapshots"]] == [
        "a_feature",
        "z_feature",
    ]
    assert "LIMIT 10" in query
    assert "target" not in query.lower().replace("base_cases", "")


def test_resumable_completed_families() -> None:
    rows = [
        {"family": "a", "completion_status": "success", "smoke": True},
        {"family": "b", "completion_status": "failed", "smoke": True},
        {"family": "c", "completion_status": "success", "smoke": False},
    ]

    assert completed_families(rows, smoke=True) == {"a"}
    assert completed_families(rows, smoke=False) == {"c"}


def test_output_path_for_family() -> None:
    assert output_path_for_family(Path("out"), "credit bureau histories", smoke=True) == Path(
        "out/credit_bureau_histories_smoke.parquet"
    )


def test_final_join_and_validation(tmp_path: Path) -> None:
    output_dir = tmp_path / "families"
    final_path = tmp_path / "modeling_features_smoke.parquet"
    manifest_path = tmp_path / "manifest.json"
    _write_parquet(
        output_dir / "base_labels_smoke.parquet",
        {
            "case_id": [1, 2],
            "target": [0, 1],
            "date_decision": ["2020-01-01", "2020-01-02"],
            "WEEK_NUM": [1, 1],
            "MONTH": [202001, 202001],
        },
    )
    _write_parquet(
        output_dir / "static_applicant_bureau_smoke.parquet",
        {"case_id": [1, 2], "static_feature": [10.0, 20.0]},
    )
    manifest_path.write_text(
        json.dumps([{"feature_name": "static_feature", "decision": "keep"}]),
        encoding="utf-8",
    )

    result = join_family_parquets(output_dir=output_dir, output_path=final_path, smoke=True)
    validation = validate_joined_parquet(final_path, manifest_path)

    assert result["row_count"] == 2
    assert result["column_count"] == len(BASE_COLUMNS) + 1
    assert validation["required_columns_present"]
    assert validation["target_not_in_predictors"]
    assert validation["lineage_matches_manifest"]


def test_parquet_writer_uses_stable_numeric_schema_across_null_first_batch(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "stable.parquet"
    schema = pa.schema([("case_id", pa.int64()), ("x", pa.float64())])

    rows, columns = write_row_batches_to_parquet(
        [[(1, None), (2, None)], [(3, 1.0), (4, 2.0)]],
        schema,
        output_path,
    )
    table = pq.read_table(output_path)

    assert (rows, columns) == (4, 2)
    assert table.schema.field("x").type == pa.float64()
    assert table.column("x").to_pylist() == [None, None, 1.0, 2.0]


def test_parquet_writer_preserves_all_null_numeric_column(tmp_path: Path) -> None:
    output_path = tmp_path / "all_null.parquet"
    schema = pa.schema([("case_id", pa.int64()), ("x", pa.float64())])

    write_row_batches_to_parquet([[(1, None), (2, None)]], schema, output_path)

    assert pq.read_schema(output_path).field("x").type == pa.float64()


def test_parquet_writer_preserves_nullable_strings_and_integer_counts(tmp_path: Path) -> None:
    output_path = tmp_path / "mixed.parquet"
    schema = pa.schema(
        [("case_id", pa.int64()), ("category", pa.string()), ("count_feature", pa.int64())]
    )

    write_row_batches_to_parquet(
        [[(1, None, None), (2, "A", 3)]],
        schema,
        output_path,
    )
    table = pq.read_table(output_path)

    assert table.schema.field("category").type == pa.string()
    assert table.schema.field("count_feature").type == pa.int64()
    assert table.column("category").to_pylist() == [None, "A"]
    assert table.column("count_feature").to_pylist() == [None, 3]


def test_failed_export_removes_temp_and_does_not_replace_final_file(tmp_path: Path) -> None:
    output_path = tmp_path / "atomic.parquet"
    original_schema = pa.schema([("case_id", pa.int64()), ("x", pa.float64())])
    write_row_batches_to_parquet([[(1, 1.0)]], original_schema, output_path)
    original_size = output_path.stat().st_size

    bad_schema = pa.schema([("case_id", pa.int64()), ("x", pa.float64())])
    try:
        write_row_batches_to_parquet([[(1, 1.0)], [(2, "not-a-float")]], bad_schema, output_path)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        pass

    assert output_path.exists()
    assert output_path.stat().st_size == original_size
    assert not output_path.with_name(output_path.name + ".tmp").exists()


def test_failed_family_manifest_row_is_not_completed() -> None:
    rows = [
        {"family": "credit bureau histories", "completion_status": "failed", "smoke": False}
    ]

    assert completed_families(rows, smoke=False) == set()


def test_family_schema_preserves_static_types_and_numeric_aggregates() -> None:
    specs = [
        FeatureSpec(
            "static_text",
            "static_0",
            ("category",),
            "direct",
            "static applicant/bureau snapshots",
            "text",
            "approved",
            '"static_0"."category"',
        ),
        FeatureSpec(
            "history_mean",
            "applprev_1",
            ("amount",),
            "mean",
            "previous applications",
            "mean",
            "approved",
            'avg(r."amount")::double precision',
        ),
    ]
    catalog = {
        ("static_0", "category"): {"postgres_type": "text"},
        ("applprev_1", "amount"): {"postgres_type": "double precision"},
    }

    schema = family_arrow_schema(specs, catalog)

    assert schema.field("case_id").type == pa.int64()
    assert schema.field("static_text").type == pa.string()
    assert schema.field("history_mean").type == pa.float64()
