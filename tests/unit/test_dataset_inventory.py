from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from credit_risk.ingestion.dataset_inventory import build_inventory, classify_table


def _write_parquet(path: Path, payload: dict[str, list[object]]) -> None:
    table = pa.Table.from_pydict(payload)
    pq.write_table(table, path)


def test_classify_table_extracts_family_and_depth() -> None:
    assert classify_table("train_base.parquet") == ("base", None)
    assert classify_table("train_static_0_1.parquet") == ("static", 0)
    assert classify_table("train_credit_bureau_a_2_10.parquet") == ("credit_bureau_a", 2)
    assert classify_table("train_applprev_1_0.parquet") == ("previous_application", 1)


def test_build_inventory_profiles_parquet_and_feature_definitions(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_parquet(
        raw_dir / "train_base.parquet",
        {
            "case_id": [1, 2],
            "date_decision": ["2020-01-01", "2020-01-02"],
            "WEEK_NUM": [1, 1],
            "target": [0, 1],
            "amt_annuity": [10.5, 20.0],
        },
    )
    _write_parquet(
        raw_dir / "train_person_2.parquet",
        {
            "case_id": [1, 1, 2],
            "num_group1": [0, 1, 0],
            "num_group2": [0, 0, 1],
            "person_role": ["main", "relative", "main"],
        },
    )
    pd.DataFrame(
        {
            "Variable": ["case_id", "amt_annuity"],
            "Description": ["Application identifier", "Annuity amount"],
        }
    ).to_csv(raw_dir / "feature_definitions.csv", index=False)

    inventory = build_inventory(raw_dir)

    assert inventory["table_count"] == 2
    assert inventory["total_rows"] == 5
    assert inventory["feature_definitions"]["row_count"] == 2
    assert inventory["feature_definitions"]["feature_name_column"] == "Variable"

    base = next(
        table for table in inventory["tables"] if table["file_name"] == "train_base.parquet"
    )
    assert base["row_count"] == 2
    assert base["column_count"] == 5
    assert base["logical_family"] == "base"
    assert base["depth"] is None
    assert base["key_columns_present"] == ["case_id", "date_decision", "WEEK_NUM", "target"]
    assert "amt_annuity" in base["schema"]

    person = next(
        table for table in inventory["tables"] if table["file_name"] == "train_person_2.parquet"
    )
    assert person["logical_family"] == "person"
    assert person["depth"] == 2
    assert person["key_columns_present"] == ["case_id", "num_group1", "num_group2"]
