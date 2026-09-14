from pathlib import Path

import polars as pl

from credit_risk.models.temporal_split import (
    choose_week_boundaries,
    split_diagnostics,
    split_labels,
    temporal_week_profile,
    validate_split,
    write_split_outputs,
)


def _labels() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "case_id": list(range(1, 11)),
            "target": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            "date_decision": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-08",
                "2020-01-09",
                "2020-01-15",
                "2020-01-16",
                "2020-01-22",
                "2020-01-23",
                "2020-01-29",
                "2020-01-30",
            ],
            "WEEK_NUM": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "MONTH": [202001] * 10,
        }
    ).lazy()


def test_deterministic_split_boundaries() -> None:
    profile = temporal_week_profile(_labels())

    first = choose_week_boundaries(profile)
    second = choose_week_boundaries(profile)

    assert first == second
    assert first.train_start_week == 1
    assert first.train_end_week == 3
    assert first.validation_start_week == 4
    assert first.validation_end_week == 4
    assert first.test_start_week == 5
    assert first.test_end_week == 5


def test_chronological_ordering_and_no_overlapping_weeks() -> None:
    profile = temporal_week_profile(_labels())
    boundary = choose_week_boundaries(profile)
    split_frame = split_labels(_labels(), boundary)
    validation = validate_split(split_frame, boundary)

    assert validation["chronological_ordering"]
    assert validation["no_overlapping_weeks"]
    assert validation["overlapping_weeks"] == []


def test_every_case_assigned_exactly_once_and_no_case_overlap() -> None:
    profile = temporal_week_profile(_labels())
    boundary = choose_week_boundaries(profile)
    split_frame = split_labels(_labels(), boundary)
    validation = validate_split(split_frame, boundary)

    assert validation["assigned_rows"] == 10
    assert validation["unique_case_ids"] == 10
    assert validation["duplicate_case_id_count"] == 0
    assert validation["every_case_assigned_once"]


def test_split_diagnostics() -> None:
    profile = temporal_week_profile(_labels())
    boundary = choose_week_boundaries(profile)
    diagnostics = split_diagnostics(split_labels(_labels(), boundary))

    assert diagnostics["train"]["rows"] == 6
    assert diagnostics["validation"]["rows"] == 2
    assert diagnostics["test"]["rows"] == 2
    assert diagnostics["train"]["week_start"] == 1
    assert diagnostics["test"]["week_end"] == 5


def test_write_split_outputs(tmp_path: Path) -> None:
    modeling_path = tmp_path / "modeling.parquet"
    config_path = tmp_path / "temporal_split_config.json"
    labels_path = tmp_path / "modeling_split_labels.parquet"
    _labels().collect().write_parquet(modeling_path)

    config = write_split_outputs(
        modeling_path,
        config_path=config_path,
        labels_path=labels_path,
    )

    assert config_path.exists()
    assert labels_path.exists()
    assert config["validation"]["every_case_assigned_once"]
    assert pl.read_parquet(labels_path).height == 10
