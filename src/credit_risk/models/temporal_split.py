"""Chronological train/validation/test split utilities for modeling features."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

LABEL_COLUMNS = ("case_id", "target", "date_decision", "WEEK_NUM", "MONTH")
SPLIT_ORDER = ("train", "validation", "test")


@dataclass(frozen=True)
class WeekBoundary:
    train_start_week: int
    train_end_week: int
    validation_start_week: int
    validation_end_week: int
    test_start_week: int
    test_end_week: int


def read_label_frame(modeling_path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(modeling_path).select(list(LABEL_COLUMNS))


def temporal_week_profile(labels: pl.LazyFrame) -> pl.DataFrame:
    weekly = (
        labels.group_by("WEEK_NUM")
        .agg(
            [
                pl.len().alias("row_count"),
                pl.sum("target").alias("positive_cases"),
                pl.mean("target").alias("default_rate"),
                pl.min("date_decision").alias("min_date_decision"),
                pl.max("date_decision").alias("max_date_decision"),
            ]
        )
        .sort("WEEK_NUM")
        .collect()
    )
    total_rows = int(weekly["row_count"].sum())
    return weekly.with_columns(
        [
            pl.col("row_count").cum_sum().alias("cumulative_row_count"),
            (pl.col("row_count").cum_sum() / total_rows).alias("cumulative_pct"),
        ]
    )


def choose_week_boundaries(
    weekly_profile: pl.DataFrame,
    *,
    train_target_pct: float = 0.70,
    validation_target_pct: float = 0.15,
) -> WeekBoundary:
    if weekly_profile.is_empty():
        raise ValueError("Cannot choose temporal split boundaries from an empty profile.")
    if weekly_profile.height < 3:
        raise ValueError("At least three WEEK_NUM values are required for train/validation/test.")

    weeks = weekly_profile["WEEK_NUM"].to_list()
    cumulative = weekly_profile["cumulative_pct"].to_list()
    train_end_idx = _nearest_cumulative_index(cumulative, train_target_pct)
    validation_end_idx = _nearest_cumulative_index(
        cumulative,
        train_target_pct + validation_target_pct,
    )

    train_end_idx = max(0, min(train_end_idx, len(weeks) - 3))
    validation_end_idx = max(train_end_idx + 1, min(validation_end_idx, len(weeks) - 2))

    return WeekBoundary(
        train_start_week=int(weeks[0]),
        train_end_week=int(weeks[train_end_idx]),
        validation_start_week=int(weeks[train_end_idx + 1]),
        validation_end_week=int(weeks[validation_end_idx]),
        test_start_week=int(weeks[validation_end_idx + 1]),
        test_end_week=int(weeks[-1]),
    )


def _nearest_cumulative_index(cumulative: list[float], target_pct: float) -> int:
    return min(range(len(cumulative)), key=lambda idx: abs(cumulative[idx] - target_pct))


def assign_split_expression(boundary: WeekBoundary) -> pl.Expr:
    return (
        pl.when(pl.col("WEEK_NUM") <= boundary.train_end_week)
        .then(pl.lit("train"))
        .when(pl.col("WEEK_NUM") <= boundary.validation_end_week)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )


def split_labels(labels: pl.LazyFrame, boundary: WeekBoundary) -> pl.DataFrame:
    return labels.with_columns(assign_split_expression(boundary)).collect()


def split_diagnostics(split_frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    total_rows = split_frame.height
    grouped = (
        split_frame.group_by("split")
        .agg(
            [
                pl.len().alias("rows"),
                pl.sum("target").alias("positive_cases"),
                pl.mean("target").alias("default_rate"),
                pl.min("date_decision").alias("date_start"),
                pl.max("date_decision").alias("date_end"),
                pl.min("WEEK_NUM").alias("week_start"),
                pl.max("WEEK_NUM").alias("week_end"),
                pl.n_unique("case_id").alias("unique_case_ids"),
            ]
        )
        .to_dicts()
    )
    by_split = {row["split"]: row for row in grouped}
    for split in SPLIT_ORDER:
        row = by_split.get(split)
        if row is None:
            raise ValueError(f"Split {split!r} has no rows.")
        rows = int(row["rows"])
        diagnostics[split] = {
            "rows": rows,
            "row_pct": rows / total_rows,
            "positive_cases": int(row["positive_cases"]),
            "default_rate": float(row["default_rate"]),
            "date_start": str(row["date_start"]),
            "date_end": str(row["date_end"]),
            "week_start": int(row["week_start"]),
            "week_end": int(row["week_end"]),
            "unique_case_ids": int(row["unique_case_ids"]),
        }
    return diagnostics


def validate_split(split_frame: pl.DataFrame, boundary: WeekBoundary) -> dict[str, Any]:
    assigned_rows = split_frame.height
    unique_cases = split_frame["case_id"].n_unique()
    weeks_by_split = {
        split: set(
            split_frame.filter(pl.col("split") == split)["WEEK_NUM"].unique().to_list()
        )
        for split in SPLIT_ORDER
    }
    overlapping_weeks = sorted(
        (weeks_by_split["train"] & weeks_by_split["validation"])
        | (weeks_by_split["train"] & weeks_by_split["test"])
        | (weeks_by_split["validation"] & weeks_by_split["test"])
    )
    chronological = (
        boundary.train_end_week < boundary.validation_start_week
        and boundary.validation_end_week < boundary.test_start_week
    )
    duplicate_case_ids = assigned_rows - unique_cases
    return {
        "assigned_rows": assigned_rows,
        "unique_case_ids": unique_cases,
        "every_case_assigned_once": assigned_rows == unique_cases,
        "duplicate_case_id_count": duplicate_case_ids,
        "overlapping_weeks": overlapping_weeks,
        "no_overlapping_weeks": not overlapping_weeks,
        "chronological_ordering": chronological,
    }


def target_drift_summary(diagnostics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train_rate = diagnostics["train"]["default_rate"]
    validation_rate = diagnostics["validation"]["default_rate"]
    test_rate = diagnostics["test"]["default_rate"]
    max_abs_shift = max(
        abs(validation_rate - train_rate),
        abs(test_rate - train_rate),
        abs(test_rate - validation_rate),
    )
    return {
        "train_validation_abs_diff": abs(validation_rate - train_rate),
        "train_test_abs_diff": abs(test_rate - train_rate),
        "validation_test_abs_diff": abs(test_rate - validation_rate),
        "max_abs_diff": max_abs_shift,
        "material_shift": max_abs_shift >= 0.02,
        "material_shift_threshold": 0.02,
    }


def write_split_outputs(
    modeling_path: Path,
    *,
    config_path: Path = Path("artifacts/modeling/temporal_split_config.json"),
    labels_path: Path = Path("data/processed/modeling_split_labels.parquet"),
) -> dict[str, Any]:
    labels = read_label_frame(modeling_path)
    weekly_profile = temporal_week_profile(labels)
    boundary = choose_week_boundaries(weekly_profile)
    split_frame = split_labels(labels, boundary)
    diagnostics = split_diagnostics(split_frame)
    validation = validate_split(split_frame, boundary)
    drift = target_drift_summary(diagnostics)

    if not validation["every_case_assigned_once"]:
        raise ValueError("Temporal split did not assign every case exactly once.")
    if not validation["no_overlapping_weeks"]:
        raise ValueError("Temporal split produced overlapping WEEK_NUM assignments.")
    if not validation["chronological_ordering"]:
        raise ValueError("Temporal split boundaries are not chronological.")

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    split_frame.select(["case_id", "split", "WEEK_NUM", "date_decision", "target"]).write_parquet(
        labels_path,
        compression="zstd",
    )

    config = {
        "created_at": datetime.now(UTC).isoformat(),
        "modeling_path": str(modeling_path),
        "split_labels_path": str(labels_path),
        "target_split_pct": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "week_boundaries": asdict(boundary),
        "date_decision_min": str(weekly_profile["min_date_decision"].min()),
        "date_decision_max": str(weekly_profile["max_date_decision"].max()),
        "week_num_min": int(weekly_profile["WEEK_NUM"].min()),
        "week_num_max": int(weekly_profile["WEEK_NUM"].max()),
        "weekly_profile": weekly_profile.to_dicts(),
        "split_diagnostics": diagnostics,
        "target_drift": drift,
        "validation": validation,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config
