"""Sparse, split-aware Logistic Regression baseline training."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight

from credit_risk.evaluation.metrics import evaluate_predictions

BASE_COLUMNS = ("case_id", "target", "date_decision", "WEEK_NUM", "MONTH")
DEFAULT_RANDOM_SEED = 20260913
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class PredictorSets:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    boolean: tuple[str, ...]

    @property
    def all_predictors(self) -> tuple[str, ...]:
        return self.numeric + self.categorical + self.boolean


@dataclass(frozen=True)
class PreprocessingConfig:
    numeric_imputation: str
    numeric_scaling: str
    categorical_encoding: str
    categorical_missing_token: str
    categorical_hash_features: int
    boolean_missing_value: float


@dataclass(frozen=True)
class ModelConfig:
    estimator: str
    loss: str
    penalty: str
    alpha: float
    class_weight: str | float | None
    random_seed: int
    max_epochs: int
    batch_size: int
    train_row_limit: int | None


def load_keep_predictors(
    manifest_path: Path = Path("artifacts/features/applicant_feature_selection_manifest.json"),
) -> list[str]:
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    return sorted(row["feature_name"] for row in rows if row["decision"] == "keep")


def infer_predictor_sets(modeling_path: Path, keep_predictors: Sequence[str]) -> PredictorSets:
    schema = pq.read_schema(modeling_path)
    available = set(schema.names)
    missing = sorted(set(keep_predictors) - available)
    if missing:
        raise ValueError(f"KEEP predictors missing from modeling parquet: {missing[:10]}")
    numeric: list[str] = []
    categorical: list[str] = []
    boolean: list[str] = []
    for name in sorted(keep_predictors):
        dtype = str(schema.field(name).type)
        if dtype in {"double", "float", "int64", "int32"}:
            numeric.append(name)
        elif dtype == "bool":
            boolean.append(name)
        else:
            categorical.append(name)
    return PredictorSets(tuple(numeric), tuple(categorical), tuple(boolean))


def split_case_ids(labels_path: Path, split: str) -> set[int]:
    table = pq.read_table(labels_path, columns=["case_id", "split"])
    splits = table.column("split").to_pylist()
    case_ids = table.column("case_id").to_pylist()
    return {
        int(case_id)
        for case_id, row_split in zip(case_ids, splits, strict=True)
        if row_split == split
    }


def split_week_range(labels_path: Path, split: str) -> tuple[int, int]:
    row = (
        pl.scan_parquet(labels_path)
        .filter(pl.col("split") == split)
        .select(
            [
                pl.min("WEEK_NUM").alias("week_min"),
                pl.max("WEEK_NUM").alias("week_max"),
            ]
        )
        .collect()
        .row(0, named=True)
    )
    return int(row["week_min"]), int(row["week_max"])


def iter_split_batches(
    modeling_path: Path,
    labels_path: Path,
    split: str,
    columns: Sequence[str],
    *,
    batch_size: int,
    row_limit: int | None = None,
) -> Iterator[dict[str, list[Any]]]:
    for batch in iter_split_record_batches(
        modeling_path,
        labels_path,
        split,
        columns,
        batch_size=batch_size,
        row_limit=row_limit,
    ):
        yield batch.to_pydict()


def iter_split_record_batches(
    modeling_path: Path,
    labels_path: Path,
    split: str,
    columns: Sequence[str],
    *,
    batch_size: int,
    row_limit: int | None = None,
) -> Iterator[pa.RecordBatch]:
    if row_limit is not None:
        yield from iter_limited_split_record_batches(
            modeling_path,
            labels_path,
            split,
            columns,
            batch_size=batch_size,
            row_limit=row_limit,
        )
        return

    week_min, week_max = split_week_range(labels_path, split)
    parquet_file = pq.ParquetFile(modeling_path)
    read_columns = list(dict.fromkeys(["case_id", "target", "WEEK_NUM", *columns]))
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
        week_values = batch.column("WEEK_NUM")
        mask = pc.and_(
            pc.greater_equal(week_values, pa.scalar(week_min, type=pa.int64())),
            pc.less_equal(week_values, pa.scalar(week_max, type=pa.int64())),
        )
        filtered = batch.filter(mask)
        if filtered.num_rows == 0:
            continue
        yield filtered.drop_columns(["WEEK_NUM"])


def iter_limited_split_batches(
    modeling_path: Path,
    labels_path: Path,
    split: str,
    columns: Sequence[str],
    *,
    batch_size: int,
    row_limit: int,
) -> Iterator[dict[str, list[Any]]]:
    for batch in iter_limited_split_record_batches(
        modeling_path,
        labels_path,
        split,
        columns,
        batch_size=batch_size,
        row_limit=row_limit,
    ):
        yield batch.to_pydict()


def iter_limited_split_record_batches(
    modeling_path: Path,
    labels_path: Path,
    split: str,
    columns: Sequence[str],
    *,
    batch_size: int,
    row_limit: int,
) -> Iterator[pa.RecordBatch]:
    read_columns = list(dict.fromkeys(["case_id", "target", *columns]))
    sample_ids = (
        pl.scan_parquet(labels_path)
        .filter(pl.col("split") == split)
        .select("case_id")
        .head(row_limit)
        .collect()
    )
    case_ids = sample_ids["case_id"].to_list()
    frame = (
        pl.scan_parquet(modeling_path)
        .filter(pl.col("case_id").is_in(case_ids))
        .select(read_columns)
        .collect()
    )
    frame = frame.join(sample_ids.with_row_index("split_order"), on="case_id").sort("split_order")
    table = frame.select(read_columns).to_arrow()
    yield from table.to_batches(max_chunksize=batch_size)


def fit_numeric_stats(
    modeling_path: Path,
    labels_path: Path,
    predictors: PredictorSets,
    *,
    batch_size: int,
    row_limit: int | None,
    verbose: bool = False,
) -> dict[str, dict[str, float]]:
    sums = np.zeros(len(predictors.numeric), dtype=np.float64)
    sum_squares = np.zeros(len(predictors.numeric), dtype=np.float64)
    counts = np.zeros(len(predictors.numeric), dtype=np.float64)
    expected_rows = expected_split_rows(labels_path, "train", row_limit=row_limit)
    expected_batches = math.ceil(expected_rows / batch_size)
    started = perf_counter()
    rows_seen = 0
    batch_iterator = iter_split_record_batches(
        modeling_path,
        labels_path,
        "train",
        predictors.numeric,
        batch_size=batch_size,
        row_limit=row_limit,
    )
    for batch_number in range(1, expected_batches + 1):
        if verbose:
            print(
                f"preprocessing_stats batch={batch_number}/{expected_batches} "
                f"stage=starting cumulative_elapsed_seconds={perf_counter() - started:.1f} "
                f"rows_processed={rows_seen}/{expected_rows}",
                flush=True,
            )
        read_started = perf_counter()
        try:
            batch = next(batch_iterator)
        except StopIteration:
            break
        read_seconds = perf_counter() - read_started
        if verbose:
            print(
                f"preprocessing_stats batch={batch_number}/{expected_batches} "
                f"stage=arrow_read_filter_complete stage_seconds={read_seconds:.1f} "
                f"cumulative_elapsed_seconds={perf_counter() - started:.1f} "
                f"batch_rows={batch.num_rows} rows_processed={rows_seen}/{expected_rows}",
                flush=True,
            )
        matrix_started = perf_counter()
        matrix = numeric_values_from_record_batch(
            batch,
            predictors.numeric,
            means=None,
            scales=None,
            fit_mode=True,
        )
        matrix_seconds = perf_counter() - matrix_started
        if verbose:
            print(
                f"preprocessing_stats batch={batch_number}/{expected_batches} "
                f"stage=numeric_conversion_complete stage_seconds={matrix_seconds:.1f} "
                f"cumulative_elapsed_seconds={perf_counter() - started:.1f} "
                f"batch_rows={batch.num_rows} rows_processed={rows_seen}/{expected_rows}",
                flush=True,
            )
        aggregate_started = perf_counter()
        valid = ~np.isnan(matrix)
        filled = np.nan_to_num(matrix, nan=0.0)
        sums += filled.sum(axis=0)
        sum_squares += (filled * filled).sum(axis=0)
        counts += valid.sum(axis=0)
        aggregate_seconds = perf_counter() - aggregate_started
        if verbose:
            print(
                f"preprocessing_stats batch={batch_number}/{expected_batches} "
                f"stage=stats_accumulation_complete stage_seconds={aggregate_seconds:.1f} "
                f"cumulative_elapsed_seconds={perf_counter() - started:.1f} "
                f"batch_rows={batch.num_rows} rows_processed={rows_seen}/{expected_rows}",
                flush=True,
            )
        rows_seen += batch.num_rows
        if verbose:
            elapsed = perf_counter() - started
            print(
                f"preprocessing_stats batch={batch_number}/{expected_batches} "
                f"stage=batch_complete cumulative_elapsed_seconds={elapsed:.1f} "
                f"rows_processed={rows_seen}/{expected_rows} "
                f"rows_per_second={rows_seen / elapsed:.0f}",
                flush=True,
            )
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.divide(sum_squares, counts, out=np.zeros_like(sums), where=counts > 0) - means**2
    scales = np.sqrt(np.maximum(variances, 0.0))
    scales[scales == 0.0] = 1.0
    return {
        "mean": dict(zip(predictors.numeric, means.tolist(), strict=True)),
        "scale": dict(zip(predictors.numeric, scales.tolist(), strict=True)),
    }


def training_targets(
    modeling_path: Path,
    labels_path: Path,
    *,
    batch_size: int,
    row_limit: int | None,
) -> np.ndarray:
    targets: list[np.ndarray] = []
    for batch in iter_split_record_batches(
        modeling_path,
        labels_path,
        "train",
        [],
        batch_size=batch_size,
        row_limit=row_limit,
    ):
        targets.append(record_column_numpy(batch, "target").astype(np.int8, copy=False))
    return np.concatenate(targets)


def resolved_class_weight(
    modeling_path: Path,
    labels_path: Path,
    model_config: ModelConfig,
) -> dict[int, float] | None:
    if model_config.class_weight is None:
        return None
    if isinstance(model_config.class_weight, int | float):
        return {0: 1.0, 1: float(model_config.class_weight)}
    if model_config.class_weight != "balanced":
        raise ValueError(f"Unsupported class_weight: {model_config.class_weight!r}")
    classes = np.array([0, 1], dtype=np.int8)
    y_train = training_targets(
        modeling_path,
        labels_path,
        batch_size=model_config.batch_size,
        row_limit=model_config.train_row_limit,
    )
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, weights, strict=True)}


def numeric_values(
    batch: dict[str, list[Any]],
    columns: Sequence[str],
    *,
    means: np.ndarray | None,
    scales: np.ndarray | None,
    fit_mode: bool = False,
) -> np.ndarray:
    matrix = np.empty((len(batch["case_id"]), len(columns)), dtype=np.float32)
    for idx, column in enumerate(columns):
        matrix[:, idx] = [np.nan if value is None else float(value) for value in batch[column]]
    if fit_mode:
        return matrix
    assert means is not None and scales is not None
    matrix = np.where(np.isnan(matrix), means, matrix)
    return ((matrix - means) / scales).astype(np.float32, copy=False)


def record_column_numpy(batch: pa.RecordBatch, column: str) -> np.ndarray:
    return batch.column(batch.schema.get_field_index(column)).to_numpy(zero_copy_only=False)


def numeric_values_from_record_batch(
    batch: pa.RecordBatch,
    columns: Sequence[str],
    *,
    means: np.ndarray | None,
    scales: np.ndarray | None,
    fit_mode: bool = False,
) -> np.ndarray:
    if not columns:
        return np.empty((batch.num_rows, 0), dtype=np.float32)
    frame = pa.Table.from_batches([batch]).select(columns).to_pandas(split_blocks=True)
    matrix = frame.to_numpy(dtype=np.float32, na_value=np.nan, copy=not fit_mode)
    if fit_mode:
        return matrix
    assert means is not None and scales is not None
    np.nan_to_num(matrix, copy=False, nan=0.0)
    missing = frame.isna().to_numpy()
    if missing.any():
        matrix[missing] = np.broadcast_to(means, matrix.shape)[missing]
    matrix -= means
    matrix /= scales
    return matrix.astype(np.float32, copy=False)


def categorical_tokens(batch: dict[str, list[Any]], columns: Sequence[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    row_count = len(batch["case_id"])
    for row_idx in range(row_count):
        tokens = []
        for column in columns:
            value = batch[column][row_idx]
            token_value = "__MISSING__" if value is None else str(value)
            tokens.append(f"{column}={token_value}")
        rows.append(tokens)
    return rows


def categorical_tokens_from_record_batch(
    batch: pa.RecordBatch,
    columns: Sequence[str],
) -> list[list[str]]:
    if not columns:
        return [[] for _ in range(batch.num_rows)]
    frame = pa.Table.from_batches([batch]).select(columns).to_pandas(split_blocks=True)
    frame = frame.astype("string").fillna("__MISSING__")
    values = frame.to_numpy(dtype=object, copy=False)
    return [
        [f"{column}={values[row_idx, col_idx]}" for col_idx, column in enumerate(columns)]
        for row_idx in range(values.shape[0])
    ]


def transform_batch(
    batch: dict[str, list[Any]],
    predictors: PredictorSets,
    stats: dict[str, dict[str, float]],
    hasher: FeatureHasher,
) -> sparse.csr_matrix:
    means = np.array([stats["mean"][name] for name in predictors.numeric], dtype=np.float32)
    scales = np.array([stats["scale"][name] for name in predictors.numeric], dtype=np.float32)
    numeric = sparse.csr_matrix(
        numeric_values(batch, predictors.numeric, means=means, scales=scales),
        dtype=np.float32,
    )
    bool_matrix = np.empty((len(batch["case_id"]), len(predictors.boolean)), dtype=np.float32)
    for idx, column in enumerate(predictors.boolean):
        bool_matrix[:, idx] = [
            -1.0 if value is None else float(bool(value)) for value in batch[column]
        ]
    categorical = hasher.transform(categorical_tokens(batch, predictors.categorical))
    return sparse.hstack(
        [numeric, sparse.csr_matrix(bool_matrix), categorical],
        format="csr",
        dtype=np.float32,
    )


def transform_record_batch(
    batch: pa.RecordBatch,
    predictors: PredictorSets,
    stats: dict[str, dict[str, float]],
    hasher: FeatureHasher,
) -> sparse.csr_matrix:
    means = np.array([stats["mean"][name] for name in predictors.numeric], dtype=np.float32)
    scales = np.array([stats["scale"][name] for name in predictors.numeric], dtype=np.float32)
    numeric = sparse.csr_matrix(
        numeric_values_from_record_batch(batch, predictors.numeric, means=means, scales=scales),
        dtype=np.float32,
    )
    if predictors.boolean:
        frame = (
            pa.Table.from_batches([batch])
            .select(predictors.boolean)
            .to_pandas(split_blocks=True)
        )
        bool_values = frame.astype("Float32").fillna(-1.0).to_numpy(dtype=np.float32, copy=False)
    else:
        bool_values = np.empty((batch.num_rows, 0), dtype=np.float32)
    categorical = hasher.transform(
        categorical_tokens_from_record_batch(batch, predictors.categorical)
    )
    return sparse.hstack(
        [numeric, sparse.csr_matrix(bool_values), categorical],
        format="csr",
        dtype=np.float32,
    )


def collect_scores(
    model: SGDClassifier,
    modeling_path: Path,
    labels_path: Path,
    split: str,
    predictors: PredictorSets,
    stats: dict[str, dict[str, float]],
    hasher: FeatureHasher,
    *,
    batch_size: int,
    row_limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for batch in iter_split_record_batches(
        modeling_path,
        labels_path,
        split,
        predictors.all_predictors,
        batch_size=batch_size,
        row_limit=row_limit,
    ):
        x_batch = transform_record_batch(batch, predictors, stats, hasher)
        targets.append(record_column_numpy(batch, "target").astype(np.int8, copy=False))
        scores.append(model.predict_proba(x_batch)[:, 1])
    return np.concatenate(targets), np.concatenate(scores)


def train_incremental_logistic(
    modeling_path: Path,
    labels_path: Path,
    predictors: PredictorSets,
    *,
    model_config: ModelConfig,
    preprocessing_config: PreprocessingConfig,
    verbose: bool = False,
) -> tuple[SGDClassifier, dict[str, dict[str, float]], FeatureHasher]:
    stats_started = perf_counter()
    stats = fit_numeric_stats(
        modeling_path,
        labels_path,
        predictors,
        batch_size=model_config.batch_size,
        row_limit=model_config.train_row_limit,
        verbose=verbose,
    )
    stats_seconds = perf_counter() - stats_started
    if verbose:
        print(f"preprocessing_stats_seconds={stats_seconds:.1f}", flush=True)
    hasher = FeatureHasher(
        n_features=preprocessing_config.categorical_hash_features,
        input_type="string",
        alternate_sign=False,
        dtype=np.float32,
    )
    class_weight = resolved_class_weight(modeling_path, labels_path, model_config)
    model = SGDClassifier(
        loss=model_config.loss,
        penalty=model_config.penalty,
        alpha=model_config.alpha,
        class_weight=class_weight,
        random_state=model_config.random_seed,
        max_iter=1,
        tol=None,
        learning_rate="optimal",
    )
    classes = np.array([0, 1], dtype=np.int8)
    expected_rows = expected_split_rows(
        labels_path,
        "train",
        row_limit=model_config.train_row_limit,
    )
    expected_batches = math.ceil(expected_rows / model_config.batch_size)
    total_training_started = perf_counter()
    for epoch in range(1, model_config.max_epochs + 1):
        epoch_started = perf_counter()
        rows_seen = 0
        for batch_number, batch in enumerate(
            iter_split_record_batches(
                modeling_path,
                labels_path,
                "train",
                predictors.all_predictors,
                batch_size=model_config.batch_size,
                row_limit=model_config.train_row_limit,
            ),
            start=1,
        ):
            transform_started = perf_counter()
            x_batch = transform_record_batch(batch, predictors, stats, hasher)
            transform_seconds = perf_counter() - transform_started
            y_batch = record_column_numpy(batch, "target").astype(np.int8, copy=False)
            fit_started = perf_counter()
            model.partial_fit(x_batch, y_batch, classes=classes)
            fit_seconds = perf_counter() - fit_started
            rows_seen += batch.num_rows
            if verbose:
                elapsed = perf_counter() - epoch_started
                rows_per_second = rows_seen / elapsed if elapsed else 0.0
                print(
                    f"epoch={epoch}/{model_config.max_epochs} "
                    f"batch={batch_number}/{expected_batches} "
                    f"rows={rows_seen}/{expected_rows} "
                    f"epoch_elapsed_seconds={elapsed:.1f} "
                    f"rows_per_second={rows_per_second:.0f} "
                    f"transform_seconds={transform_seconds:.2f} "
                    f"fit_seconds={fit_seconds:.2f}",
                    flush=True,
                )
        epoch_seconds = perf_counter() - epoch_started
        if verbose:
            remaining_epochs = model_config.max_epochs - epoch
            print(
                f"epoch={epoch} complete epoch_seconds={epoch_seconds:.1f} "
                f"estimated_remaining_training_seconds={remaining_epochs * epoch_seconds:.1f}",
                flush=True,
            )
    if verbose:
        print(
            f"training_seconds={perf_counter() - total_training_started:.1f}",
            flush=True,
        )
    return model, stats, hasher


def expected_split_rows(labels_path: Path, split: str, *, row_limit: int | None) -> int:
    rows = (
        pl.scan_parquet(labels_path)
        .filter(pl.col("split") == split)
        .select(pl.len().alias("rows"))
        .collect()
        .item()
    )
    return min(int(rows), row_limit) if row_limit is not None else int(rows)


def run_logistic_baseline(
    *,
    modeling_path: Path = Path("data/processed/modeling_features.parquet"),
    labels_path: Path = Path("data/processed/modeling_split_labels.parquet"),
    output_dir: Path = Path("artifacts/models/logistic_baseline"),
    manifest_path: Path = Path("artifacts/features/applicant_feature_selection_manifest.json"),
    train_row_limit: int | None = 100_000,
    eval_row_limit: int | None = 50_000,
    batch_size: int = 25_000,
    max_epochs: int = 2,
    alpha: float = 1e-4,
    class_weight: str | float | None = None,
    categorical_hash_features: int = 2**12,
    verbose: bool = False,
) -> dict[str, Any]:
    keep_predictors = load_keep_predictors(manifest_path)
    predictors = infer_predictor_sets(modeling_path, keep_predictors)
    preprocessing = PreprocessingConfig(
        numeric_imputation="train_mean",
        numeric_scaling="train_standardization",
        categorical_encoding="feature_hashing",
        categorical_missing_token="__MISSING__",
        categorical_hash_features=categorical_hash_features,
        boolean_missing_value=-1.0,
    )
    model_config = ModelConfig(
        estimator="sklearn.linear_model.SGDClassifier",
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight=class_weight,
        random_seed=DEFAULT_RANDOM_SEED,
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_row_limit=train_row_limit,
    )
    model, stats, hasher = train_incremental_logistic(
        modeling_path,
        labels_path,
        predictors,
        model_config=model_config,
        preprocessing_config=preprocessing,
        verbose=verbose,
    )
    metrics = {}
    for split in ("train", "validation", "test"):
        y_true, y_score = collect_scores(
            model,
            modeling_path,
            labels_path,
            split,
            predictors,
            stats,
            hasher,
            batch_size=batch_size,
            row_limit=eval_row_limit,
        )
        metrics[split] = evaluate_predictions(y_true, y_score)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "modeling_path": str(modeling_path),
        "labels_path": str(labels_path),
        "predictor_counts": {
            "total": len(predictors.all_predictors),
            "numeric": len(predictors.numeric),
            "categorical": len(predictors.categorical),
            "boolean": len(predictors.boolean),
        },
        "preprocessing": asdict(preprocessing),
        "model_config": asdict(model_config),
        "training_mode": "full" if train_row_limit is None else "bounded_smoke",
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for split in ("validation", "test"):
        (output_dir / f"{split}_deciles.json").write_text(
            json.dumps(metrics[split]["deciles"], indent=2),
            encoding="utf-8",
        )
    return {"metadata": metadata, "metrics": metrics}


def verify_baseline_inputs(
    modeling_path: Path,
    labels_path: Path,
    predictors: PredictorSets,
) -> dict[str, Any]:
    label_frame = pl.scan_parquet(labels_path).select(["case_id", "split"]).collect()
    split_sets = {
        split: set(label_frame.filter(pl.col("split") == split)["case_id"].to_list())
        for split in ("train", "validation", "test")
    }
    target_values = (
        pl.scan_parquet(modeling_path)
        .select("target")
        .unique()
        .collect()
        .sort("target")["target"]
        .to_list()
    )
    overlaps = {
        "train_validation": len(split_sets["train"] & split_sets["validation"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["validation"] & split_sets["test"]),
    }
    return {
        "target_values": target_values,
        "target_encoding": "0/1 with 1 interpreted as default",
        "score_column": "predict_proba(...)[..., 1]",
        "probability_direction": "higher score means higher estimated default risk",
        "preprocessing_fit_scope": (
            "numeric means/scales and balanced class weights fit on train only"
        ),
        "transform_scope": "validation/test reuse train-fitted preprocessing and hashing config",
        "predictor_count": len(predictors.all_predictors),
        "label_columns_excluded_from_predictors": all(
            column not in predictors.all_predictors for column in BASE_COLUMNS
        ),
        "split_case_overlaps": overlaps,
        "no_split_case_overlap": all(count == 0 for count in overlaps.values()),
    }


def run_class_weight_diagnostic(
    *,
    modeling_path: Path = Path("data/processed/modeling_features.parquet"),
    labels_path: Path = Path("data/processed/modeling_split_labels.parquet"),
    output_dir: Path = Path("artifacts/models/logistic_class_weight_diagnostic"),
    manifest_path: Path = Path("artifacts/features/applicant_feature_selection_manifest.json"),
    train_row_limit: int = 250_000,
    eval_row_limit: int = 100_000,
    batch_size: int = 25_000,
    max_epochs: int = 6,
    alpha: float = 1e-4,
    class_weights: Sequence[str | float | None] = (None, 3.0, 5.0, "balanced"),
    categorical_hash_features: int = 2**12,
) -> dict[str, Any]:
    keep_predictors = load_keep_predictors(manifest_path)
    predictors = infer_predictor_sets(modeling_path, keep_predictors)
    verification = verify_baseline_inputs(modeling_path, labels_path, predictors)
    preprocessing = PreprocessingConfig(
        numeric_imputation="train_mean",
        numeric_scaling="train_standardization",
        categorical_encoding="feature_hashing",
        categorical_missing_token="__MISSING__",
        categorical_hash_features=categorical_hash_features,
        boolean_missing_value=-1.0,
    )
    stats = fit_numeric_stats(
        modeling_path,
        labels_path,
        predictors,
        batch_size=batch_size,
        row_limit=train_row_limit,
    )
    hasher = FeatureHasher(
        n_features=categorical_hash_features,
        input_type="string",
        alternate_sign=False,
        dtype=np.float32,
    )
    train_batches = materialize_transformed_split(
        modeling_path,
        labels_path,
        "train",
        predictors,
        stats,
        hasher,
        batch_size=batch_size,
        row_limit=train_row_limit,
    )
    eval_batches = {
        split: materialize_transformed_split(
            modeling_path,
            labels_path,
            split,
            predictors,
            stats,
            hasher,
            batch_size=batch_size,
            row_limit=eval_row_limit,
        )
        for split in ("train", "validation")
    }
    train_targets_array = np.concatenate([y_batch for _, y_batch in train_batches])
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for weight in class_weights:
        started = perf_counter()
        name = class_weight_name(weight)
        model = fit_cached_logistic(
            train_batches,
            class_weight=resolved_cached_class_weight(weight, train_targets_array),
            alpha=alpha,
            max_epochs=max_epochs,
        )
        runtime_seconds = perf_counter() - started
        metrics = {
            split: evaluate_cached_batches(model, batches)
            for split, batches in eval_batches.items()
        }
        validation_top_decile = metrics["validation"]["deciles"][0]
        results.append(
            {
                "configuration": name,
                "class_weight": weight,
                "train_roc_auc": metrics["train"]["roc_auc"],
                "validation_roc_auc": metrics["validation"]["roc_auc"],
                "validation_pr_auc": metrics["validation"]["pr_auc"],
                "validation_ks": metrics["validation"]["ks"],
                "validation_brier": metrics["validation"]["brier_score"],
                "validation_top_decile_lift": validation_top_decile["lift"],
                "epochs_completed": max_epochs,
                "runtime_seconds": runtime_seconds,
                "train_positive_rate": metrics["train"]["positive_rate"],
                "validation_positive_rate": metrics["validation"]["positive_rate"],
            }
        )
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "bounded convergence/class-weight diagnostic before full logistic baseline",
        "train_row_limit": train_row_limit,
        "eval_row_limit": eval_row_limit,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "alpha": alpha,
        "preprocessing": asdict(preprocessing),
        "predictor_counts": {
            "total": len(predictors.all_predictors),
            "numeric": len(predictors.numeric),
            "categorical": len(predictors.categorical),
            "boolean": len(predictors.boolean),
        },
        "verification": verification,
        "results": results,
    }
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def class_weight_name(weight: str | float | None) -> str:
    if weight is None:
        return "unweighted"
    if isinstance(weight, int | float):
        return f"positive_weight_{weight:g}".replace(".", "_")
    return str(weight)


def benchmark_training_throughput(
    *,
    modeling_path: Path = Path("data/processed/modeling_features.parquet"),
    labels_path: Path = Path("data/processed/modeling_split_labels.parquet"),
    manifest_path: Path = Path("artifacts/features/applicant_feature_selection_manifest.json"),
    output_path: Path = Path("artifacts/models/logistic_throughput_benchmark.json"),
    train_row_limit: int = 200_000,
    eval_row_limit: int = 100_000,
    batch_size: int = 100_000,
    alpha: float = 1e-4,
    class_weight: float = 3.0,
) -> dict[str, Any]:
    keep_predictors = load_keep_predictors(manifest_path)
    predictors = infer_predictor_sets(modeling_path, keep_predictors)

    stats_started = perf_counter()
    stats = fit_numeric_stats(
        modeling_path,
        labels_path,
        predictors,
        batch_size=batch_size,
        row_limit=train_row_limit,
    )
    stats_seconds = perf_counter() - stats_started
    hasher = FeatureHasher(
        n_features=2**12,
        input_type="string",
        alternate_sign=False,
        dtype=np.float32,
    )

    batch_timings = time_representative_batch(
        modeling_path,
        labels_path,
        predictors,
        stats,
        hasher,
        batch_size=batch_size,
    )

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight={0: 1.0, 1: class_weight},
        random_state=DEFAULT_RANDOM_SEED,
        max_iter=1,
        tol=None,
        learning_rate="optimal",
    )
    classes = np.array([0, 1], dtype=np.int8)
    train_started = perf_counter()
    rows_seen = 0
    batch_count = 0
    for batch in iter_split_record_batches(
        modeling_path,
        labels_path,
        "train",
        predictors.all_predictors,
        batch_size=batch_size,
        row_limit=train_row_limit,
    ):
        x_batch = transform_record_batch(batch, predictors, stats, hasher)
        y_batch = record_column_numpy(batch, "target").astype(np.int8, copy=False)
        model.partial_fit(x_batch, y_batch, classes=classes)
        rows_seen += batch.num_rows
        batch_count += 1
    training_seconds = perf_counter() - train_started

    eval_started = perf_counter()
    y_true, y_score = collect_scores(
        model,
        modeling_path,
        labels_path,
        "validation",
        predictors,
        stats,
        hasher,
        batch_size=batch_size,
        row_limit=eval_row_limit,
    )
    validation_auc = float(evaluate_predictions(y_true, y_score)["roc_auc"])
    eval_seconds = perf_counter() - eval_started

    full_train_rows = expected_split_rows(labels_path, "train", row_limit=None)
    rows_per_second = rows_seen / training_seconds if training_seconds else 0.0
    one_epoch_seconds = full_train_rows / rows_per_second if rows_per_second else float("inf")
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "predictor_counts": {
            "total": len(predictors.all_predictors),
            "numeric": len(predictors.numeric),
            "categorical": len(predictors.categorical),
            "boolean": len(predictors.boolean),
        },
        "train_row_limit": train_row_limit,
        "eval_row_limit": eval_row_limit,
        "batch_size": batch_size,
        "preprocessing_stats_seconds": stats_seconds,
        "representative_batch_timing": batch_timings,
        "benchmark_training_rows": rows_seen,
        "benchmark_training_batches": batch_count,
        "benchmark_training_seconds": training_seconds,
        "benchmark_rows_per_second": rows_per_second,
        "validation_eval_seconds": eval_seconds,
        "validation_auc_after_one_epoch": validation_auc,
        "full_train_rows": full_train_rows,
        "estimated_one_full_epoch_seconds": one_epoch_seconds,
        "estimated_three_full_epoch_seconds": one_epoch_seconds * 3,
        "go_no_go": "READY FOR FULL RUN" if one_epoch_seconds * 3 <= 3600 else (
            "NOT WORTH FURTHER LR OPTIMIZATION"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def time_representative_batch(
    modeling_path: Path,
    labels_path: Path,
    predictors: PredictorSets,
    stats: dict[str, dict[str, float]],
    hasher: FeatureHasher,
    *,
    batch_size: int,
) -> dict[str, float | int]:
    scan_started = perf_counter()
    batch = next(
        iter_split_record_batches(
            modeling_path,
            labels_path,
            "train",
            predictors.all_predictors,
            batch_size=batch_size,
            row_limit=batch_size,
        )
    )
    scan_seconds = perf_counter() - scan_started

    means = np.array([stats["mean"][name] for name in predictors.numeric], dtype=np.float32)
    scales = np.array([stats["scale"][name] for name in predictors.numeric], dtype=np.float32)
    numeric_extract_started = perf_counter()
    numeric_matrix = numeric_values_from_record_batch(
        batch,
        predictors.numeric,
        means=None,
        scales=None,
        fit_mode=True,
    )
    numeric_extract_seconds = perf_counter() - numeric_extract_started

    numeric_standardize_started = perf_counter()
    numeric_standardized = np.nan_to_num(numeric_matrix.copy(), nan=0.0)
    missing = np.isnan(numeric_matrix)
    if missing.any():
        numeric_standardized[missing] = np.broadcast_to(means, numeric_matrix.shape)[missing]
    numeric_standardized -= means
    numeric_standardized /= scales
    numeric_standardize_seconds = perf_counter() - numeric_standardize_started

    categorical_started = perf_counter()
    categorical = hasher.transform(
        categorical_tokens_from_record_batch(batch, predictors.categorical)
    )
    categorical_seconds = perf_counter() - categorical_started

    boolean_started = perf_counter()
    if predictors.boolean:
        bool_frame = (
            pa.Table.from_batches([batch])
            .select(predictors.boolean)
            .to_pandas(split_blocks=True)
        )
        bool_values = bool_frame.astype("Float32").fillna(-1.0).to_numpy(
            dtype=np.float32,
            copy=False,
        )
    else:
        bool_values = np.empty((batch.num_rows, 0), dtype=np.float32)
    boolean_seconds = perf_counter() - boolean_started

    assembly_started = perf_counter()
    x_batch = sparse.hstack(
        [
            sparse.csr_matrix(numeric_standardized.astype(np.float32, copy=False)),
            sparse.csr_matrix(bool_values),
            categorical,
        ],
        format="csr",
        dtype=np.float32,
    )
    assembly_seconds = perf_counter() - assembly_started

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        class_weight={0: 1.0, 1: 3.0},
        random_state=DEFAULT_RANDOM_SEED,
        max_iter=1,
        tol=None,
        learning_rate="optimal",
    )
    y_batch = record_column_numpy(batch, "target").astype(np.int8, copy=False)
    fit_started = perf_counter()
    model.partial_fit(x_batch, y_batch, classes=np.array([0, 1], dtype=np.int8))
    fit_seconds = perf_counter() - fit_started

    predict_started = perf_counter()
    model.predict_proba(x_batch)[:, 1]
    predict_seconds = perf_counter() - predict_started

    old_started = perf_counter()
    transform_batch(batch.to_pydict(), predictors, stats, hasher)
    old_transform_seconds = perf_counter() - old_started

    return {
        "rows": batch.num_rows,
        "parquet_arrow_scan_filter_seconds": scan_seconds,
        "numeric_extraction_seconds": numeric_extract_seconds,
        "numeric_impute_standardize_seconds": numeric_standardize_seconds,
        "categorical_extraction_hashing_seconds": categorical_seconds,
        "boolean_extraction_seconds": boolean_seconds,
        "sparse_assembly_seconds": assembly_seconds,
        "partial_fit_seconds": fit_seconds,
        "predict_proba_seconds": predict_seconds,
        "old_pydict_transform_seconds": old_transform_seconds,
    }


def materialize_transformed_split(
    modeling_path: Path,
    labels_path: Path,
    split: str,
    predictors: PredictorSets,
    stats: dict[str, dict[str, float]],
    hasher: FeatureHasher,
    *,
    batch_size: int,
    row_limit: int,
) -> list[tuple[sparse.csr_matrix, np.ndarray]]:
    batches = []
    for batch in iter_split_record_batches(
        modeling_path,
        labels_path,
        split,
        predictors.all_predictors,
        batch_size=batch_size,
        row_limit=row_limit,
    ):
        batches.append(
            (
                transform_record_batch(batch, predictors, stats, hasher),
                record_column_numpy(batch, "target").astype(np.int8, copy=False),
            )
        )
    return batches


def resolved_cached_class_weight(
    weight: str | float | None,
    y_train: np.ndarray,
) -> dict[int, float] | None:
    if weight is None:
        return None
    if isinstance(weight, int | float):
        return {0: 1.0, 1: float(weight)}
    if weight != "balanced":
        raise ValueError(f"Unsupported class_weight: {weight!r}")
    classes = np.array([0, 1], dtype=np.int8)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(value) for label, value in zip(classes, weights, strict=True)}


def fit_cached_logistic(
    train_batches: Sequence[tuple[sparse.csr_matrix, np.ndarray]],
    *,
    class_weight: dict[int, float] | None,
    alpha: float,
    max_epochs: int,
) -> SGDClassifier:
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight=class_weight,
        random_state=DEFAULT_RANDOM_SEED,
        max_iter=1,
        tol=None,
        learning_rate="optimal",
    )
    classes = np.array([0, 1], dtype=np.int8)
    for _epoch in range(max_epochs):
        for x_batch, y_batch in train_batches:
            model.partial_fit(x_batch, y_batch, classes=classes)
    return model


def evaluate_cached_batches(
    model: SGDClassifier,
    batches: Sequence[tuple[sparse.csr_matrix, np.ndarray]],
) -> dict[str, Any]:
    y_true = np.concatenate([y_batch for _, y_batch in batches])
    y_score = np.concatenate([model.predict_proba(x_batch)[:, 1] for x_batch, _ in batches])
    return evaluate_predictions(y_true, y_score)
