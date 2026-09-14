"""Benchmark Logistic Regression preprocessing/training throughput."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.models.logistic_baseline import benchmark_training_throughput  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modeling-path",
        type=Path,
        default=Path("data/processed/modeling_features.parquet"),
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("data/processed/modeling_split_labels.parquet"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/models/logistic_throughput_benchmark.json"),
    )
    parser.add_argument("--train-row-limit", type=int, default=200_000)
    parser.add_argument("--eval-row-limit", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--class-weight", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark_training_throughput(
        modeling_path=args.modeling_path,
        labels_path=args.labels_path,
        output_path=args.output_path,
        train_row_limit=args.train_row_limit,
        eval_row_limit=args.eval_row_limit,
        batch_size=args.batch_size,
        alpha=args.alpha,
        class_weight=args.class_weight,
    )
    print(f"predictor_counts: {result['predictor_counts']}")
    print(f"representative_batch_timing: {result['representative_batch_timing']}")
    print(f"preprocessing_stats_seconds: {result['preprocessing_stats_seconds']:.1f}")
    print(f"benchmark_training_seconds: {result['benchmark_training_seconds']:.1f}")
    print(f"benchmark_rows_per_second: {result['benchmark_rows_per_second']:.0f}")
    print(f"validation_eval_seconds: {result['validation_eval_seconds']:.1f}")
    print(f"validation_auc_after_one_epoch: {result['validation_auc_after_one_epoch']:.4f}")
    print(f"estimated_one_full_epoch_seconds: {result['estimated_one_full_epoch_seconds']:.1f}")
    print(f"estimated_three_full_epoch_seconds: {result['estimated_three_full_epoch_seconds']:.1f}")
    print(result["go_no_go"])


if __name__ == "__main__":
    main()
