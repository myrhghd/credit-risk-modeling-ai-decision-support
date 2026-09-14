"""Train and evaluate the Logistic Regression credit-risk baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.models.logistic_baseline import run_logistic_baseline  # noqa: E402


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
        "--output-dir",
        type=Path,
        default=Path("artifacts/models/logistic_baseline"),
    )
    parser.add_argument("--batch-size", type=int, default=25_000)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument(
        "--class-weight",
        default=None,
        help='Use "balanced" or a numeric positive-class weight such as 3 or 5.',
    )
    parser.add_argument("--categorical-hash-features", type=int, default=2**12)
    parser.add_argument("--train-row-limit", type=int, default=100_000)
    parser.add_argument("--eval-row-limit", type=int, default=50_000)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_class_weight(value: str | None) -> str | float | None:
    if value is None:
        return None
    if value == "balanced":
        return value
    return float(value)


def main() -> None:
    args = parse_args()
    result = run_logistic_baseline(
        modeling_path=args.modeling_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        train_row_limit=None if args.full else args.train_row_limit,
        eval_row_limit=None if args.full else args.eval_row_limit,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        alpha=args.alpha,
        class_weight=parse_class_weight(args.class_weight),
        categorical_hash_features=args.categorical_hash_features,
        verbose=args.verbose,
    )
    metadata = result["metadata"]
    print(f"training_mode: {metadata['training_mode']}")
    print(f"predictor_counts: {metadata['predictor_counts']}")
    print(f"model_config: {metadata['model_config']}")
    for split, metrics in result["metrics"].items():
        print(
            f"{split}: rows={metrics['rows']}, positive_rate={metrics['positive_rate']:.4%}, "
            f"roc_auc={metrics['roc_auc']:.4f}, gini={metrics['gini']:.4f}, "
            f"pr_auc={metrics['pr_auc']:.4f}, ks={metrics['ks']:.4f}, "
            f"brier={metrics['brier_score']:.5f}, "
            f"calibration_intercept={metrics['calibration_intercept']:.4f}, "
            f"calibration_slope={metrics['calibration_slope']:.4f}"
        )


if __name__ == "__main__":
    main()
