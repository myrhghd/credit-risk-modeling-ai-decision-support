"""Run a bounded Logistic Regression convergence/class-weight diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.models.logistic_baseline import run_class_weight_diagnostic  # noqa: E402


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
        default=Path("artifacts/models/logistic_class_weight_diagnostic"),
    )
    parser.add_argument("--train-row-limit", type=int, default=250_000)
    parser.add_argument("--eval-row-limit", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=25_000)
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--class-weight", action="append", default=None)
    return parser.parse_args()


def parse_class_weight(value: str) -> str | float | None:
    if value == "none":
        return None
    if value == "balanced":
        return value
    return float(value)


def main() -> None:
    args = parse_args()
    result = run_class_weight_diagnostic(
        modeling_path=args.modeling_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        train_row_limit=args.train_row_limit,
        eval_row_limit=args.eval_row_limit,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        alpha=args.alpha,
        class_weights=[
            parse_class_weight(value)
            for value in (args.class_weight or ["none", "3", "5", "balanced"])
        ],
    )
    print(f"verification: {result['verification']}")
    for row in result["results"]:
        print(
            f"{row['configuration']}: "
            f"train_auc={row['train_roc_auc']:.4f}, "
            f"validation_auc={row['validation_roc_auc']:.4f}, "
            f"validation_pr_auc={row['validation_pr_auc']:.4f}, "
            f"validation_ks={row['validation_ks']:.4f}, "
            f"validation_brier={row['validation_brier']:.5f}, "
            f"top_decile_lift={row['validation_top_decile_lift']:.2f}, "
            f"epochs={row['epochs_completed']}, "
            f"runtime_seconds={row['runtime_seconds']:.1f}"
        )


if __name__ == "__main__":
    main()
