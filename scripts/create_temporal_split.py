"""Create chronological train/validation/test split metadata and labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.models.temporal_split import write_split_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modeling-path",
        type=Path,
        default=Path("data/processed/modeling_features.parquet"),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("artifacts/modeling/temporal_split_config.json"),
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("data/processed/modeling_split_labels.parquet"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = write_split_outputs(
        args.modeling_path,
        config_path=args.config_path,
        labels_path=args.labels_path,
    )
    print(f"date_decision: {config['date_decision_min']} to {config['date_decision_max']}")
    print(f"WEEK_NUM: {config['week_num_min']} to {config['week_num_max']}")
    print(f"week boundaries: {config['week_boundaries']}")
    for split, row in config["split_diagnostics"].items():
        print(
            f"{split}: {row['rows']} rows ({row['row_pct']:.2%}), "
            f"default_rate={row['default_rate']:.4%}, "
            f"weeks={row['week_start']}-{row['week_end']}, "
            f"dates={row['date_start']} to {row['date_end']}"
        )
    print(f"target drift: {config['target_drift']}")
    print(f"labels: {config['split_labels_path']}")
    print(f"config: {args.config_path}")


if __name__ == "__main__":
    main()
