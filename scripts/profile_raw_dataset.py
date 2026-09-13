"""Create a reproducible inventory for raw Home Credit training data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.ingestion.dataset_inventory import build_inventory, write_inventory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/raw_dataset_inventory.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = build_inventory(args.raw_dir)
    write_inventory(inventory, args.output)
    print(f"Wrote inventory for {inventory['table_count']} tables to {args.output}")


if __name__ == "__main__":
    main()
