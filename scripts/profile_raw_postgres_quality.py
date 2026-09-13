"""Run raw PostgreSQL relationship profiling and data-quality checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.validation.raw_quality import run_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=Path("data/raw"), type=Path)
    parser.add_argument("--output-dir", default=Path("artifacts/raw_quality"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_profile(raw_dir=args.raw_dir, output_dir=args.output_dir)
    base = payload["base_profile"]
    summary = payload["quality_status_summary"]
    print(f"wrote {args.output_dir / 'relationship_inventory.json'}")
    print(f"wrote {args.output_dir / 'relationship_summary.md'}")
    print(
        "base: "
        f"{base['row_count']:,} rows; default_rate={base['default_rate']:.6f}; "
        f"date_decision={base['date_decision_min']}..{base['date_decision_max']}; "
        f"WEEK_NUM={base['week_num_min']}..{base['week_num_max']}"
    )
    print(
        "quality: "
        f"PASS={summary['PASS']}; WARNING={summary['WARNING']}; FAIL={summary['FAIL']}"
    )
    print(
        "database_size: "
        f"{payload['database_size_before']} before; {payload['database_size_after']} after"
    )


if __name__ == "__main__":
    main()
