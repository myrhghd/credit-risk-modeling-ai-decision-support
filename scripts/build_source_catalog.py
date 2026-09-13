"""Build the modeling-oriented Home Credit source/feature catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.features.source_catalog import build_source_catalog  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=Path("data/raw"), type=Path)
    parser.add_argument("--output-dir", default=Path("artifacts/source_catalog"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_source_catalog(raw_dir=args.raw_dir, output_dir=args.output_dir)
    coverage = payload["feature_definition_coverage"]
    print(f"wrote {args.output_dir / 'feature_catalog.json'}")
    print(f"wrote {args.output_dir / 'table_family_summary.md'}")
    print(
        "coverage: "
        f"{coverage['represented_source_column_count']:,}/"
        f"{coverage['source_column_count']:,} raw columns represented"
    )
    print(f"logical tables: {len(payload['tables'])}")
    print(f"catalog columns: {len(payload['columns'])}")


if __name__ == "__main__":
    main()
