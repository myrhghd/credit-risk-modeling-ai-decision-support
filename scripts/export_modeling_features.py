"""Export compact applicant modeling features to compressed Parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.features.parquet_export import (  # noqa: E402
    DEFAULT_COMPRESSION,
    DEFAULT_EXPORT_DIR,
    DEFAULT_METADATA_PATH,
    export_feature_families,
    join_family_parquets,
    validate_joined_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export-families", "join", "validate"))
    parser.add_argument("--output-dir", default=DEFAULT_EXPORT_DIR, type=Path)
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH, type=Path)
    parser.add_argument(
        "--output-path",
        default=Path("data/processed/modeling_features_smoke.parquet"),
        type=Path,
    )
    parser.add_argument("--smoke-limit", default=1_000, type=int)
    parser.add_argument("--batch-size", default=50_000, type=int)
    parser.add_argument("--compression", default=DEFAULT_COMPRESSION)
    parser.add_argument("--family", action="append", dest="families")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke_limit = None if args.full else args.smoke_limit
    smoke = not args.full

    if args.command == "export-families":
        result = export_feature_families(
            output_dir=args.output_dir,
            metadata_path=args.metadata_path,
            smoke_limit=smoke_limit,
            batch_size=args.batch_size,
            compression=args.compression,
            force=args.force,
            families=args.families,
        )
        print(f"feature counts by family: {result['feature_counts_by_family']}")
        print(f"skipped families: {result['skipped_families']}")
        for row in result["exports"]:
            print(
                f"{row['family']}: {row['row_count']} rows x {row['column_count']} columns; "
                f"{row['file_size_bytes']} bytes"
            )
        return

    if args.command == "join":
        result = join_family_parquets(
            output_dir=args.output_dir,
            output_path=args.output_path,
            smoke=smoke,
            compression=args.compression,
        )
        print(
            f"{result['output_path']}: "
            f"{result['row_count']} rows x {result['column_count']} columns; "
            f"{result['file_size_bytes']} bytes"
        )
        return

    if args.command == "validate":
        result = validate_joined_parquet(args.output_path)
        print(result)


if __name__ == "__main__":
    main()
