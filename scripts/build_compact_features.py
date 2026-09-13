"""Build the compact first-generation applicant feature manifest and SQL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.features.feature_selection import build_compact_feature_set  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-path",
        default=Path("artifacts/features/applicant_feature_registry.json"),
        type=Path,
    )
    parser.add_argument(
        "--source-catalog-path",
        default=Path("artifacts/source_catalog/feature_catalog.json"),
        type=Path,
    )
    parser.add_argument("--smoke-table", default="applicant_features_smoke")
    parser.add_argument(
        "--manifest-path",
        default=Path("artifacts/features/applicant_feature_selection_manifest.json"),
        type=Path,
    )
    parser.add_argument(
        "--sql-path",
        default=Path("sql/features/002_applicant_features_compact.sql"),
        type=Path,
    )
    parser.add_argument("--output-table", default="applicant_features_compact_smoke")
    parser.add_argument("--smoke-limit", default=1_000, type=int)
    parser.add_argument("--full", action="store_true", help="Generate full-case compact SQL.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm-full-execute",
        action="store_true",
        help="Required with --full --execute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke_limit = None if args.full else args.smoke_limit
    if args.full and args.execute and not args.confirm_full_execute:
        raise SystemExit("--full --execute requires --confirm-full-execute.")
    result = build_compact_feature_set(
        registry_path=args.registry_path,
        source_catalog_path=args.source_catalog_path,
        smoke_table=args.smoke_table,
        manifest_path=args.manifest_path,
        sql_path=args.sql_path,
        output_table=args.output_table,
        smoke_limit=smoke_limit,
        execute=args.execute,
    )
    print(f"original features: {result['original_feature_count']}")
    print(f"retained features: {result['retained_feature_count']}")
    print(f"excluded: {result['excluded_count']}")
    print(f"review_later: {result['review_later_count']}")
    print(f"retained by family: {result['retained_by_family']}")
    print(f"wrote {result['manifest_path']}")
    print(f"wrote {result['ambiguous_review_path']}")
    print(f"wrote {result['sql_path']}")
    if result["shape"]:
        print(
            f"{result['output_table']} shape: "
            f"{result['shape'][0]} rows x {result['shape'][1]} columns"
        )
    print(f"estimated materialized bytes: {result['estimated_materialized_bytes']:,}")


if __name__ == "__main__":
    main()
