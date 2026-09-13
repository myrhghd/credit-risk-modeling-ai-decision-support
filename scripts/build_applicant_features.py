"""Build first-layer applicant feature registry, SQL, and optional smoke table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.features.applicant_features import build_applicant_feature_layer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-path",
        default=Path("artifacts/source_catalog/feature_catalog.json"),
        type=Path,
    )
    parser.add_argument(
        "--registry-path",
        default=Path("artifacts/features/applicant_feature_registry.json"),
        type=Path,
    )
    parser.add_argument(
        "--sql-path",
        default=Path("sql/features/001_applicant_features.sql"),
        type=Path,
    )
    parser.add_argument("--output-table", default="applicant_features_smoke")
    parser.add_argument("--smoke-limit", default=1_000, type=int)
    parser.add_argument("--full", action="store_true", help="Generate full-case SQL without LIMIT.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm-full-execute",
        action="store_true",
        help="Required with --full --execute to materialize all cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke_limit = None if args.full else args.smoke_limit
    if args.full and args.execute and not args.confirm_full_execute:
        raise SystemExit("--full --execute requires --confirm-full-execute.")
    result = build_applicant_feature_layer(
        catalog_path=args.catalog_path,
        registry_path=args.registry_path,
        sql_path=args.sql_path,
        output_table=args.output_table,
        smoke_limit=smoke_limit,
        execute=args.execute,
    )
    print(f"features: {result['feature_count']}")
    print(f"families: {result['feature_counts_by_family']}")
    print(f"excluded leakage fields: {result['excluded_leakage_fields']}")
    print(f"wrote {args.registry_path}")
    print(f"wrote {args.sql_path}")
    if result["shape"]:
        print(
            f"{result['output_table']} shape: "
            f"{result['shape'][0]} rows x {result['shape'][1]} columns"
        )
    print(f"estimated materialized bytes: {result['storage_estimate_bytes']:,}")


if __name__ == "__main__":
    main()
