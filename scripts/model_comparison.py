"""Update the deterministic supervised model comparison registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def safe_repo_relative_path(path: Path, *, repo_root: Path = PROJECT_ROOT) -> str:
    resolved_root = repo_root.resolve()
    resolved_path = path.resolve() if path.is_absolute() else (resolved_root / path).resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("artifact directory must be inside the repository") from exc
    return relative_path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("artifacts/modeling/temporal_split_config.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/models/model_comparison.json"),
    )
    parser.add_argument("--model-name", default="logistic_regression_unweighted")
    parser.add_argument("--model-family", default="Logistic Regression")
    parser.add_argument("--role", default="baseline")
    parser.add_argument("--status", default="official")
    parser.add_argument(
        "--notes",
        default=(
            "Official Logistic Regression benchmark. Retained as both ranking and "
            "probability reference; recalibration should be reviewed before policy use."
        ),
    )
    return parser.parse_args()


def split_metric(metrics: dict[str, Any], split: str) -> dict[str, float]:
    row = metrics[split]
    return {
        "roc_auc": row["roc_auc"],
        "gini": row["gini"],
        "pr_auc": row["pr_auc"],
        "ks": row["ks"],
        "brier": row["brier_score"],
    }


def calibration(metrics: dict[str, Any], split: str) -> dict[str, float]:
    row = metrics[split]
    return {
        "intercept": row["calibration_intercept"],
        "slope": row["calibration_slope"],
    }


def top_decile(metrics: dict[str, Any], split: str) -> dict[str, float]:
    row = metrics[split]["deciles"][0]
    return {
        "default_rate": row["default_rate"],
        "lift": row["lift"],
    }


def top_30_gain(metrics: dict[str, Any], split: str) -> dict[str, float]:
    row = metrics[split]["deciles"][2]
    return {
        "cumulative_positive_pct": row["cumulative_positive_pct"],
        "cumulative_lift": row["cumulative_lift"],
    }


def build_entry(
    *,
    artifact_dir: Path,
    split_config_path: Path,
    model_name: str,
    model_family: str,
    role: str,
    status: str,
    notes: str,
) -> dict[str, Any]:
    safe_artifact_dir = safe_repo_relative_path(artifact_dir)
    artifact_path = PROJECT_ROOT / safe_artifact_dir
    metrics = json.loads((artifact_path / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((artifact_path / "model_metadata.json").read_text(encoding="utf-8"))
    split_config = json.loads(split_config_path.read_text(encoding="utf-8"))
    model_config = metadata["model_config"]
    split_diagnostics = split_config["split_diagnostics"]
    return {
        "model_name": model_name,
        "model_family": model_family,
        "role": role,
        "status": status,
        "class_weight": model_config["class_weight"],
        "epochs": model_config["max_epochs"],
        "batch_size": model_config["batch_size"],
        "temporal_ranges": {
            split: {
                "week_start": split_diagnostics[split]["week_start"],
                "week_end": split_diagnostics[split]["week_end"],
                "date_start": split_diagnostics[split]["date_start"],
                "date_end": split_diagnostics[split]["date_end"],
            }
            for split in ("train", "validation", "test")
        },
        "metrics": {
            split: split_metric(metrics, split) for split in ("train", "validation", "test")
        },
        "calibration": {
            split: calibration(metrics, split) for split in ("validation", "test")
        },
        "ranking_segments": {
            "validation_top_decile": top_decile(metrics, "validation"),
            "test_top_decile": top_decile(metrics, "test"),
            "validation_top_30_pct": top_30_gain(metrics, "validation"),
            "test_top_30_pct": top_30_gain(metrics, "test"),
        },
        "artifact_directory": safe_artifact_dir,
        "recommendation_notes": notes,
    }


def write_registry(output_path: Path, entry: dict[str, Any]) -> None:
    registry = {
        "schema_version": 1,
        "official_benchmark": entry["model_name"],
        "models": [entry],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    try:
        entry = build_entry(
            artifact_dir=args.artifact_dir,
            split_config_path=args.split_config,
            model_name=args.model_name,
            model_family=args.model_family,
            role=args.role,
            status=args.status,
            notes=args.notes,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    write_registry(args.output_path, entry)
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
