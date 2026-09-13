"""Compact first-generation applicant feature selection."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from credit_risk.features.applicant_features import (
    EXCLUDED_LEAKAGE_FIELDS,
    FeatureSpec,
    build_feature_sql,
    estimate_materialized_size_bytes,
    execute_sql,
    table_shape,
    write_sql,
)
from credit_risk.ingestion.raw_postgres import connect_from_env, quote_ident

SMOKE_MIN_NON_NULL = 10
SMOKE_EFFECTIVELY_CONSTANT_SHARE = 0.995

AMBIGUOUS_SOURCE_DECISIONS: Mapping[tuple[str, str], tuple[str, str]] = {
    ("applprev_2", "cacccardblochreas_147M"): (
        "KEEP",
        "Card blocking reason is a status-like credit behavior signal.",
    ),
    ("credit_bureau_b_1", "credor_3940957M"): (
        "EXCLUDE",
        "Creditor name is identifier-like and likely high-cardinality/free-text.",
    ),
    ("other_1", "amtdepositbalance_4809441A"): (
        "KEEP",
        "Definition supports a deposit-balance amount signal.",
    ),
    ("person_1", "relationshiptoclient_415T"): (
        "KEEP",
        "Relationship to client is a participant-context categorical signal.",
    ),
    ("person_1", "relationshiptoclient_642T"): (
        "KEEP",
        "Relationship to client is a participant-context categorical signal.",
    ),
    ("person_1", "remitter_829L"): (
        "REVIEW_LATER",
        "Remitter flag may be useful, but definition is too sparse for first compact set.",
    ),
    ("person_1", "safeguarantyflag_411L"): (
        "REVIEW_LATER",
        "Product/safeguard flag needs timing and policy review.",
    ),
    ("static_0", "eir_270L"): ("KEEP", "Interest rate is a direct loan-pricing signal."),
    ("static_0", "interestrategrace_34L"): (
        "REVIEW_LATER",
        "Grace-period interest rate has unclear applicability and sparse smoke evidence.",
    ),
    ("static_0", "isbidproduct_1095L"): (
        "REVIEW_LATER",
        "Cross-sell product flag needs timing and policy review.",
    ),
    ("static_0", "isbidproductrequest_292L"): (
        "REVIEW_LATER",
        "Cross-sell request flag needs timing and policy review.",
    ),
    ("static_0", "isdebitcard_729L"): (
        "REVIEW_LATER",
        "Debit-card product flag needs timing and product-policy review.",
    ),
    ("static_cb_0", "for3years_504L"): (
        "REVIEW_LATER",
        "Credit-history window field is broad and semantically underspecified.",
    ),
    ("static_cb_0", "forquarter_634L"): (
        "REVIEW_LATER",
        "Credit-history window field is broad and semantically underspecified.",
    ),
    ("static_cb_0", "fortoday_1092L"): (
        "REVIEW_LATER",
        "Same-day credit-history field needs timing/leakage review.",
    ),
    ("static_cb_0", "foryear_850L"): (
        "REVIEW_LATER",
        "Credit-history window field is broad and semantically underspecified.",
    ),
    ("static_cb_0", "riskassesment_302T"): (
        "EXCLUDE",
        "Known leakage-review field estimating default probability.",
    ),
    ("static_cb_0", "riskassesment_940T"): (
        "EXCLUDE",
        "Creditworthiness estimate is model/score-like and unsafe for first-generation model.",
    ),
    ("tax_registry_a_1", "name_4527232M"): (
        "EXCLUDE",
        "Employer name is identifier-like and unsuitable without privacy/cardinality treatment.",
    ),
    ("tax_registry_b_1", "name_4917606M"): (
        "EXCLUDE",
        "Employer name is identifier-like and unsuitable without privacy/cardinality treatment.",
    ),
    ("tax_registry_c_1", "employername_160M"): (
        "EXCLUDE",
        "Employer name is identifier-like and unsuitable without privacy/cardinality treatment.",
    ),
}


@dataclass(frozen=True)
class FeatureSelectionDecision:
    feature_name: str
    feature_family: str
    source_table: str
    source_columns: tuple[str, ...]
    transformation: str
    decision: str
    reason: str
    leakage_status: str


def load_registry(path: Path) -> list[FeatureSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        FeatureSpec(
            feature_name=item["feature_name"],
            source_table=item["source_table"],
            source_columns=tuple(item["source_columns"]),
            transformation=item["transformation"],
            feature_family=item["feature_family"],
            description=item["description"],
            leakage_review_status=item["leakage_review_status"],
            sql_expression=item["sql_expression"],
        )
        for item in payload
    ]


def load_source_catalog(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (column["logical_table"], column["column_name"]): column for column in payload["columns"]
    }


def fetch_smoke_stats(table_name: str = "applicant_features_smoke") -> dict[str, dict[str, Any]]:
    conn = connect_from_env()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'features' AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (table_name,),
            )
            columns = [row[0] for row in cur.fetchall()]
            feature_columns = [
                column
                for column in columns
                if column not in {"case_id", "target", "date_decision", "WEEK_NUM", "MONTH"}
            ]
            stats: dict[str, dict[str, Any]] = {}
            for column in feature_columns:
                cur.execute(
                    f"""
                    SELECT
                        count({quote_ident(column)})::bigint AS non_null_count,
                        count(DISTINCT {quote_ident(column)})::bigint AS distinct_count
                    FROM features.{quote_ident(table_name)};
                    """
                )
                non_null_count, distinct_count = cur.fetchone()
                stats[column] = {
                    "non_null_count": int(non_null_count),
                    "distinct_count": int(distinct_count),
                    "max_value_count": fetch_max_value_count(cur, table_name, column),
                }
            return stats
    finally:
        conn.close()


def fetch_max_value_count(cur: Any, table_name: str, column: str) -> int:
    cur.execute(
        f"""
        SELECT COALESCE(max(value_count), 0)::bigint
        FROM (
            SELECT count(*)::bigint AS value_count
            FROM features.{quote_ident(table_name)}
            WHERE {quote_ident(column)} IS NOT NULL
            GROUP BY {quote_ident(column)}
        ) value_counts;
        """
    )
    return int(cur.fetchone()[0])


def smoke_structural_decision(
    spec: FeatureSpec, smoke_stats: Mapping[str, Mapping[str, Any]]
) -> tuple[str, str] | None:
    stats = smoke_stats.get(spec.feature_name)
    if stats is None:
        return "EXCLUDE", "Feature is absent from smoke table."
    non_null_count = int(stats["non_null_count"])
    distinct_count = int(stats["distinct_count"])
    max_value_count = int(stats.get("max_value_count", 0))
    if non_null_count == 0:
        return None
    if non_null_count < SMOKE_MIN_NON_NULL and spec.transformation in {"direct", "is_missing"}:
        return "REVIEW_LATER", "Extremely sparse in smoke sample."
    if distinct_count <= 1 and spec.transformation in {"direct", "is_missing", "n_distinct"}:
        return "REVIEW_LATER", "Constant in smoke sample."
    if (
        max_value_count / non_null_count >= SMOKE_EFFECTIVELY_CONSTANT_SHARE
        and spec.transformation in {"direct", "is_missing", "n_distinct"}
    ):
        return "REVIEW_LATER", "Effectively constant in smoke sample."
    return None


def source_semantic_decision(
    spec: FeatureSpec, source_catalog: Mapping[tuple[str, str], Mapping[str, Any]]
) -> tuple[str, str] | None:
    source_key = (spec.source_table, spec.source_columns[0])
    if source_key in EXCLUDED_LEAKAGE_FIELDS:
        return "EXCLUDE", "Explicit leakage exclusion."
    if source_key in AMBIGUOUS_SOURCE_DECISIONS:
        return AMBIGUOUS_SOURCE_DECISIONS[source_key]
    source = source_catalog.get(source_key)
    if source and source.get("semantic_category") == "ambiguous":
        return "REVIEW_LATER", "Ambiguous source meaning from supplied definition."
    if source and identifier_like_source(source):
        return "EXCLUDE", "Identifier-like/free-text source field."
    return None


def identifier_like_source(source: Mapping[str, Any]) -> bool:
    text = f"{source['column_name']} {source.get('description') or ''}".lower()
    return bool(re.search(r"\b(name|employer|creditor|remitter)\b", text))


def select_features(
    specs: Sequence[FeatureSpec],
    source_catalog: Mapping[tuple[str, str], Mapping[str, Any]],
    smoke_stats: Mapping[str, Mapping[str, Any]],
) -> list[FeatureSelectionDecision]:
    decisions = []
    seen_names: set[str] = set()
    seen_specs: set[tuple[str, str, tuple[str, ...], str]] = set()
    for spec in sorted(specs, key=lambda item: item.feature_name):
        source_key = (spec.source_table, spec.source_columns[0])
        leakage_status = "excluded_leakage" if source_key in EXCLUDED_LEAKAGE_FIELDS else "reviewed"
        decision = "KEEP"
        reason = "Retained for first-generation breadth and supported lineage."

        spec_key = (
            spec.source_table,
            spec.feature_family,
            spec.source_columns,
            spec.transformation,
        )
        if spec.feature_name in seen_names or spec_key in seen_specs:
            decision, reason = "EXCLUDE", "Duplicate feature name or duplicate specification."
        else:
            semantic = source_semantic_decision(spec, source_catalog)
            structural = smoke_structural_decision(spec, smoke_stats)
            if semantic and semantic[0] == "EXCLUDE":
                decision, reason = semantic
            elif structural and structural[0] == "EXCLUDE":
                decision, reason = structural
            elif semantic and semantic[0] == "REVIEW_LATER":
                decision, reason = semantic
            elif structural and structural[0] == "REVIEW_LATER":
                decision, reason = structural
            elif semantic and semantic[0] == "KEEP":
                decision, reason = semantic

        seen_names.add(spec.feature_name)
        seen_specs.add(spec_key)
        decisions.append(
            FeatureSelectionDecision(
                feature_name=spec.feature_name,
                feature_family=spec.feature_family,
                source_table=spec.source_table,
                source_columns=spec.source_columns,
                transformation=spec.transformation,
                decision=decision.lower(),
                reason=reason,
                leakage_status=leakage_status,
            )
        )
    return decisions


def write_manifest(path: Path, decisions: Sequence[FeatureSelectionDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(decision) for decision in decisions], indent=2),
        encoding="utf-8",
    )


def write_ambiguous_review(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "source_table": table,
            "source_column": column,
            "decision": decision.lower(),
            "reason": reason,
        }
        for (table, column), (decision, reason) in sorted(AMBIGUOUS_SOURCE_DECISIONS.items())
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def kept_specs(
    specs: Sequence[FeatureSpec], decisions: Sequence[FeatureSelectionDecision]
) -> list[FeatureSpec]:
    keep_names = {decision.feature_name for decision in decisions if decision.decision == "keep"}
    return [spec for spec in specs if spec.feature_name in keep_names]


def feature_counts_by_family(
    decisions: Sequence[FeatureSelectionDecision], decision: str
) -> dict[str, int]:
    return dict(
        Counter(item.feature_family for item in decisions if item.decision == decision)
    )


def build_compact_feature_set(
    *,
    registry_path: Path = Path("artifacts/features/applicant_feature_registry.json"),
    source_catalog_path: Path = Path("artifacts/source_catalog/feature_catalog.json"),
    smoke_table: str = "applicant_features_smoke",
    manifest_path: Path = Path("artifacts/features/applicant_feature_selection_manifest.json"),
    ambiguous_review_path: Path = Path("artifacts/features/ambiguous_source_review.json"),
    sql_path: Path = Path("sql/features/002_applicant_features_compact.sql"),
    output_table: str = "applicant_features_compact",
    smoke_limit: int | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    specs = load_registry(registry_path)
    catalog = load_source_catalog(source_catalog_path)
    smoke_stats = fetch_smoke_stats(smoke_table)
    decisions = select_features(specs, catalog, smoke_stats)
    compact_specs = kept_specs(specs, decisions)
    write_manifest(manifest_path, decisions)
    write_ambiguous_review(ambiguous_review_path)
    sql = build_feature_sql(compact_specs, output_table=output_table, smoke_limit=smoke_limit)
    write_sql(sql_path, sql)

    shape = None
    if execute:
        conn = connect_from_env()
        try:
            execute_sql(conn, sql)
            shape = table_shape(conn, output_table)
        finally:
            conn.close()

    base_cases = 1_526_659 if smoke_limit is None else smoke_limit
    estimate = estimate_materialized_size_bytes(base_cases, len(compact_specs) + 5)
    return {
        "original_feature_count": len(specs),
        "retained_feature_count": len(compact_specs),
        "excluded_count": sum(1 for item in decisions if item.decision == "exclude"),
        "review_later_count": sum(1 for item in decisions if item.decision == "review_later"),
        "retained_by_family": feature_counts_by_family(decisions, "keep"),
        "manifest_path": str(manifest_path),
        "ambiguous_review_path": str(ambiguous_review_path),
        "sql_path": str(sql_path),
        "output_table": f"features.{output_table}",
        "shape": shape,
        "estimated_materialized_bytes": estimate,
    }
