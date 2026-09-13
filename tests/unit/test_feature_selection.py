from credit_risk.features.applicant_features import FeatureSpec, build_feature_sql
from credit_risk.features.feature_selection import (
    AMBIGUOUS_SOURCE_DECISIONS,
    FeatureSelectionDecision,
    kept_specs,
    select_features,
    smoke_structural_decision,
)


def _spec(name: str, table: str, column: str, transformation: str = "direct") -> FeatureSpec:
    return FeatureSpec(
        feature_name=name,
        source_table=table,
        source_columns=(column,),
        transformation=transformation,
        feature_family="static applicant/bureau snapshots",
        description="Test feature.",
        leakage_review_status="approved",
        sql_expression=f'"{table}"."{column}"',
    )


def test_smoke_structural_decision_flags_empty_sparse_and_constant() -> None:
    empty = {"x": {"non_null_count": 0, "distinct_count": 0}}
    sparse = {"x": {"non_null_count": 2, "distinct_count": 2}}
    constant = {"x": {"non_null_count": 10, "distinct_count": 1}}
    useful = {"x": {"non_null_count": 10, "distinct_count": 2, "max_value_count": 5}}

    assert smoke_structural_decision(_spec("x", "static_0", "c"), empty) is None
    assert smoke_structural_decision(_spec("x", "t", "c"), empty) is None
    assert smoke_structural_decision(_spec("x", "static_0", "c"), sparse)[0] == "REVIEW_LATER"
    assert smoke_structural_decision(_spec("x", "t", "c"), constant)[0] == "REVIEW_LATER"
    assert smoke_structural_decision(_spec("x", "t", "c"), useful) is None


def test_select_features_is_deterministic_and_complete() -> None:
    specs = [
        _spec("b", "static_0", "income_001A"),
        _spec("a", "static_cb_0", "riskassesment_940T"),
    ]
    catalog = {
        ("static_0", "income_001A"): {"semantic_category": "amount", "column_name": "income_001A"},
        ("static_cb_0", "riskassesment_940T"): {
            "semantic_category": "ambiguous",
            "column_name": "riskassesment_940T",
            "description": "Estimate of client's creditworthiness.",
        },
    }
    smoke = {
        "a": {"non_null_count": 10, "distinct_count": 3},
        "b": {"non_null_count": 10, "distinct_count": 3},
    }

    first = select_features(specs, catalog, smoke)
    second = select_features(list(reversed(specs)), catalog, smoke)

    assert [item.feature_name for item in first] == ["a", "b"]
    assert [item.feature_name for item in second] == ["a", "b"]
    assert len(first) == len(specs)
    assert {item.decision for item in first} == {"exclude", "keep"}


def test_known_leakage_and_review_fields_are_not_kept_in_compact_sql() -> None:
    specs = [
        _spec("risk", "static_cb_0", "riskassesment_940T"),
        _spec("approved", "static_0", "income_001A"),
    ]
    catalog = {
        ("static_cb_0", "riskassesment_940T"): {
            "semantic_category": "ambiguous",
            "column_name": "riskassesment_940T",
        },
        ("static_0", "income_001A"): {"semantic_category": "amount", "column_name": "income_001A"},
    }
    smoke = {
        "risk": {"non_null_count": 10, "distinct_count": 2},
        "approved": {"non_null_count": 10, "distinct_count": 2},
    }

    decisions = select_features(specs, catalog, smoke)
    compact_specs = kept_specs(specs, decisions)
    sql = build_feature_sql(compact_specs, output_table="compact", smoke_limit=10)

    assert "riskassesment_940T" not in sql
    assert "income_001A" in sql
    assert all(decision.reason for decision in decisions)


def test_all_retained_features_have_lineage() -> None:
    decisions = [
        FeatureSelectionDecision(
            feature_name="x",
            feature_family="family",
            source_table="table",
            source_columns=("column",),
            transformation="mean",
            decision="keep",
            reason="ok",
            leakage_status="reviewed",
        )
    ]

    assert all(
        item.source_table and item.source_columns
        for item in decisions
        if item.decision == "keep"
    )


def test_ambiguous_review_map_covers_known_fields() -> None:
    assert AMBIGUOUS_SOURCE_DECISIONS[("static_cb_0", "riskassesment_940T")][0] == "EXCLUDE"
    assert AMBIGUOUS_SOURCE_DECISIONS[("other_1", "amtdepositbalance_4809441A")][0] == "KEEP"
