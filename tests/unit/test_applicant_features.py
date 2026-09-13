from credit_risk.features.applicant_features import (
    EXCLUDED_LEAKAGE_FIELDS,
    build_feature_registry,
    build_feature_sql,
    deterministic_feature_name,
    estimate_materialized_size_bytes,
    feature_counts_by_family,
)


def _column(
    table: str,
    name: str,
    *,
    pg_type: str = "double precision",
    category: str = "amount",
    description: str = "Amount field.",
    depth: str = "depth 1",
) -> dict[str, object]:
    return {
        "logical_table": table,
        "column_name": name,
        "description": description,
        "postgres_type": pg_type,
        "source_type": "double",
        "table_depth": depth,
        "is_key_or_grouping_field": name in {"case_id", "num_group1", "num_group2"},
        "is_target": name == "target",
        "is_metadata_field": name in {"WEEK_NUM", "MONTH", "date_decision"},
        "semantic_category": category,
        "static_or_repeated": "static" if depth in {"base", "depth 0"} else "repeated",
        "representative_values": [],
        "modeling_role": "candidate predictor",
        "leakage_candidate": (table, name) in EXCLUDED_LEAKAGE_FIELDS,
        "notes": "",
    }


def _catalog() -> dict[str, object]:
    return {
        "tables": [
            {"logical_table": "base", "logical_family": "base"},
            {"logical_table": "static_0", "logical_family": "static"},
            {"logical_table": "applprev_1", "logical_family": "previous_application"},
        ],
        "columns": [
            _column("base", "case_id", category="key/grouping", depth="base"),
            _column("base", "target", category="target", depth="base"),
            _column("static_0", "case_id", category="key/grouping", depth="depth 0"),
            _column(
                "static_0",
                "lastapprdate_640D",
                pg_type="date",
                category="date/days",
                description="Date of previous approval.",
                depth="depth 0",
            ),
            _column(
                "static_0",
                "income_001A",
                pg_type="double precision",
                category="amount",
                description="Income amount.",
                depth="depth 0",
            ),
            _column("applprev_1", "case_id", category="key/grouping"),
            _column("applprev_1", "num_group1", category="key/grouping"),
            _column(
                "applprev_1",
                "annuity_853A",
                pg_type="double precision",
                category="amount",
                description="Monthly annuity for previous applications.",
            ),
            _column(
                "applprev_1",
                "status_219L",
                pg_type="text",
                category="categorical status",
                description="Previous application status.",
            ),
        ],
    }


def test_deterministic_feature_name() -> None:
    assert deterministic_feature_name("static_0", "income_001A", "direct") == (
        "static_0__income_001a__direct"
    )


def test_registry_excludes_target_and_review_leakage_and_has_lineage() -> None:
    specs = build_feature_registry(_catalog())
    names = {spec.feature_name for spec in specs}
    sources = {(spec.source_table, spec.source_columns[0]) for spec in specs}

    assert ("base", "target") not in sources
    assert ("static_0", "lastapprdate_640D") not in sources
    assert "static_0__income_001a__direct" in names
    assert all(spec.description for spec in specs)
    assert all(spec.feature_family for spec in specs)
    assert all(spec.leakage_review_status for spec in specs)


def test_registry_generates_selective_repeated_aggregations() -> None:
    specs = build_feature_registry(_catalog())
    transforms = {
        spec.transformation for spec in specs if spec.source_columns == ("annuity_853A",)
    }

    assert transforms == {"min", "mean", "max", "std", "sum"}
    assert any(
        spec.source_columns == ("status_219L",) and spec.transformation == "n_distinct"
        for spec in specs
    )


def test_feature_sql_has_one_row_per_case_anchor_and_expected_aggregates() -> None:
    specs = build_feature_registry(_catalog())
    sql = build_feature_sql(specs, output_table="smoke", smoke_limit=10)

    assert "FROM raw.base" in sql
    assert "LIMIT 10" in sql
    assert "CREATE TABLE features.\"smoke\" AS" in sql
    assert "GROUP BY r.case_id" in sql
    assert "count(*)::double precision" in sql
    assert "sum(r.\"annuity_853A\")::double precision" in sql


def test_feature_counts_and_storage_estimate() -> None:
    specs = build_feature_registry(_catalog())
    counts = feature_counts_by_family(specs)

    assert counts["static applicant/bureau snapshots"] == 2
    assert counts["previous applications"] == 7
    assert estimate_materialized_size_bytes(10, 5) == 540
