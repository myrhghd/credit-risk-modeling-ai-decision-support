import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction import FeatureHasher

from credit_risk.evaluation.metrics import decile_table, gini_from_auc, ks_statistic
from credit_risk.models.logistic_baseline import (
    ModelConfig,
    PredictorSets,
    categorical_tokens,
    infer_predictor_sets,
    iter_split_batches,
    load_keep_predictors,
    resolved_class_weight,
    transform_batch,
    transform_record_batch,
    verify_baseline_inputs,
)


def test_gini_equals_two_auc_minus_one() -> None:
    assert gini_from_auc(0.75) == 0.5


def test_ks_statistic() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    assert round(ks_statistic(y_true, y_score), 6) == 0.5


def test_decile_assignment_and_lift() -> None:
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    rows = decile_table(y_true, y_score)

    assert len(rows) == 10
    assert rows[0]["decile"] == 1
    assert rows[0]["default_rate"] == 1.0
    assert rows[0]["lift"] == 5.0
    assert rows[-1]["cumulative_positive_pct"] == 1.0


def test_load_keep_predictors_excludes_leakage(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {"feature_name": "safe_feature", "decision": "keep"},
                {"feature_name": "base__target__direct", "decision": "exclude"},
            ]
        ),
        encoding="utf-8",
    )

    assert load_keep_predictors(manifest_path) == ["safe_feature"]


def test_infer_predictor_sets_from_parquet_schema(tmp_path: Path) -> None:
    path = tmp_path / "modeling.parquet"
    pl.DataFrame(
        {
            "case_id": [1],
            "target": [0],
            "date_decision": ["2020-01-01"],
            "WEEK_NUM": [1],
            "MONTH": [202001],
            "num_feature": [1.0],
            "cat_feature": ["A"],
            "bool_feature": [True],
        }
    ).write_parquet(path)

    predictors = infer_predictor_sets(path, ["num_feature", "cat_feature", "bool_feature"])

    assert predictors.numeric == ("num_feature",)
    assert predictors.categorical == ("cat_feature",)
    assert predictors.boolean == ("bool_feature",)


def test_split_aware_loading(tmp_path: Path) -> None:
    modeling_path = tmp_path / "modeling.parquet"
    labels_path = tmp_path / "labels.parquet"
    pl.DataFrame(
        {
            "case_id": [1, 2, 3],
            "target": [0, 1, 0],
            "WEEK_NUM": [1, 2, 1],
            "x": [1.0, 2.0, 3.0],
        }
    ).write_parquet(modeling_path)
    pl.DataFrame(
        {"case_id": [1, 2, 3], "WEEK_NUM": [1, 2, 1], "split": ["train", "validation", "train"]}
    ).write_parquet(labels_path)

    batches = list(
        iter_split_batches(
            modeling_path,
            labels_path,
            "train",
            ["x"],
            batch_size=2,
        )
    )

    assert [case_id for batch in batches for case_id in batch["case_id"]] == [1, 3]
    assert [target for batch in batches for target in batch["target"]] == [0, 0]


def test_preprocessing_handles_missing_and_unseen_categories() -> None:
    predictors = PredictorSets(
        numeric=("num_feature",),
        categorical=("cat_feature",),
        boolean=("bool_feature",),
    )
    stats = {"mean": {"num_feature": 2.0}, "scale": {"num_feature": 2.0}}
    hasher = FeatureHasher(n_features=8, input_type="string", alternate_sign=False)
    batch = {
        "case_id": [1, 2],
        "num_feature": [None, 4.0],
        "cat_feature": [None, "previously_unseen"],
        "bool_feature": [None, True],
    }

    tokens = categorical_tokens(batch, predictors.categorical)
    matrix = transform_batch(batch, predictors, stats, hasher)

    assert tokens[0] == ["cat_feature=__MISSING__"]
    assert tokens[1] == ["cat_feature=previously_unseen"]
    assert matrix.shape == (2, 10)
    assert matrix.nnz > 0


def test_optimized_transform_matches_prior_transform_for_fixture() -> None:
    predictors = PredictorSets(
        numeric=("num_feature",),
        categorical=("cat_feature",),
        boolean=("bool_feature",),
    )
    stats = {"mean": {"num_feature": 2.0}, "scale": {"num_feature": 2.0}}
    hasher = FeatureHasher(n_features=8, input_type="string", alternate_sign=False)
    frame = pl.DataFrame(
        {
            "case_id": [1, 2],
            "target": [0, 1],
            "num_feature": [None, 4.0],
            "cat_feature": [None, "A"],
            "bool_feature": [None, True],
        }
    )
    batch = frame.to_arrow().to_batches()[0]

    prior = transform_batch(frame.to_dict(as_series=False), predictors, stats, hasher)
    optimized = transform_record_batch(batch, predictors, stats, hasher)

    assert optimized.getformat() == "csr"
    assert optimized.shape == prior.shape == (2, 10)
    np.testing.assert_allclose(optimized.toarray(), prior.toarray())


def test_predictor_feature_order_is_stable() -> None:
    predictors = PredictorSets(
        numeric=("n1", "n2"),
        categorical=("c1",),
        boolean=("b1",),
    )

    assert predictors.all_predictors == ("n1", "n2", "c1", "b1")


def test_moderate_positive_class_weight_resolution() -> None:
    config = ModelConfig(
        estimator="sklearn.linear_model.SGDClassifier",
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        class_weight=3.0,
        random_seed=1,
        max_epochs=1,
        batch_size=2,
        train_row_limit=2,
    )

    assert resolved_class_weight(Path("unused"), Path("unused"), config) == {0: 1.0, 1: 3.0}


def test_baseline_input_verification(tmp_path: Path) -> None:
    modeling_path = tmp_path / "modeling.parquet"
    labels_path = tmp_path / "labels.parquet"
    pl.DataFrame(
        {
            "case_id": [1, 2, 3],
            "target": [0, 1, 0],
            "date_decision": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "WEEK_NUM": [1, 1, 2],
            "MONTH": [202001, 202001, 202001],
            "x": [1.0, 2.0, 3.0],
        }
    ).write_parquet(modeling_path)
    pl.DataFrame({"case_id": [1, 2, 3], "split": ["train", "validation", "test"]}).write_parquet(
        labels_path
    )

    checks = verify_baseline_inputs(
        modeling_path,
        labels_path,
        PredictorSets(numeric=("x",), categorical=(), boolean=()),
    )

    assert checks["target_values"] == [0, 1]
    assert checks["label_columns_excluded_from_predictors"]
    assert checks["no_split_case_overlap"]
    assert checks["score_column"] == "predict_proba(...)[..., 1]"
