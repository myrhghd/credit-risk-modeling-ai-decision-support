"""Reusable credit-risk model evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def gini_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    positives = y_sorted.sum()
    negatives = len(y_sorted) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    cum_pos = np.cumsum(y_sorted) / positives
    cum_neg = np.cumsum(1 - y_sorted) / negatives
    return float(np.max(np.abs(cum_pos - cum_neg)))


def calibration_intercept_slope(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    clipped = np.clip(np.asarray(y_score), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=200)
    model.fit(logits, y_true)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0][0]),
    }


def probability_summary(y_score: np.ndarray) -> dict[str, float]:
    values = np.asarray(y_score)
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def decile_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    total_rows = len(y_sorted)
    total_positives = int(y_sorted.sum())
    rows: list[dict[str, Any]] = []
    cumulative_rows = 0
    cumulative_positives = 0
    baseline_rate = total_positives / total_rows if total_rows else float("nan")
    for idx, indices in enumerate(np.array_split(np.arange(total_rows), n_bins), start=1):
        if len(indices) == 0:
            continue
        positives = int(y_sorted[indices].sum())
        row_count = int(len(indices))
        cumulative_rows += row_count
        cumulative_positives += positives
        default_rate = positives / row_count
        rows.append(
            {
                "decile": idx,
                "rows": row_count,
                "positive_cases": positives,
                "default_rate": default_rate,
                "score_min": float(score_sorted[indices].min()),
                "score_max": float(score_sorted[indices].max()),
                "cumulative_rows": cumulative_rows,
                "cumulative_row_pct": cumulative_rows / total_rows,
                "cumulative_positives": cumulative_positives,
                "cumulative_positive_pct": (
                    cumulative_positives / total_positives if total_positives else float("nan")
                ),
                "lift": default_rate / baseline_rate if baseline_rate else float("nan"),
                "cumulative_lift": (
                    (cumulative_positives / cumulative_rows) / baseline_rate
                    if baseline_rate
                    else float("nan")
                ),
            }
        )
    return rows


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc = float(roc_auc_score(y_true, y_score))
    metrics: dict[str, Any] = {
        "rows": int(len(y_true)),
        "positive_cases": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "roc_auc": auc,
        "gini": gini_from_auc(auc),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "probability_summary": probability_summary(y_score),
        "deciles": decile_table(y_true, y_score),
    }
    metrics.update(calibration_intercept_slope(y_true, y_score))
    return metrics
