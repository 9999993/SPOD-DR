from __future__ import annotations

from typing import Any

import numpy as np

from calibmamba.engine.evaluate import compute_binary_metrics


def _macro_f1_from_confusion(confusion: np.ndarray) -> float:
    scores: list[float] = []
    for grade in range(confusion.shape[0]):
        tp = float(confusion[grade, grade])
        fp = float(confusion[:, grade].sum() - tp)
        fn = float(confusion[grade, :].sum() - tp)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if precision + recall <= 0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else float("nan")


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_grades: int = 5) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return float("nan")

    confusion = np.zeros((num_grades, num_grades), dtype=np.float64)
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < num_grades and 0 <= pred < num_grades:
            confusion[truth, pred] += 1.0

    hist_true = confusion.sum(axis=1)
    hist_pred = confusion.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(confusion.sum(), 1.0)
    weights = np.zeros_like(confusion)
    denom = max((num_grades - 1) ** 2, 1)
    for i in range(num_grades):
        for j in range(num_grades):
            weights[i, j] = ((i - j) ** 2) / denom

    observed = float((weights * confusion).sum())
    expected_score = float((weights * expected).sum())
    if expected_score <= 0:
        return float("nan")
    return float(1.0 - observed / expected_score)


def per_grade_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, num_grades: int = 5) -> dict[str, float]:
    result: dict[str, float] = {}
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    for grade in range(num_grades):
        mask = y_true == grade
        result[str(grade)] = float((y_pred[mask] == grade).mean()) if np.any(mask) else float("nan")
    return result


def compute_ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    confusion = np.zeros((5, 5), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < 5 and 0 <= pred < 5:
            confusion[truth, pred] += 1
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred, num_grades=5),
        "macro_f1": _macro_f1_from_confusion(confusion),
        "ordinal_mae": float(np.abs(y_true - y_pred).mean()) if y_true.size else float("nan"),
        "per_grade_sensitivity": per_grade_sensitivity(y_true, y_pred, num_grades=5),
    }


def summarize_predictions(
    *,
    y_true_binary: np.ndarray,
    prob_pos: np.ndarray,
    logits_pos: np.ndarray | None,
    y_true_grade: np.ndarray,
    pred_grade: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    metrics = compute_binary_metrics(y_true_binary, prob_pos, logits_pos=logits_pos)
    pred_binary = (np.asarray(prob_pos, dtype=np.float64) >= float(threshold)).astype(np.int64)
    tp = int(((pred_binary == 1) & (y_true_binary == 1)).sum())
    tn = int(((pred_binary == 0) & (y_true_binary == 0)).sum())
    fp = int(((pred_binary == 1) & (y_true_binary == 0)).sum())
    fn = int(((pred_binary == 0) & (y_true_binary == 1)).sum())
    metrics.update(
        {
            "threshold": float(threshold),
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "tn": tn,
        }
    )
    metrics.update(compute_ordinal_metrics(y_true_grade, pred_grade))
    return metrics
