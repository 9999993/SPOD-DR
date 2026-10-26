from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def compute_ece(y_true: np.ndarray, prob_pos: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    predicted_risk = np.asarray(prob_pos, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)

    ece = 0.0
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (predicted_risk >= left) & (predicted_risk <= right)
        else:
            mask = (predicted_risk >= left) & (predicted_risk < right)

        if not np.any(mask):
            continue

        event_rate = labels[mask].mean()
        confidence = predicted_risk[mask].mean()
        ece += abs(float(event_rate) - float(confidence)) * (mask.sum() / len(predicted_risk))
    return float(ece)


def compute_brier_score(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    prob_pos = np.asarray(prob_pos, dtype=np.float64)
    return float(np.mean((prob_pos - y_true) ** 2))


def compute_auc_rank(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(prob_pos, dtype=np.float64)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = float(ranks[positives].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_average_precision(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(prob_pos, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    precision = tp / np.arange(1, len(sorted_labels) + 1)
    return float(precision[sorted_labels == 1].sum() / n_pos)


def calibration_slope_intercept(y_true: np.ndarray, prob_pos: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(y_true, dtype=np.float64)
    probs = np.clip(np.asarray(prob_pos, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if labels.size < 3 or np.unique(labels).size < 2:
        return float("nan"), float("nan")

    logits = np.log(probs / (1.0 - probs))
    if float(np.std(logits)) < 1e-8:
        return float("nan"), float("nan")

    design = np.column_stack([np.ones_like(logits), logits])
    coef = np.zeros(2, dtype=np.float64)
    for _ in range(50):
        linear = design @ coef
        fitted = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
        weights = np.clip(fitted * (1.0 - fitted), 1e-6, None)
        gradient = design.T @ (labels - fitted)
        hessian = -(design.T * weights) @ design
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        coef -= step
        if float(np.max(np.abs(step))) < 1e-6:
            break

    return float(coef[1]), float(coef[0])


def reliability_bins(y_true: np.ndarray, prob_pos: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    labels = np.asarray(y_true, dtype=np.int64)
    confidences = np.asarray(prob_pos, dtype=np.float64)
    rows: list[dict[str, float]] = []

    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": float(idx),
                    "left": float(left),
                    "right": float(right),
                    "count": 0.0,
                    "accuracy": float("nan"),
                    "observed_positive_fraction": float("nan"),
                    "confidence": float("nan"),
                }
            )
            continue

        observed_positive_fraction = float(labels[mask].mean())
        rows.append(
            {
                "bin": float(idx),
                "left": float(left),
                "right": float(right),
                "count": float(count),
                "accuracy": observed_positive_fraction,
                "observed_positive_fraction": observed_positive_fraction,
                "confidence": float(confidences[mask].mean()),
            }
        )

    return rows


def _finite_stats(values: np.ndarray | None) -> dict[str, float]:
    if values is None:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def compute_binary_metrics(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    logits_pos: np.ndarray | None = None,
) -> dict[str, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
    except Exception:
        average_precision_score = None
        roc_auc_score = None

    y_true = np.asarray(y_true, dtype=np.int64)
    prob_pos = np.asarray(prob_pos, dtype=np.float64)
    nonfinite_prob_count = int((~np.isfinite(prob_pos)).sum())
    if nonfinite_prob_count > 0:
        prob_pos = np.nan_to_num(prob_pos, nan=0.5, posinf=1.0, neginf=0.0)
    prob_pos = np.clip(prob_pos, 0.0, 1.0)
    pred = (prob_pos >= 0.5).astype(np.int64)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(len(y_true), 1)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    auc = float("nan")
    pr_auc = float("nan")
    if len(np.unique(y_true)) > 1:
        if roc_auc_score is not None:
            auc = float(roc_auc_score(y_true, prob_pos))
        else:
            auc = compute_auc_rank(y_true, prob_pos)
        if average_precision_score is not None:
            pr_auc = float(average_precision_score(y_true, prob_pos))
        else:
            pr_auc = compute_average_precision(y_true, prob_pos)
    calibration_slope, calibration_intercept = calibration_slope_intercept(y_true, prob_pos)
    logit_stats = _finite_stats(logits_pos)

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "f1": float(f1),
        "ece": compute_ece(y_true, prob_pos),
        "brier": compute_brier_score(y_true, prob_pos),
        "nonfinite_prob_count": float(nonfinite_prob_count),
        "pred_pos_frac": float(pred.mean()) if len(pred) else float("nan"),
        "prob_mean": float(prob_pos.mean()) if len(prob_pos) else float("nan"),
        "prob_std": float(prob_pos.std()) if len(prob_pos) else float("nan"),
        "prob_min": float(prob_pos.min()) if len(prob_pos) else float("nan"),
        "prob_max": float(prob_pos.max()) if len(prob_pos) else float("nan"),
        "logit_mean": logit_stats["mean"],
        "logit_std": logit_stats["std"],
        "logit_min": logit_stats["min"],
        "logit_max": logit_stats["max"],
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
    }


def aggregate_prediction_records(
    records: Iterable[dict[str, object]],
    by: str = "lesion_id",
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (record["patient_id"], record[by]) if by == "lesion_id" else (record[by],)
        grouped[key].append(record)

    aggregated: list[dict[str, object]] = []
    for group in grouped.values():
        probs = np.asarray([float(item["prob_pos"]) for item in group], dtype=np.float64)
        finite_probs = probs[np.isfinite(probs)]
        prob_pos = float(finite_probs.mean()) if finite_probs.size > 0 else 0.5
        logits = np.asarray(
            [float(item["logit_pos"]) for item in group if item.get("logit_pos") is not None],
            dtype=np.float64,
        )
        finite_logits = logits[np.isfinite(logits)] if logits.size > 0 else np.asarray([], dtype=np.float64)
        logit_pos = float(finite_logits.mean()) if finite_logits.size > 0 else None
        uncertainties = [
            float(item["uncertainty"])
            for item in group
            if item["uncertainty"] is not None and np.isfinite(float(item["uncertainty"]))
        ]
        evidence_strengths = [
            float(item["evidence_strength"])
            for item in group
            if item.get("evidence_strength") is not None and np.isfinite(float(item["evidence_strength"]))
        ]
        alpha_strengths = [
            float(item["alpha_strength"])
            for item in group
            if item.get("alpha_strength") is not None and np.isfinite(float(item["alpha_strength"]))
        ]
        aggregated.append(
            {
                "patient_id": group[0]["patient_id"],
                "lesion_id": group[0].get("lesion_id"),
                "label": int(group[0]["label"]),
                "prob_pos": prob_pos,
                "logit_pos": logit_pos,
                "uncertainty": float(np.mean(uncertainties)) if uncertainties else None,
                "evidence_strength": float(np.mean(evidence_strengths)) if evidence_strengths else None,
                "alpha_strength": float(np.mean(alpha_strengths)) if alpha_strengths else None,
                "num_slices": len(group),
                "nonfinite_slice_probs": int((~np.isfinite(probs)).sum()),
            }
        )
    return aggregated


def save_prediction_records(path: str | Path, records: Iterable[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
