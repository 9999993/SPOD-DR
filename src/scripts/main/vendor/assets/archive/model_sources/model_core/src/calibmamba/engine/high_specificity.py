from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def binary_metrics_at_threshold(
    labels: Sequence[int | float],
    probs: Sequence[int | float],
    threshold: float,
) -> dict[str, float]:
    y_true = [int(float(item)) for item in labels]
    pred = [1 if float(prob) >= threshold else 0 for prob in probs]
    tp = sum(1 for label, item in zip(y_true, pred) if label == 1 and item == 1)
    tn = sum(1 for label, item in zip(y_true, pred) if label == 0 and item == 0)
    fp = sum(1 for label, item in zip(y_true, pred) if label == 0 and item == 1)
    fn = sum(1 for label, item in zip(y_true, pred) if label == 1 and item == 0)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def metrics_at_min_specificity(
    labels: Sequence[int | float],
    probs: Sequence[int | float],
    target_specificity: float = 0.85,
) -> dict[str, float]:
    target_specificity = min(max(float(target_specificity), 0.0), 1.0)
    scores = [float(item) for item in probs]
    candidates = [1.0 + 1e-12, *sorted(set(scores), reverse=True), -1e-12]
    feasible: list[dict[str, float]] = []
    for threshold in candidates:
        metrics = binary_metrics_at_threshold(labels, scores, threshold)
        if metrics["specificity"] >= target_specificity:
            feasible.append(metrics)
    if not feasible:
        return binary_metrics_at_threshold(labels, scores, 1.0 + 1e-12)
    feasible.sort(
        key=lambda item: (
            item["sensitivity"],
            item["f1"],
            item["balanced_accuracy"],
            -item["threshold"],
        ),
        reverse=True,
    )
    return feasible[0]


def high_specificity_checkpoint_score(
    metrics: Mapping[str, Any],
    *,
    target_specificity: float = 0.85,
    weight_auc: float = 1.0,
    weight_f1: float = 0.5,
    weight_sens: float = 1.0,
    weight_ece: float = 0.5,
    weight_brier: float = 0.5,
) -> float:
    spec_suffix = int(round(float(target_specificity) * 100.0))
    sens_key = f"sensitivity_at_spec{spec_suffix}"
    f1_key = f"f1_at_spec{spec_suffix}"
    sens = _safe_float(metrics.get(sens_key, metrics.get("sensitivity_at_target_specificity", 0.0)))
    f1_at_spec = _safe_float(metrics.get(f1_key, metrics.get("f1_at_target_specificity", 0.0)))
    return (
        float(weight_auc) * _safe_float(metrics.get("auc", 0.0))
        + float(weight_f1) * max(_safe_float(metrics.get("f1", 0.0)), f1_at_spec)
        + float(weight_sens) * sens
        - float(weight_ece) * _safe_float(metrics.get("ece", 0.0))
        - float(weight_brier) * _safe_float(metrics.get("brier", 0.0))
    )


def negative_tail_loss(
    logits,
    target,
    *,
    positive_margin: float = 0.65,
    negative_margin: float = 0.25,
):
    import torch
    from torch.nn import functional as F

    if logits is None:
        raise ValueError("negative_tail_loss requires logits.")
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError(f"Expected logits shaped [B,2+], got {tuple(logits.shape)}")
    target = target.to(device=logits.device)
    prob_pos = torch.softmax(logits.float(), dim=-1)[:, 1]
    positives = target == 1
    negatives = target == 0
    if not bool(torch.any(positives)) or not bool(torch.any(negatives)):
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    positive_loss = F.relu(float(positive_margin) - prob_pos[positives]).mean()
    negative_loss = F.relu(prob_pos[negatives] - float(negative_margin)).mean()
    return (positive_loss + negative_loss).to(dtype=logits.dtype)
