from __future__ import annotations

import numpy as np


def select_threshold_for_min_sensitivity(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    *,
    target_sensitivity: float = 0.90,
) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob_pos = np.asarray(prob_pos, dtype=np.float64)
    candidates = np.unique(np.clip(prob_pos, 0.0, 1.0))
    best_threshold = 0.5
    best_specificity = -1.0
    for threshold in candidates:
        pred = prob_pos >= threshold
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        if sensitivity >= target_sensitivity and specificity > best_specificity:
            best_threshold = float(threshold)
            best_specificity = float(specificity)
    return float(best_threshold)


def select_youden_threshold(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob_pos = np.asarray(prob_pos, dtype=np.float64)
    candidates = np.unique(np.clip(prob_pos, 0.0, 1.0))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        pred = prob_pos >= threshold
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        youden = sensitivity + specificity - 1.0
        if youden > best_score:
            best_score = float(youden)
            best_threshold = float(threshold)
    return float(best_threshold)
