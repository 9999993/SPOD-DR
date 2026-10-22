"""Source-only frozen referral threshold selection."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def lower_wilson_bound(successes: int, total: int, confidence: float = 0.90) -> float:
    if total <= 0:
        return 0.0
    # One-sided normal quantile for the registered 90% lower bound.
    z = 1.2815515655446004 if abs(confidence - 0.90) < 1e-12 else 1.6448536269514722
    p = float(successes) / float(total)
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    spread = z * math.sqrt(max(p * (1.0 - p) / total + z * z / (4.0 * total * total), 0.0))
    return max(0.0, (centre - spread) / denom)


def select_lower_wilson_sensitivity_threshold(rows: Sequence[Mapping[str, Any]], target: float = 0.90) -> dict[str, Any]:
    """Select the highest source-only probability threshold with safe recall."""
    values = np.asarray([float(r["prob_pos"]) for r in rows], dtype=np.float64)
    labels = np.asarray([int(r.get("label", int(r["grade"]) >= 2)) for r in rows], dtype=np.int64)
    candidates = np.unique(np.concatenate(([0.0, 1.0], values)))
    curve: list[dict[str, float | int | bool]] = []
    valid: list[float] = []
    total = int(labels.sum())
    for threshold in candidates:
        pred = values >= float(threshold)
        successes = int((pred & (labels == 1)).sum())
        observed = successes / total if total else 0.0
        lower = lower_wilson_bound(successes, total)
        item = {"threshold": float(threshold), "successes": successes, "total_positive": total, "sensitivity": float(observed), "lower_90": float(lower), "valid": bool(lower >= target)}
        curve.append(item)
        if lower >= target:
            valid.append(float(threshold))
    selected = max(valid) if valid else 0.0
    return {"threshold": float(selected), "target_sensitivity": float(target), "confidence": 0.90, "source_rows": len(rows), "source_positive": total, "curve": curve, "external_rows_read": 0}


__all__ = ["lower_wilson_bound", "select_lower_wilson_sensitivity_threshold"]
