"""Training and evaluation loops for CalibMamba."""

from calibmamba.engine.evaluate import (
    aggregate_prediction_records,
    compute_binary_metrics,
    compute_ece,
    save_prediction_records,
)
from calibmamba.engine.train_loop import evaluate_one_epoch, train_one_epoch, train_one_epoch_with_teacher

__all__ = [
    "aggregate_prediction_records",
    "compute_binary_metrics",
    "compute_ece",
    "save_prediction_records",
    "evaluate_one_epoch",
    "train_one_epoch",
    "train_one_epoch_with_teacher",
]
