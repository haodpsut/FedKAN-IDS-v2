"""Classification metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f.tolist(),
    }
