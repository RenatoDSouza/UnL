"""Evaluation metrics for unlearning experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qfl.quantum.model import QuantumClassifier


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def evaluate_accuracy(model: QuantumClassifier, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    probs = np.asarray(model.predict_proba(x))
    preds = (probs[:, 1] >= 0.5).astype(int)
    return accuracy_score(y.astype(int), preds)


def random_baseline_accuracy(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    classes = np.unique(y.astype(int))
    return 1.0 / max(1, len(classes))


def majority_class_rate(y: np.ndarray) -> float:
    """Accuracy of an uninformed constant predictor (the majority class).

    This is the level a model that never saw the data would reach, and the
    target for unlearning: forget accuracy should fall back to it rather than
    overshoot below chance.
    """
    if len(y) == 0:
        return 0.5
    y = y.astype(int)
    p = float(np.mean(y))
    return max(p, 1.0 - p)


@dataclass(frozen=True)
class UnlearningMetrics:
    forget_accuracy: float
    retain_accuracy: float
    mia_success_rate: float
    qfi_trace: float

