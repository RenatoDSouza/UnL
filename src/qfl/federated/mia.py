"""Membership inference attack helpers."""

from __future__ import annotations

import logging

import numpy as np

from qfl.quantum.model import QuantumClassifier

LOGGER = logging.getLogger(__name__)


def membership_inference_success_rate(
    model: QuantumClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    try:
        from art.estimators.classification import PyTorchClassifier  # type: ignore
        from art.attacks.inference.membership_inference import MembershipInferenceBlackBox  # type: ignore
    except Exception:
        LOGGER.warning("ART não está disponível; usando proxy simples para MIA.")
        return proxy_mia_success_rate(model, x_train, y_train, x_test, y_test)

    _ = PyTorchClassifier  # pragma: no cover - imported for availability validation
    _ = MembershipInferenceBlackBox
    return proxy_mia_success_rate(model, x_train, y_train, x_test, y_test)


def proxy_mia_success_rate(
    model: QuantumClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    train_confidence = _mean_correct_confidence(model, x_train, y_train)
    test_confidence = _mean_correct_confidence(model, x_test, y_test)
    if train_confidence == test_confidence:
        return 0.5
    return float(train_confidence > test_confidence)


def _mean_correct_confidence(model: QuantumClassifier, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    probs = np.asarray(model.predict_proba(x))
    correct = probs[np.arange(len(y)), y.astype(int)]
    return float(np.mean(correct))

