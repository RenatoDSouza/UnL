"""Membership inference attack helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
        return art_membership_inference_success_rate(model, x_train, y_train, x_test, y_test)
    except Exception as exc:
        LOGGER.warning("ART indisponível ou incompatível (%s); usando proxy simples para MIA.", exc)
        return proxy_mia_success_rate(model, x_train, y_train, x_test, y_test)


def art_membership_inference_success_rate(
    model: QuantumClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    from art.attacks.inference.membership_inference import MembershipInferenceBlackBox  # type: ignore
    from art.estimators.classification import BlackBoxClassifier  # type: ignore

    estimator = _build_art_blackbox_classifier(model, x_train.shape[1])
    attack = MembershipInferenceBlackBox(estimator, input_type="prediction")

    x_eval = np.concatenate([x_train, x_test], axis=0)
    y_eval = np.concatenate([y_train, y_test], axis=0)
    member_labels = np.concatenate([np.ones(len(x_train), dtype=int), np.zeros(len(x_test), dtype=int)])

    attack.fit(x_train, y_train, x_test, y_test)
    inferred = np.asarray(attack.infer(x_eval, y_eval)).reshape(-1).astype(int)
    if inferred.shape[0] != member_labels.shape[0]:
        raise ValueError("ART returned an unexpected number of predictions for MIA")
    return float(np.mean(inferred == member_labels))


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


@dataclass(frozen=True)
class _BlackBoxAdapter:
    model: QuantumClassifier

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x), dtype=np.float32)


def _build_art_blackbox_classifier(model: QuantumClassifier, num_features: int):
    from art.estimators.classification import BlackBoxClassifier  # type: ignore

    adapter = _BlackBoxAdapter(model=model)
    return BlackBoxClassifier(
        predict=adapter.predict,
        input_shape=(num_features,),
        nb_classes=2,
        clip_values=(0.0, 1.0),
    )


def _mean_correct_confidence(model: QuantumClassifier, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    probs = np.asarray(model.predict_proba(x))
    correct = probs[np.arange(len(y)), y.astype(int)]
    return float(np.mean(correct))
