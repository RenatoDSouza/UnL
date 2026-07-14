"""Membership inference attack (loss-threshold, AUC-based).

The attacker observes the model's per-sample loss and tries to distinguish
*members* (samples the model was trained on) from *non-members* (held-out
samples of the same distribution that were never seen during training). This is
the standard Yeom-style loss-threshold attack; we report the ROC-AUC of the
membership classifier, which is threshold-free and interpretable:

* ``AUC = 0.5`` -> no membership leakage (ideal for a forgotten client);
* ``AUC -> 1.0`` -> the model memorised its members.

An effective unlearning method should drive the AUC of the forgotten client
back towards 0.5 without harming the retained clients.
"""

from __future__ import annotations

import numpy as np

from qfl.quantum.model import QuantumClassifier


def per_sample_loss(model: QuantumClassifier, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared error between P(class=1) and the label, per sample."""
    if len(x) == 0:
        return np.zeros(0, dtype=float)
    probs = np.asarray(model.predict_proba(x))[:, 1]
    return (probs - y.astype(float)) ** 2


def membership_inference_auc(
    model: QuantumClassifier,
    x_members: np.ndarray,
    y_members: np.ndarray,
    x_nonmembers: np.ndarray,
    y_nonmembers: np.ndarray,
) -> float:
    """ROC-AUC of a loss-threshold membership inference attack.

    Members are expected to have *lower* loss, so we score samples by the
    negative loss (higher score = more member-like). Returns 0.5 when either
    group is empty or degenerate.
    """
    if len(x_members) == 0 or len(x_nonmembers) == 0:
        return 0.5
    loss_m = per_sample_loss(model, x_members, y_members)
    loss_n = per_sample_loss(model, x_nonmembers, y_nonmembers)
    scores = np.concatenate([-loss_m, -loss_n])
    labels = np.concatenate([np.ones(len(loss_m)), np.zeros(len(loss_n))])
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except Exception:
        # Fallback: probability that a random member scores above a random
        # non-member (equivalent to AUC, computed directly).
        greater = (loss_m[:, None] < loss_n[None, :]).mean()
        ties = (loss_m[:, None] == loss_n[None, :]).mean()
        return float(greater + 0.5 * ties)
