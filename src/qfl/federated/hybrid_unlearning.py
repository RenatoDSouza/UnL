"""Hybrid SHAP + QFI unlearning for quantum federated models.

Method summary (the paper's contribution)
------------------------------------------
1. **Shapley attribution.** Treat the variational parameter blocks (one per
   qubit wire) as cooperative players and estimate, by Monte-Carlo sampling of
   coalitions, how much each block contributes to *fitting the forget set*.
   These are genuine Shapley values, not a one-shot ablation.
2. **SHAP mask.** Select the highest-attributed blocks -- the parameters most
   responsible for memorising the client to be forgotten.
3. **QFI-preconditioned ascent.** Perform natural-gradient *ascent* on the
   forget-set loss, restricted to the masked parameters and pre-conditioned by
   the quantum Fisher information matrix, until the forget performance falls
   back to the uninformed (retrain-equivalent) baseline.

Evaluation reports forget/retain accuracy and loss, membership-inference AUC
(members = forget-set, non-members = held-out), and the QFI trace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qfl.federated.metrics import evaluate_accuracy, majority_class_rate
from qfl.federated.mia import membership_inference_auc
from qfl.quantum.model import QuantumClassifier


@dataclass(frozen=True)
class UnlearningReport:
    qfi_trace_before: float
    qfi_trace_after: float
    forget_accuracy_before: float
    forget_accuracy_after: float
    forget_loss_before: float
    forget_loss_after: float
    retain_accuracy_before: float
    retain_accuracy_after: float
    retain_loss_before: float
    retain_loss_after: float
    mia_auc_before: float
    mia_auc_after: float
    uninformed_accuracy: float
    unlearning_steps: int
    shap_drop_mean: float


class HybridSHAPQFIUnlearner:
    def __init__(self, model: QuantumClassifier):
        self.model = model

    # ------------------------------------------------------------------ utils
    def _flatten(self, weights=None) -> np.ndarray:
        values = self.model.weights if weights is None else weights
        return np.asarray(values, dtype=float).reshape(-1)

    def _assign_weights(self, flat_weights: np.ndarray) -> None:
        self.model.weights = pnp.array(flat_weights.reshape(self.model.weights.shape), requires_grad=True)

    def wire_parameter_groups(self) -> list[np.ndarray]:
        """Group flattened parameters by qubit wire.

        Weights have shape ``(num_layers, num_wires, 3)``; the flattened index of
        entry ``(l, w, r)`` is ``l*num_wires*3 + w*3 + r``. Grouping by wire gives
        ``num_wires`` interpretable Shapley players.
        """
        num_layers, num_wires, rot = self.model.weights.shape
        groups: list[list[int]] = [[] for _ in range(num_wires)]
        for layer in range(num_layers):
            for wire in range(num_wires):
                base = layer * num_wires * rot + wire * rot
                groups[wire].extend(range(base, base + rot))
        return [np.array(g, dtype=int) for g in groups]

    # -------------------------------------------------------------- attribution
    def shapley_attribution(
        self,
        x_ref: np.ndarray,
        y_ref: np.ndarray,
        groups: list[np.ndarray] | None = None,
        num_permutations: int = 16,
        seed: int = 0,
    ) -> np.ndarray:
        """Monte-Carlo Shapley values of parameter groups for fitting ``x_ref``.

        The value of a coalition ``S`` is the negative forget-set loss obtained
        when only the parameters in ``S`` are active (others set to their neutral
        value, 0). The Shapley value of a group is its average marginal
        contribution over random player orderings. Returns a per-parameter array
        where every parameter carries its group's Shapley value.
        """
        base = self._flatten()
        if len(x_ref) == 0:
            return np.zeros(base.size, dtype=float)
        groups = groups or self.wire_parameter_groups()
        rng = np.random.default_rng(seed)

        def value(active: np.ndarray) -> float:
            self._assign_weights(base * active)
            return -float(self.model.loss(x_ref, y_ref))

        phi_group = np.zeros(len(groups), dtype=float)
        for _ in range(max(1, num_permutations)):
            active = np.zeros(base.size, dtype=float)
            prev_v = value(active)
            for g in rng.permutation(len(groups)):
                active[groups[g]] = 1.0
                v = value(active)
                phi_group[g] += v - prev_v
                prev_v = v
        phi_group /= max(1, num_permutations)

        self._assign_weights(base)
        attributions = np.zeros(base.size, dtype=float)
        for g, group in enumerate(groups):
            attributions[group] = phi_group[g]
        return attributions

    def shap_mask(self, shap_values: np.ndarray, threshold_quantile: float = 0.5) -> np.ndarray:
        if shap_values.size == 0:
            return shap_values
        threshold = float(np.quantile(shap_values, threshold_quantile))
        return (shap_values >= threshold).astype(float)

    # --------------------------------------------------------------- unlearning
    def ascent(
        self,
        x_forget: np.ndarray,
        y_forget: np.ndarray,
        mask: np.ndarray,
        use_qfi: bool,
        lr: float = 0.4,
        max_steps: int = 12,
        damping: float = 1e-3,
        target_accuracy: float | None = None,
    ) -> int:
        """(QFI-preconditioned) gradient *ascent* on the forget-set loss.

        Ascends along the masked gradient, optionally pre-conditioned by the
        inverse QFI, and stops early once forget accuracy drops to
        ``target_accuracy`` (the uninformed/retrain-equivalent level), avoiding
        over-forgetting. Returns the number of steps actually taken.
        """
        if len(x_forget) == 0 or len(y_forget) == 0:
            return 0

        def loss_fn(flat_weights):
            # Direct assignment keeps the autograd tape connected (see client.py).
            self.model.weights = flat_weights.reshape(self.model.weights.shape)
            return self.model.loss(x_forget, y_forget)

        qfi_inv = None
        if use_qfi:
            qfi = np.asarray(self.model.qfi_matrix(x_forget))
            qfi_inv = np.linalg.inv(qfi + damping * np.eye(qfi.shape[0]))

        weights = pnp.array(self._flatten(), requires_grad=True)
        steps = 0
        for _ in range(max(1, max_steps)):
            grad = np.asarray(qml.grad(loss_fn)(weights)) * mask
            if not grad.any():
                break
            direction = qfi_inv @ grad if qfi_inv is not None else grad
            weights = pnp.array(np.asarray(weights) + lr * direction, requires_grad=True)
            self._assign_weights(np.asarray(weights))
            steps += 1
            if target_accuracy is not None and evaluate_accuracy(self.model, x_forget, y_forget) <= target_accuracy:
                break
        self._assign_weights(np.asarray(weights))
        return steps

    # --------------------------------------------------------------------- run
    def run(
        self,
        x_forget: np.ndarray,
        y_forget: np.ndarray,
        x_retain: np.ndarray,
        y_retain: np.ndarray,
        x_forget_eval: np.ndarray | None = None,
        y_forget_eval: np.ndarray | None = None,
        num_shap_permutations: int = 16,
        mask_quantile: float = 0.5,
        lr: float = 0.4,
        max_steps: int = 12,
        mode: str = "shap_qfi",
        seed: int = 0,
    ) -> tuple[UnlearningReport, dict[str, np.ndarray]]:
        x_forget_eval = np.empty((0, self.model.num_wires)) if x_forget_eval is None else x_forget_eval
        y_forget_eval = np.empty((0,)) if y_forget_eval is None else y_forget_eval

        def _loss(x, y):
            return float(self.model.loss(x, y)) if len(x) else 0.0

        def _mia():
            return membership_inference_auc(self.model, x_forget, y_forget, x_forget_eval, y_forget_eval)

        uninformed = majority_class_rate(y_forget if len(y_forget) else y_retain)

        qfi_trace_before = self.model.qfi_trace(x_forget if len(x_forget) else x_retain[:1])
        forget_accuracy_before = evaluate_accuracy(self.model, x_forget, y_forget)
        retain_accuracy_before = evaluate_accuracy(self.model, x_retain, y_retain)
        forget_loss_before = _loss(x_forget, y_forget)
        retain_loss_before = _loss(x_retain, y_retain)
        mia_before = _mia()

        shap_before = self.shapley_attribution(x_forget, y_forget, num_permutations=num_shap_permutations, seed=seed)

        if len(x_forget) == 0 or len(y_forget) == 0:
            mode = "no_unlearning"

        steps = 0
        if mode == "no_unlearning":
            mask = np.zeros_like(shap_before)
        elif mode == "qfi_only":
            mask = np.ones_like(shap_before)
            steps = self.ascent(x_forget, y_forget, mask, use_qfi=True, lr=lr, max_steps=max_steps, target_accuracy=uninformed)
        elif mode == "shap_only":
            mask = self.shap_mask(shap_before, threshold_quantile=mask_quantile)
            steps = self.ascent(x_forget, y_forget, mask, use_qfi=False, lr=lr, max_steps=max_steps, target_accuracy=uninformed)
        elif mode == "shap_qfi":
            mask = self.shap_mask(shap_before, threshold_quantile=mask_quantile)
            steps = self.ascent(x_forget, y_forget, mask, use_qfi=True, lr=lr, max_steps=max_steps, target_accuracy=uninformed)
        else:
            raise ValueError(f"Unknown unlearning mode: {mode}")

        qfi_trace_after = self.model.qfi_trace(x_forget if len(x_forget) else x_retain[:1])
        forget_accuracy_after = evaluate_accuracy(self.model, x_forget, y_forget)
        retain_accuracy_after = evaluate_accuracy(self.model, x_retain, y_retain)
        forget_loss_after = _loss(x_forget, y_forget)
        retain_loss_after = _loss(x_retain, y_retain)
        mia_after = _mia()
        shap_after = self.shapley_attribution(x_forget, y_forget, num_permutations=num_shap_permutations, seed=seed)

        report = UnlearningReport(
            qfi_trace_before=qfi_trace_before,
            qfi_trace_after=qfi_trace_after,
            forget_accuracy_before=forget_accuracy_before,
            forget_accuracy_after=forget_accuracy_after,
            forget_loss_before=forget_loss_before,
            forget_loss_after=forget_loss_after,
            retain_accuracy_before=retain_accuracy_before,
            retain_accuracy_after=retain_accuracy_after,
            retain_loss_before=retain_loss_before,
            retain_loss_after=retain_loss_after,
            mia_auc_before=mia_before,
            mia_auc_after=mia_after,
            uninformed_accuracy=uninformed,
            unlearning_steps=steps,
            shap_drop_mean=float(np.mean(np.abs(shap_before) - np.abs(shap_after))) if shap_before.size else 0.0,
        )
        return report, {"shap_before": shap_before, "shap_after": shap_after, "mask": mask}
