"""Hybrid SHAP + QFI unlearning pipeline for quantum federated models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qfl.federated.metrics import evaluate_accuracy, random_baseline_accuracy
from qfl.federated.mia import membership_inference_success_rate
from qfl.quantum.model import QuantumClassifier


@dataclass(frozen=True)
class UnlearningReport:
    qfi_trace_before: float
    qfi_trace_after: float
    forget_accuracy_before: float
    forget_accuracy_after: float
    retain_accuracy_before: float
    retain_accuracy_after: float
    mia_success_rate_before: float
    mia_success_rate_after: float
    random_baseline_accuracy: float
    shap_drop_mean: float


class HybridSHAPQFIUnlearner:
    def __init__(self, model: QuantumClassifier):
        self.model = model

    def _flatten(self, weights=None) -> np.ndarray:
        values = self.model.weights if weights is None else weights
        return np.asarray(values, dtype=float).reshape(-1)

    def _assign_weights(self, flat_weights: np.ndarray) -> None:
        self.model.weights = pnp.array(flat_weights.reshape(self.model.weights.shape), requires_grad=True)

    def shap_attribution(
        self,
        x_ref: np.ndarray,
        y_ref: np.ndarray,
        num_permutations: int = 8,
        parameter_groups: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        if len(x_ref) == 0:
            return np.zeros(self.model.weights.size, dtype=float)
        base_weights = self._flatten()
        baseline_loss = float(self.model.loss(x_ref, y_ref))
        attributions = np.zeros(base_weights.size, dtype=float)
        indices = parameter_groups or [np.array([i]) for i in range(base_weights.size)]

        for group in indices:
            estimates = []
            for _ in range(max(1, num_permutations)):
                perturbed = base_weights.copy()
                perturbed[group] = 0.0
                self._assign_weights(perturbed)
                estimates.append(abs(float(self.model.loss(x_ref, y_ref)) - baseline_loss))
            attributions[group] = float(np.mean(estimates))

        self._assign_weights(base_weights)
        return attributions

    def shap_mask(self, shap_values: np.ndarray, threshold_quantile: float = 0.75) -> np.ndarray:
        if shap_values.size == 0:
            return shap_values
        threshold = float(np.quantile(shap_values, threshold_quantile))
        return (shap_values >= threshold).astype(float)

    def qfi_step(
        self,
        x_forget: np.ndarray,
        y_forget: np.ndarray,
        shap_mask: np.ndarray,
        lr: float = 0.05,
        damping: float = 1e-6,
    ) -> np.ndarray:
        if len(x_forget) == 0 or len(y_forget) == 0:
            return self._flatten()
        base_weights = self._flatten()

        def loss_fn(flat_weights):
            self._assign_weights(np.asarray(flat_weights))
            return self.model.loss(x_forget, y_forget)

        grad = np.asarray(qml.grad(loss_fn)(base_weights))
        if grad.size == 0:
            return base_weights
        qfi_inputs = x_forget if len(x_forget) else np.zeros((1, self.model.num_wires), dtype=float)
        qfi = np.asarray(self.model.qfi_matrix(qfi_inputs))
        qfi = qfi + damping * np.eye(qfi.shape[0])
        masked_grad = grad * shap_mask
        delta = np.linalg.solve(qfi, masked_grad)
        new_weights = base_weights - lr * delta
        self._assign_weights(new_weights)
        return new_weights

    def run(
        self,
        x_forget: np.ndarray,
        y_forget: np.ndarray,
        x_retain: np.ndarray,
        y_retain: np.ndarray,
        num_shap_permutations: int = 8,
        mask_quantile: float = 0.75,
        lr: float = 0.05,
        mode: str = "shap_qfi",
    ) -> tuple[UnlearningReport, dict[str, np.ndarray]]:
        qfi_trace_before = self.model.qfi_trace(x_forget if len(x_forget) else x_retain[:1])
        forget_accuracy_before = evaluate_accuracy(self.model, x_forget, y_forget)
        retain_accuracy_before = evaluate_accuracy(self.model, x_retain, y_retain)
        mia_before = membership_inference_success_rate(self.model, x_forget, y_forget, x_retain, y_retain)

        shap_before = self.shap_attribution(x_forget, y_forget, num_permutations=num_shap_permutations)
        mask = self.shap_mask(shap_before, threshold_quantile=mask_quantile)

        if len(x_forget) == 0 or len(y_forget) == 0:
            mode = "no_unlearning"

        if mode == "no_unlearning":
            pass
        elif mode == "qfi_only":
            self.qfi_step(x_forget, y_forget, np.ones_like(mask), lr=lr)
        elif mode == "shap_only":
            self._masked_gradient_step(x_forget, y_forget, mask, lr=lr)
        elif mode == "shap_qfi":
            self.qfi_step(x_forget, y_forget, mask, lr=lr)
        else:
            raise ValueError(f"Unknown unlearning mode: {mode}")

        qfi_trace_after = self.model.qfi_trace(x_forget if len(x_forget) else x_retain[:1])
        forget_accuracy_after = evaluate_accuracy(self.model, x_forget, y_forget)
        retain_accuracy_after = evaluate_accuracy(self.model, x_retain, y_retain)
        mia_after = membership_inference_success_rate(self.model, x_forget, y_forget, x_retain, y_retain)
        shap_after = self.shap_attribution(x_forget, y_forget, num_permutations=num_shap_permutations)

        report = UnlearningReport(
            qfi_trace_before=qfi_trace_before,
            qfi_trace_after=qfi_trace_after,
            forget_accuracy_before=forget_accuracy_before,
            forget_accuracy_after=forget_accuracy_after,
            retain_accuracy_before=retain_accuracy_before,
            retain_accuracy_after=retain_accuracy_after,
            mia_success_rate_before=mia_before,
            mia_success_rate_after=mia_after,
            random_baseline_accuracy=random_baseline_accuracy(y_forget if len(y_forget) else y_retain),
            shap_drop_mean=float(np.mean(shap_before - shap_after)) if shap_before.size else 0.0,
        )
        return report, {"shap_before": shap_before, "shap_after": shap_after, "mask": mask}

    def _masked_gradient_step(self, x_forget: np.ndarray, y_forget: np.ndarray, shap_mask: np.ndarray, lr: float = 0.05) -> np.ndarray:
        base_weights = self._flatten()

        def loss_fn(flat_weights):
            self._assign_weights(np.asarray(flat_weights))
            return self.model.loss(x_forget, y_forget)

        grad = np.asarray(qml.grad(loss_fn)(base_weights))
        if grad.size == 0:
            return base_weights
        new_weights = base_weights - lr * (grad * shap_mask)
        self._assign_weights(new_weights)
        return new_weights
