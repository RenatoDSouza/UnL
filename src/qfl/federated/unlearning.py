"""Backward-compatible unlearning entry point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pennylane import numpy as pnp

from qfl.federated.hybrid_unlearning import HybridSHAPQFIUnlearner
from qfl.federated.strategy import FederatedTrainingRun
from qfl.quantum.model import QuantumClassifier


@dataclass
class QFIUnlearningRun:
    training_run: FederatedTrainingRun
    excluded_client_id: str
    prefer_gpu: bool = True
    encoding: str = "angle"
    data_reuploads: int = 1

    def run(self, num_rounds: int = 1) -> dict[str, float | dict[str, float]]:
        active_clients = [client for client in self.training_run.clients if client.client_id != self.excluded_client_id]
        excluded_clients = [client for client in self.training_run.clients if client.client_id == self.excluded_client_id]
        if not active_clients:
            raise ValueError("At least one active client is required")

        base_results = FederatedTrainingRun(self.training_run.server, active_clients).run(num_rounds=num_rounds)
        model = QuantumClassifier(
            num_wires=active_clients[0].x_train.shape[1],
            prefer_gpu=self.prefer_gpu,
            encoding=self.encoding,
            data_reuploads=self.data_reuploads,
        )
        model.weights = pnp.array(base_results[-1].global_weights, requires_grad=True).reshape(model.weights.shape)

        forget_x = excluded_clients[0].x_train if excluded_clients else np.empty((0, active_clients[0].x_train.shape[1]))
        forget_y = excluded_clients[0].y_train if excluded_clients else np.empty((0,))
        retain_x = np.concatenate([client.x_train for client in active_clients], axis=0)
        retain_y = np.concatenate([client.y_train for client in active_clients], axis=0)

        report, artifacts = HybridSHAPQFIUnlearner(model).run(forget_x, forget_y, retain_x, retain_y, mode="shap_qfi")
        return {
            "global_weights": np.asarray(base_results[-1].global_weights, dtype=float).tolist(),
            "qfi_trace_before": report.qfi_trace_before,
            "qfi_trace_after": report.qfi_trace_after,
            "forget_set_accuracy_before": report.forget_accuracy_before,
            "forget_set_accuracy_after": report.forget_accuracy_after,
            "retain_set_accuracy_before": report.retain_accuracy_before,
            "retain_set_accuracy_after": report.retain_accuracy_after,
            "random_baseline_accuracy": report.random_baseline_accuracy,
            "mia_success_rate_before": report.mia_success_rate_before,
            "mia_success_rate_after": report.mia_success_rate_after,
            "shap_drop_mean": report.shap_drop_mean,
            "remaining_clients": float(len(active_clients)),
            "excluded_client": self.excluded_client_id,
            "shap_before_mean": float(np.mean(artifacts["shap_before"])) if artifacts["shap_before"].size else 0.0,
            "shap_after_mean": float(np.mean(artifacts["shap_after"])) if artifacts["shap_after"].size else 0.0,
            "mask_density": float(np.mean(artifacts["mask"])) if artifacts["mask"].size else 0.0,
        }
