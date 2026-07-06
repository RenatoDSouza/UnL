"""Machine unlearning utilities based on QFI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pennylane import numpy as pnp

from qfl.federated.mia import membership_inference_success_rate
from qfl.federated.metrics import UnlearningMetrics, evaluate_accuracy, random_baseline_accuracy
from qfl.federated.strategy import FederatedTrainingRun
from qfl.quantum.model import QuantumClassifier


@dataclass
class QFIUnlearningRun:
    training_run: FederatedTrainingRun
    excluded_client_id: str

    def run(self, num_rounds: int = 1) -> dict[str, float | dict[str, float]]:
        active_clients = [client for client in self.training_run.clients if client.client_id != self.excluded_client_id]
        excluded_clients = [client for client in self.training_run.clients if client.client_id == self.excluded_client_id]
        if not active_clients:
            raise ValueError("At least one active client is required")
        base_results = FederatedTrainingRun(self.training_run.server, active_clients).run(num_rounds=num_rounds)
        model = QuantumClassifier(num_wires=active_clients[0].x_train.shape[1], prefer_gpu=True)
        model.weights = pnp.array(base_results[-1].global_weights, requires_grad=True)
        qfi_score = model.qfi_trace(active_clients[0].x_train[: min(4, len(active_clients[0].x_train))])
        forget_x = excluded_clients[0].x_train if excluded_clients else np.empty((0, active_clients[0].x_train.shape[1]))
        forget_y = excluded_clients[0].y_train if excluded_clients else np.empty((0,))
        retain_x = np.concatenate([client.x_train for client in active_clients], axis=0)
        retain_y = np.concatenate([client.y_train for client in active_clients], axis=0)
        forget_accuracy = evaluate_accuracy(model, forget_x, forget_y)
        retain_accuracy = evaluate_accuracy(model, retain_x, retain_y)
        mia_success_rate = membership_inference_success_rate(
            model=model,
            x_train=forget_x,
            y_train=forget_y,
            x_test=retain_x,
            y_test=retain_y,
        )
        return {
            "qfi_trace": qfi_score,
            "remaining_clients": float(len(active_clients)),
            "excluded_client": self.excluded_client_id,
            "forget_set_accuracy": forget_accuracy,
            "retain_set_accuracy": retain_accuracy,
            "random_baseline_accuracy": random_baseline_accuracy(forget_y if len(forget_y) else retain_y),
            "mia_success_rate": mia_success_rate,
        }
