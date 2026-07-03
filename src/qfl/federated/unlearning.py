"""Machine unlearning utilities based on QFI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qfl.federated.strategy import FederatedTrainingRun
from qfl.quantum.model import QuantumClassifier


@dataclass
class QFIUnlearningRun:
    training_run: FederatedTrainingRun
    excluded_client_id: str

    def run(self, num_rounds: int = 1) -> dict[str, float]:
        active_clients = [client for client in self.training_run.clients if client.client_id != self.excluded_client_id]
        base_results = FederatedTrainingRun(self.training_run.server, active_clients).run(num_rounds=num_rounds)
        model = QuantumClassifier(num_wires=active_clients[0].x_train.shape[1], prefer_gpu=True)
        model.weights = np.asarray(base_results[-1].global_weights)
        qfi_score = model.qfi_trace(active_clients[0].x_train[: min(4, len(active_clients[0].x_train))])
        return {
            "qfi_trace": qfi_score,
            "remaining_clients": float(len(active_clients)),
            "excluded_client": self.excluded_client_id,
        }

