"""Train the full federated model and report the pre-unlearning reference.

This trains the global model on *all* clients (including the one to be
forgotten) and reports its forget/retain metrics and membership-inference AUC.
The actual unlearning variants (shap_only, qfi_only, shap_qfi) and the
retrain-from-scratch reference are evaluated by the pipeline on copies of this
model, so the expensive full training happens only once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pennylane import numpy as pnp

from qfl.federated.metrics import evaluate_accuracy, majority_class_rate
from qfl.federated.mia import membership_inference_auc
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
        clients = self.training_run.clients
        active_clients = [c for c in clients if c.client_id != self.excluded_client_id]
        excluded = [c for c in clients if c.client_id == self.excluded_client_id]
        if not active_clients:
            raise ValueError("At least one active client is required")

        base_results = FederatedTrainingRun(self.training_run.server, clients).run(num_rounds=num_rounds)
        num_wires = active_clients[0].x_train.shape[1]
        model = QuantumClassifier(
            num_wires=num_wires,
            num_layers=active_clients[0].num_layers,
            prefer_gpu=self.prefer_gpu,
            encoding=self.encoding,
            data_reuploads=self.data_reuploads,
        )
        model.weights = pnp.array(base_results[-1].global_weights, requires_grad=True).reshape(model.weights.shape)

        forget_x = excluded[0].x_train if excluded else np.empty((0, num_wires))
        forget_y = excluded[0].y_train if excluded else np.empty((0,))
        forget_x_eval = excluded[0].x_eval if excluded and excluded[0].x_eval is not None else np.empty((0, num_wires))
        forget_y_eval = excluded[0].y_eval if excluded and excluded[0].y_eval is not None else np.empty((0,))
        retain_x = np.concatenate([c.x_train for c in active_clients], axis=0)
        retain_y = np.concatenate([c.y_train for c in active_clients], axis=0)

        def _loss(x, y):
            return float(model.loss(x, y)) if len(x) else 0.0

        return {
            "global_weights": np.asarray(base_results[-1].global_weights, dtype=float).tolist(),
            "qfi_trace": model.qfi_trace(forget_x if len(forget_x) else retain_x[:1]),
            "forget_set_accuracy": evaluate_accuracy(model, forget_x, forget_y),
            "forget_set_loss": _loss(forget_x, forget_y),
            "retain_set_accuracy": evaluate_accuracy(model, retain_x, retain_y),
            "retain_set_loss": _loss(retain_x, retain_y),
            "uninformed_accuracy": majority_class_rate(forget_y if len(forget_y) else retain_y),
            "mia_auc": membership_inference_auc(model, forget_x, forget_y, forget_x_eval, forget_y_eval),
            "remaining_clients": float(len(active_clients)),
            "excluded_client": self.excluded_client_id,
        }
