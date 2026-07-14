"""Federated client abstraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qfl.common.types import ClientUpdate
from qfl.federated.metrics import evaluate_accuracy
from qfl.quantum.model import QuantumClassifier


@dataclass
class FederatedClient:
    client_id: str
    x_train: np.ndarray
    y_train: np.ndarray
    prefer_gpu: bool = True
    encoding: str = "angle"
    data_reuploads: int = 1
    num_layers: int = 2

    epochs: int = 6
    lr: float = 0.2
    # Held-out non-members for membership inference (never trained on).
    x_eval: np.ndarray | None = None
    y_eval: np.ndarray | None = None

    def train(self, global_weights, epochs: int | None = None, lr: float | None = None) -> ClientUpdate:
        """Run local quantum training via true gradient descent.

        The model is built once (reusing a single quantum device) and its
        parameters are updated with the analytic gradient of the mean-squared
        loss computed by PennyLane's autograd interface. This replaces the old
        ``np.sign`` heuristic, which applied the same scalar step to every
        weight and did not perform real optimisation.
        """
        model = QuantumClassifier(
            num_wires=self.x_train.shape[1],
            num_layers=self.num_layers,
            prefer_gpu=self.prefer_gpu,
            encoding=self.encoding,
            data_reuploads=self.data_reuploads,
        )
        epochs = self.epochs if epochs is None else epochs
        lr = self.lr if lr is None else lr
        x = pnp.array(self.x_train, requires_grad=False)
        y = pnp.array(self.y_train.astype(float), requires_grad=False)
        flat = pnp.array(np.asarray(global_weights, dtype=float).reshape(-1), requires_grad=True)
        shape = model.weights.shape

        def loss_fn(flat_weights):
            model.weights = flat_weights.reshape(shape)
            return model.loss(x, y)

        grad_fn = qml.grad(loss_fn)
        for _ in range(max(1, epochs)):
            gradient = pnp.asarray(grad_fn(flat))
            flat = flat - lr * gradient

        model.weights = pnp.array(flat, requires_grad=True).reshape(shape)
        return ClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.x_train),
            weights=model.weights.tolist(),
            metadata={
                "train_loss": float(model.loss(self.x_train, self.y_train)),
                "train_accuracy": evaluate_accuracy(model, self.x_train, self.y_train),
            },
        )
