"""Federated client abstraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pennylane import numpy as pnp

from qfl.common.types import ClientUpdate
from qfl.quantum.model import QuantumClassifier


@dataclass
class FederatedClient:
    client_id: str
    x_train: np.ndarray
    y_train: np.ndarray

    def train(self, global_weights, epochs: int = 1, lr: float = 0.05) -> ClientUpdate:
        model = QuantumClassifier(num_wires=self.x_train.shape[1], prefer_gpu=True)
        model.weights = pnp.array(global_weights, requires_grad=True)
        for _ in range(epochs):
            for sample, target in zip(self.x_train, self.y_train):
                grad = np.sign(float(model._qnode(sample, model.weights)) - float(target))
                model.weights = model.weights - lr * grad
        return ClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.x_train),
            weights=model.weights.tolist(),
            metadata={"loss_proxy": float(np.mean(model.predict_proba(self.x_train)[:, 1]))},
        )
