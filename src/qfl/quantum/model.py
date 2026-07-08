"""Quantum model used by the federated clients."""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp

from qfl.quantum.device import create_device


class QuantumClassifier:
    def __init__(self, num_wires: int = 4, num_layers: int = 2, prefer_gpu: bool = True):
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.device = create_device(num_wires=num_wires, prefer_gpu=prefer_gpu)
        self.weights = pnp.zeros((num_layers, num_wires, 3), requires_grad=True)
        self._qnode = qml.QNode(self._circuit, self.device, interface="autograd")

    def _circuit(self, inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(self.num_wires), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(self.num_wires))
        return qml.expval(qml.PauliZ(0))

    def predict_proba(self, inputs):
        values = [self._qnode(sample, self.weights) for sample in inputs]
        probs = (pnp.array(values) + 1.0) / 2.0
        return pnp.stack([1.0 - probs, probs], axis=1)

    def loss(self, inputs, targets):
        probs = self.predict_proba(inputs)[:, 1]
        return pnp.mean((probs - targets) ** 2)

    @property
    def flat_weights(self):
        return pnp.asarray(self.weights).reshape(-1)

    @flat_weights.setter
    def flat_weights(self, values):
        self.weights = pnp.array(values, requires_grad=True).reshape(self.weights.shape)

    def qfi_matrix(self, inputs, regularization: float = 1e-6):
        metric_tensor = qml.metric_tensor(self._qnode, approx="block-diag")
        n = self.weights.size
        tensors = [metric_tensor(sample, self.weights) for sample in inputs]
        qfi = sum(pnp.asarray(tensor).reshape(n, n) for tensor in tensors) / max(1, len(tensors))
        return qfi + regularization * pnp.eye(n)

    def qfi_trace(self, inputs):
        return float(pnp.trace(self.qfi_matrix(inputs)))
