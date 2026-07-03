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

    def qfi_trace(self, inputs):
        metric_tensor = qml.metric_tensor(self._qnode, approx="block-diag")
        tensors = [metric_tensor(sample, self.weights) for sample in inputs]
        return float(sum(pnp.trace(tensor) for tensor in tensors))
