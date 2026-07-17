"""Quantum model used by the federated clients."""

from __future__ import annotations

import logging

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qfl.quantum.device import create_device

LOGGER = logging.getLogger(__name__)


class QFIComputationError(RuntimeError):
    """Raised when a mathematically valid quantum Fisher matrix is unavailable."""


class QuantumClassifier:
    def __init__(
        self,
        num_wires: int = 4,
        num_layers: int = 2,
        prefer_gpu: bool = True,
        encoding: str = "angle",
        data_reuploads: int = 1,
    ):
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.encoding = encoding
        self.data_reuploads = max(1, data_reuploads)
        self.device = create_device(num_wires=num_wires, prefer_gpu=prefer_gpu)
        self.weights = pnp.zeros(self.weight_shape(num_layers, num_wires, encoding, self.data_reuploads), requires_grad=True)
        self._qnode = qml.QNode(self._circuit, self.device, interface="autograd")

    @staticmethod
    def weight_shape(num_layers: int, num_wires: int, encoding: str = "angle", data_reuploads: int = 1) -> tuple[int, int, int]:
        """Return the variational-weight shape for the chosen circuit.

        A re-upload circuit has a distinct variational block after each data
        embedding. ``num_layers`` is the number of entangling layers *per
        upload*, hence its total number of trainable layers is multiplied here.
        """
        blocks = num_layers * max(1, data_reuploads) if encoding == "reupload" else num_layers
        return (blocks, num_wires, 3)

    def _encode(self, inputs):
        if self.encoding == "angle":
            qml.AngleEmbedding(inputs, wires=range(self.num_wires), rotation="Y")
        elif self.encoding == "iqp":
            qml.IQPEmbedding(inputs, wires=range(self.num_wires))
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def _circuit(self, inputs, weights):
        if self.encoding == "reupload":
            for upload in range(self.data_reuploads):
                qml.AngleEmbedding(inputs, wires=range(self.num_wires), rotation="Y")
                start = upload * self.num_layers
                qml.StronglyEntanglingLayers(weights[start : start + self.num_layers], wires=range(self.num_wires))
        else:
            self._encode(inputs)
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

    def qfi_matrix(self, inputs, regularization: float = 0.0):
        """Return the pure-state quantum Fisher information matrix (QFIM).

        PennyLane's metric tensor is the Fubini--Study metric for pure states;
        the QFIM is therefore ``4 * metric_tensor``.  No numerical fallback is
        substituted: callers must label a run invalid if this computation fails.
        ``regularization`` is retained for backwards compatibility but must be
        zero; damping belongs to the linear solver, not to the measured QFIM.
        """
        if regularization != 0.0:
            raise ValueError("QFIM regularization is not allowed; use solver damping in ascent")
        if len(inputs) == 0:
            raise QFIComputationError("Cannot estimate a QFIM from an empty input set")
        n = self.weights.size
        weights = pnp.array(self.weights, requires_grad=True)
        try:
            metric_tensor = qml.metric_tensor(self._qnode, approx="block-diag")
            tensors = [metric_tensor(pnp.array(sample, requires_grad=False), weights) for sample in inputs]
            qfi = 4.0 * sum(pnp.asarray(tensor).reshape(n, n) for tensor in tensors) / len(tensors)
        except Exception as exc:
            raise QFIComputationError("PennyLane could not compute the pure-state QFIM") from exc
        qfi = np.asarray(qfi, dtype=float)
        qfi = (qfi + qfi.T) / 2.0
        if not np.isfinite(qfi).all():
            raise QFIComputationError("QFIM contains non-finite values")
        min_eigenvalue = float(np.linalg.eigvalsh(qfi)[0])
        if min_eigenvalue < -1e-8:
            raise QFIComputationError(f"QFIM is not positive semidefinite (min eigenvalue={min_eigenvalue:.3e})")
        return pnp.array(qfi, requires_grad=False)

    def qfi_trace(self, inputs):
        return float(pnp.trace(self.qfi_matrix(inputs)))
