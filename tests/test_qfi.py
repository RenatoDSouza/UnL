"""Numerical checks for the QFIM convention used by the unlearning method."""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qfl.quantum.model import QFIComputationError, QuantumClassifier


def test_pure_state_qfim_is_four_times_fubini_study_metric():
    """For |psi(theta)>=RY(theta)|0>, g_FS=1/4 and QFI=1 analytically."""
    device = qml.device("default.qubit", wires=1)

    @qml.qnode(device, interface="autograd")
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    metric = float(qml.metric_tensor(circuit)(pnp.array(0.37, requires_grad=True)))
    assert np.isclose(metric, 0.25, atol=1e-8)
    assert np.isclose(4.0 * metric, 1.0, atol=1e-8)


def test_qfim_failure_is_explicit(monkeypatch):
    model = QuantumClassifier(num_wires=1, num_layers=1, prefer_gpu=False)

    def broken_metric_tensor(*args, **kwargs):
        raise RuntimeError("backend failure")

    monkeypatch.setattr(qml, "metric_tensor", broken_metric_tensor)
    try:
        model.qfi_matrix(np.zeros((1, 1)))
    except QFIComputationError:
        pass
    else:  # pragma: no cover - protects against an invalid identity fallback
        raise AssertionError("QFIM failure must not be replaced by an identity matrix")


def test_reupload_has_distinct_variational_block_per_embedding():
    model = QuantumClassifier(num_wires=2, num_layers=2, data_reuploads=3, encoding="reupload", prefer_gpu=False)
    assert model.weights.shape == (6, 2, 3)
