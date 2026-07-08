from __future__ import annotations

import numpy as np
from pennylane import numpy as pnp

from qfl.federated.hybrid_unlearning import HybridSHAPQFIUnlearner
from qfl.quantum.model import QuantumClassifier


def test_hybrid_unlearning_returns_report(monkeypatch):
    model = QuantumClassifier(num_wires=4, num_layers=1, prefer_gpu=False)
    model.weights = pnp.zeros_like(model.weights, requires_grad=True)
    unlearner = HybridSHAPQFIUnlearner(model)

    monkeypatch.setattr(model, "qfi_matrix", lambda inputs, regularization=1e-6: np.eye(model.weights.size))
    monkeypatch.setattr(model, "qfi_trace", lambda inputs: 1.0)

    x_forget = np.zeros((2, 4), dtype=float)
    y_forget = np.zeros(2, dtype=int)
    x_retain = np.zeros((3, 4), dtype=float)
    y_retain = np.ones(3, dtype=int)

    report, artifacts = unlearner.run(x_forget, y_forget, x_retain, y_retain, num_shap_permutations=1, mask_quantile=0.5, lr=0.01)

    assert report.qfi_trace_before == 1.0
    assert "mask" in artifacts
    assert artifacts["mask"].shape == (model.weights.size,)
    assert report.random_baseline_accuracy in (0.5, 1.0)


def test_qfi_unlearning_run_is_backward_compatible(monkeypatch):
    from qfl.common.types import ClientUpdate
    from qfl.federated.server import FederatedServer
    from qfl.federated.strategy import FederatedTrainingRun
    from qfl.federated.unlearning import QFIUnlearningRun
    from qfl.federated.client import FederatedClient

    server = FederatedServer(initial_weights=np.zeros((1, 4, 3), dtype=float))
    clients = [
        FederatedClient("client_0", np.zeros((2, 4), dtype=float), np.zeros(2, dtype=int)),
        FederatedClient("client_1", np.zeros((2, 4), dtype=float), np.ones(2, dtype=int)),
    ]

    monkeypatch.setattr(FederatedClient, "train", lambda self, global_weights, epochs=1, lr=0.05: ClientUpdate(self.client_id, len(self.x_train), list(np.asarray(global_weights).reshape(-1)), {}))
    monkeypatch.setattr("qfl.federated.hybrid_unlearning.HybridSHAPQFIUnlearner.run", lambda self, forget_x, forget_y, retain_x, retain_y: (__import__("types").SimpleNamespace(qfi_trace_before=1.0, qfi_trace_after=0.9, forget_accuracy_before=1.0, forget_accuracy_after=0.0, retain_accuracy_before=1.0, retain_accuracy_after=1.0, mia_success_rate_before=1.0, mia_success_rate_after=0.0, random_baseline_accuracy=0.5, shap_drop_mean=0.1), {"shap_before": np.array([1.0]), "shap_after": np.array([0.0]), "mask": np.array([1.0])}))

    run = QFIUnlearningRun(FederatedTrainingRun(server, clients), "client_0")
    summary = run.run(num_rounds=1)

    assert summary["excluded_client"] == "client_0"
    assert summary["mask_density"] == 1.0
