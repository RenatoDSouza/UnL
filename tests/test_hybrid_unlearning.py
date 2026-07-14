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

    report, artifacts = unlearner.run(
        x_forget, y_forget, x_retain, y_retain, num_shap_permutations=2, mask_quantile=0.5, lr=0.01
    )

    assert report.qfi_trace_before == 1.0
    assert "mask" in artifacts
    assert artifacts["mask"].shape == (model.weights.size,)
    assert 0.0 <= report.mia_auc_after <= 1.0
    assert report.uninformed_accuracy in (0.5, 1.0)


def test_shapley_attribution_groups_by_wire():
    model = QuantumClassifier(num_wires=4, num_layers=2, prefer_gpu=False)
    model.weights = pnp.array(0.1 * np.random.randn(2, 4, 3), requires_grad=True)
    unlearner = HybridSHAPQFIUnlearner(model)
    groups = unlearner.wire_parameter_groups()
    assert len(groups) == 4  # one group per wire
    assert sum(len(g) for g in groups) == model.weights.size

    x = np.random.randn(6, 4)
    y = (x[:, 0] > 0).astype(int)
    phi = unlearner.shapley_attribution(x, y, num_permutations=4)
    assert phi.shape == (model.weights.size,)
    # parameters within a wire group share the same Shapley value
    for group in groups:
        assert np.allclose(phi[group], phi[group][0])


def test_qfi_unlearning_run_trains_full_model(monkeypatch):
    from qfl.common.types import ClientUpdate
    from qfl.federated.server import FederatedServer
    from qfl.federated.strategy import FederatedTrainingRun
    from qfl.federated.unlearning import QFIUnlearningRun
    from qfl.federated.client import FederatedClient

    server = FederatedServer(initial_weights=np.zeros((2, 4, 3), dtype=float))
    clients = [
        FederatedClient("client_0", np.zeros((2, 4)), np.zeros(2, dtype=int)),
        FederatedClient("client_1", np.zeros((2, 4)), np.ones(2, dtype=int)),
    ]
    monkeypatch.setattr(
        FederatedClient,
        "train",
        lambda self, global_weights, epochs=None, lr=None: ClientUpdate(
            self.client_id, len(self.x_train), list(np.asarray(global_weights).reshape(-1)), {}
        ),
    )

    run = QFIUnlearningRun(FederatedTrainingRun(server, clients), "client_0")
    summary = run.run(num_rounds=1)

    assert summary["excluded_client"] == "client_0"
    assert summary["remaining_clients"] == 1.0
    assert "forget_set_accuracy" in summary
    assert "mia_auc" in summary
    assert len(summary["global_weights"]) == 2
