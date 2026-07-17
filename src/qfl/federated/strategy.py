"""Federated execution orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from qfl.common.types import FederatedResult
from qfl.federated.client import FederatedClient
from qfl.federated.metrics import evaluate_accuracy
from qfl.federated.server import FederatedServer

LOGGER = logging.getLogger(__name__)


@dataclass
class FederatedTrainingRun:
    server: FederatedServer
    clients: list[FederatedClient]
    eval_cap: int = 300

    def run(
        self,
        num_rounds: int = 1,
        initial_weights: np.ndarray | None = None,
        on_round_complete: Callable[[FederatedResult], None] | None = None,
    ) -> list[FederatedResult]:
        from tqdm import tqdm
        current_weights = self.server.initial_weights if initial_weights is None else initial_weights
        results: list[FederatedResult] = []
        rounds_iter = tqdm(range(num_rounds), desc="Rodadas Federadas", leave=False) if num_rounds > 1 else range(num_rounds)
        for round_index in rounds_iter:
            updates = [client.train(current_weights) for client in self.clients]
            result = self.server.aggregate(updates, round_index=round_index)
            current_weights = np.asarray(result.global_weights)
            self._attach_global_metrics(result, current_weights)
            results.append(result)
            if on_round_complete:
                on_round_complete(result)
        return results

    def _attach_global_metrics(self, result: FederatedResult, global_weights: np.ndarray) -> None:
        """Evaluate the aggregated global model on a capped pool of client data.

        Produces a genuine per-round learning curve (``global_accuracy`` /
        ``global_loss``) instead of only reporting the client count.
        """
        from qfl.quantum.model import QuantumClassifier

        if not self.clients:
            return
        try:
            x = np.concatenate([c.x_train for c in self.clients], axis=0)
            y = np.concatenate([c.y_train for c in self.clients], axis=0)
            if self.eval_cap and len(x) > self.eval_cap:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(x), size=self.eval_cap, replace=False)
                x, y = x[idx], y[idx]
            ref = self.clients[0]
            model = QuantumClassifier(
                num_wires=ref.x_train.shape[1],
                num_layers=ref.num_layers,
                prefer_gpu=ref.prefer_gpu,
                encoding=ref.encoding,
                data_reuploads=ref.data_reuploads,
            )
            model.weights = np.asarray(global_weights, dtype=float).reshape(model.weights.shape)
            result.metrics["global_accuracy"] = evaluate_accuracy(model, x, y)
            result.metrics["global_loss"] = float(model.loss(x, y))
        except Exception:  # metrics must never break the training loop
            LOGGER.warning("Could not compute global round metrics", exc_info=True)
