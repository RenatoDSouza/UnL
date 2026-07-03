"""Federated execution orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qfl.common.types import FederatedResult
from qfl.federated.client import FederatedClient
from qfl.federated.server import FederatedServer


@dataclass
class FederatedTrainingRun:
    server: FederatedServer
    clients: list[FederatedClient]

    def run(self, num_rounds: int = 1) -> list[FederatedResult]:
        current_weights = self.server.initial_weights
        results: list[FederatedResult] = []
        for round_index in range(num_rounds):
            updates = [client.train(current_weights) for client in self.clients]
            result = self.server.aggregate(updates, round_index=round_index)
            current_weights = np.asarray(result.global_weights)
            results.append(result)
        return results

