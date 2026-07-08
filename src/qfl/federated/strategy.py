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
        from tqdm import tqdm
        current_weights = self.server.initial_weights
        results: list[FederatedResult] = []
        rounds_iter = tqdm(range(num_rounds), desc="Rodadas Federadas", leave=False) if num_rounds > 1 else range(num_rounds)
        for round_index in rounds_iter:
            updates = [client.train(current_weights) for client in self.clients]
            result = self.server.aggregate(updates, round_index=round_index)
            current_weights = np.asarray(result.global_weights)
            results.append(result)
        return results


