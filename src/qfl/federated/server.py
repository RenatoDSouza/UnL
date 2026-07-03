"""Central federated server."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qfl.common.types import ClientUpdate, FederatedResult


@dataclass
class FederatedServer:
    initial_weights: np.ndarray
    history: list[FederatedResult] = field(default_factory=list)

    def aggregate(self, updates: list[ClientUpdate], round_index: int) -> FederatedResult:
        if not updates:
            raise ValueError("updates cannot be empty")
        total = sum(update.num_samples for update in updates)
        weights = np.zeros_like(self.initial_weights, dtype=float)
        for update in updates:
            weights += np.asarray(update.weights) * (update.num_samples / total)
        result = FederatedResult(round_index=round_index, global_weights=weights.tolist(), metrics={"num_clients": float(len(updates))})
        self.history.append(result)
        return result

