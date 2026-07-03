from __future__ import annotations

import numpy as np

from qfl.common.types import ClientUpdate
from qfl.data.femnist import compress_to_quadrants, partition_by_client
from qfl.federated.server import FederatedServer


def test_partition_by_client_returns_requested_count():
    x = np.zeros((10, 28, 28), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    splits = partition_by_client(x, y, num_clients=5)
    assert len(splits) == 5


def test_compress_to_quadrants_returns_four_features():
    x = np.arange(2 * 28 * 28, dtype=np.float32).reshape(2, 28, 28)
    compressed = compress_to_quadrants(x)
    assert compressed.shape == (2, 4)


def test_server_aggregates_by_sample_count():
    server = FederatedServer(initial_weights=np.zeros((2, 4, 3), dtype=float))
    updates = [
        ClientUpdate("a", 1, [1.0] * 24, {}),
        ClientUpdate("b", 3, [3.0] * 24, {}),
    ]
    result = server.aggregate(updates, round_index=0)
    assert len(result.global_weights) == 2
    assert result.metrics["num_clients"] == 2.0

