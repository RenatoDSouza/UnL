from __future__ import annotations

from pathlib import Path

import numpy as np

from qfl.data.femnist import compress_to_quadrants, normalize_images, partition_by_client
from qfl.federated.client import FederatedClient
from qfl.federated.server import FederatedServer
from qfl.federated.strategy import FederatedTrainingRun
from qfl.utils.io import ensure_dir, write_json
from qfl.utils.seed import set_seed


def main() -> None:
    set_seed(42)
    data_path = Path("data/femnist_sample.npz")
    data = np.load(data_path)
    x = compress_to_quadrants(normalize_images(data["x"]))
    y = (data["y"] > 0).astype(int)
    client_splits = partition_by_client(x, y, num_clients=5)
    clients = [FederatedClient(split.client_id, split.x, split.y) for split in client_splits]
    initial_weights = np.zeros((2, x.shape[1], 3), dtype=float)
    server = FederatedServer(initial_weights=initial_weights)
    run = FederatedTrainingRun(server=server, clients=clients)
    results = run.run(num_rounds=3)
    output_dir = ensure_dir("experiments/qfl_training/outputs")
    write_json(output_dir / "training_summary.json", {"rounds": [r.metrics for r in results]})


if __name__ == "__main__":
    main()
