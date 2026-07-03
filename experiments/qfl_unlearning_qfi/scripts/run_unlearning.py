from __future__ import annotations

from pathlib import Path

import numpy as np

from qfl.data.femnist import compress_to_quadrants, normalize_images, partition_by_client
from qfl.federated.client import FederatedClient
from qfl.federated.server import FederatedServer
from qfl.federated.strategy import FederatedTrainingRun
from qfl.federated.unlearning import QFIUnlearningRun
from qfl.utils.io import ensure_dir, write_json
from qfl.utils.seed import set_seed


def main() -> None:
    set_seed(42)
    data = np.load(Path("data/femnist_sample.npz"))
    x = compress_to_quadrants(normalize_images(data["x"]))
    y = (data["y"] > 0).astype(int)
    client_splits = partition_by_client(x, y, num_clients=5)
    clients = [FederatedClient(split.client_id, split.x, split.y) for split in client_splits]
    initial_weights = np.zeros((2, x.shape[1], 3), dtype=float)
    server = FederatedServer(initial_weights=initial_weights)
    training_run = FederatedTrainingRun(server=server, clients=clients)
    unlearning_run = QFIUnlearningRun(training_run=training_run, excluded_client_id="client_0")
    summary = unlearning_run.run(num_rounds=3)
    output_dir = ensure_dir("experiments/qfl_unlearning_qfi/outputs")
    write_json(output_dir / "unlearning_summary.json", summary)


if __name__ == "__main__":
    main()
