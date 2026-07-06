from __future__ import annotations

from pathlib import Path

import numpy as np

from qfl.data.femnist import load_femnist_partitions
from qfl.federated.client import FederatedClient
from qfl.federated.server import FederatedServer
from qfl.federated.strategy import FederatedTrainingRun
from qfl.federated.unlearning import QFIUnlearningRun
from qfl.utils.checkpoint import load_checkpoint, save_checkpoint
from qfl.utils.io import ensure_dir, write_json
from qfl.utils.progress import ProgressTracker
from qfl.utils.seed import set_seed


def main() -> None:
    set_seed(42)
    client_splits = load_femnist_partitions(num_clients=5)
    clients = [FederatedClient(split.client_id, split.x, split.y) for split in client_splits]
    initial_weights = np.zeros((2, client_splits[0].x.shape[1], 3), dtype=float)
    server = FederatedServer(initial_weights=initial_weights)
    training_run = FederatedTrainingRun(server=server, clients=clients)
    unlearning_run = QFIUnlearningRun(training_run=training_run, excluded_client_id="client_0")
    output_dir = ensure_dir("experiments/qfl_unlearning_qfi/outputs")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    progress = ProgressTracker(label="QFI unlearning", total_steps=3)
    state_path = checkpoint_dir / "unlearning_state.json"
    checkpoint = load_checkpoint(state_path)
    if checkpoint and checkpoint.get("status") == "completed" and checkpoint.get("summary"):
        summary = checkpoint["summary"]
    else:
        current_step = int(checkpoint["current_step"]) if checkpoint and checkpoint.get("status") == "running" else 0
        print(progress.progress_line(current_step))
        summary = unlearning_run.run(num_rounds=3)
        save_checkpoint(
            state_path,
            progress.checkpoint_payload(3, {"status": "completed", "summary": summary}),
        )
        print(progress.progress_line(3))
    write_json(output_dir / "unlearning_summary.json", summary)


if __name__ == "__main__":
    main()
