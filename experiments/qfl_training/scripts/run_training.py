from __future__ import annotations

from pathlib import Path

import numpy as np

from qfl.data.femnist import load_femnist_partitions
from qfl.federated.client import FederatedClient
from qfl.federated.server import FederatedServer
from qfl.federated.strategy import FederatedTrainingRun
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
    run = FederatedTrainingRun(server=server, clients=clients)
    output_dir = ensure_dir("experiments/qfl_training/outputs")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    progress = ProgressTracker(label="QFL training", total_steps=3)
    start_checkpoint = checkpoint_dir / "training_state.json"
    checkpoint = load_checkpoint(start_checkpoint)
    start_round = int(checkpoint["current_step"]) if checkpoint and checkpoint.get("status") == "running" else 0
    results: list[dict[str, float]] = []
    if checkpoint and checkpoint.get("results"):
        results = list(checkpoint["results"])
    for round_index in range(start_round, 3):
        print(progress.progress_line(round_index))
        partial = run.run(num_rounds=1)
        round_metrics = partial[-1].metrics
        results.append(round_metrics)
        save_checkpoint(
            start_checkpoint,
            progress.checkpoint_payload(
                round_index + 1,
                {
                    "status": "running",
                    "results": results,
                    "last_round_metrics": round_metrics,
                },
            ),
        )
    print(progress.progress_line(3))
    save_checkpoint(
        checkpoint_dir / "training_complete.json",
        progress.checkpoint_payload(3, {"status": "completed", "results": results}),
    )
    write_json(output_dir / "training_summary.json", {"rounds": results})


if __name__ == "__main__":
    main()
