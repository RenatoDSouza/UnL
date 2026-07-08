from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qfl.experiments.pipeline import (
    TrainingExperimentConfig,
    UnlearningExperimentConfig,
    run_training_experiment,
    run_unlearning_experiment,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    base_dir = Path("validacao")
    training_out = base_dir / "training_outputs"
    unlearning_out = base_dir / "unlearning_outputs"
    dataset_path = base_dir / "synthetic_validation.npz"
    x = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.8, 0.7, 0.6, 0.5],
            [0.9, 0.8, 0.7, 0.6],
            [0.15, 0.25, 0.35, 0.45],
            [0.85, 0.75, 0.65, 0.55],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    np.savez(dataset_path, x=x, y=y)

    encodings = ["angle", "iqp", "reupload"]
    payload: dict[str, object] = {"encodings": encodings, "training": {}, "unlearning": {}}

    for encoding in encodings:
        train_config = TrainingExperimentConfig(
            dataset_path=str(dataset_path),
            num_clients=2,
            num_rounds=1,
            seed=7,
            seeds=[7],
            encoding_modes=[encoding],
            data_reuploads=2,
            prefer_gpu=False,
            output_dir=str(training_out / encoding),
        )
        payload["training"][encoding] = run_training_experiment(train_config)

        unlearn_config = UnlearningExperimentConfig(
            dataset_path=str(dataset_path),
            num_clients=2,
            num_rounds=1,
            excluded_client_id="client_0",
            seed=7,
            seeds=[7],
            encoding_modes=[encoding],
            data_reuploads=2,
            prefer_gpu=False,
            output_dir=str(unlearning_out / encoding),
        )
        payload["unlearning"][encoding] = run_unlearning_experiment(unlearn_config)

    _write(base_dir / "results.json", payload)


if __name__ == "__main__":
    main()
