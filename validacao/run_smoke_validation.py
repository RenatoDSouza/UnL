from __future__ import annotations

import json
from pathlib import Path

from qfl.experiments.pipeline import (
    TrainingExperimentConfig,
    UnlearningExperimentConfig,
    run_training_experiment,
    run_unlearning_experiment,
)

NUM_LAYERS = 2
MAX_SAMPLES_PER_CLIENT = 12
NUM_FEATURES = 4


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _assert_num_layers(summary: dict, expected: int) -> None:
    metadata = summary.get("metadata", {})
    actual = metadata.get("num_layers")
    if actual != expected:
        raise RuntimeError(f"Smoke test expected num_layers={expected}, got {actual}")


def main() -> None:
    base_dir = Path("validacao")
    training_out = base_dir / "training_outputs"
    unlearning_out = base_dir / "unlearning_outputs"
    encodings = ["angle", "iqp", "reupload"]
    payload: dict[str, object] = {
        "encodings": encodings,
        "num_layers": NUM_LAYERS,
        "dataset": "flwrlabs/femnist (digits 0 and 1 only)",
        "max_samples_per_client": MAX_SAMPLES_PER_CLIENT,
        "num_features": NUM_FEATURES,
        "training": {},
        "unlearning": {},
    }

    for encoding in encodings:
        train_config = TrainingExperimentConfig(
            num_clients=2,
            num_rounds=1,
            seed=7,
            seeds=[7],
            encoding_modes=[encoding],
            data_reuploads=2,
            num_layers=NUM_LAYERS,
            feature_mode="binary",
            n_features=NUM_FEATURES,
            max_samples_per_client=MAX_SAMPLES_PER_CLIENT,
            prefer_gpu=True,
            output_dir=str(training_out / encoding),
        )
        training_summary = run_training_experiment(train_config)
        _assert_num_layers(training_summary, NUM_LAYERS)
        payload["training"][encoding] = training_summary

        unlearn_config = UnlearningExperimentConfig(
            num_clients=2,
            num_rounds=1,
            excluded_client_id="client_0",
            seed=7,
            seeds=[7],
            encoding_modes=[encoding],
            data_reuploads=2,
            num_layers=NUM_LAYERS,
            feature_mode="binary",
            n_features=NUM_FEATURES,
            max_samples_per_client=MAX_SAMPLES_PER_CLIENT,
            shap_permutations=1,
            unlearn_max_steps=1,
            prefer_gpu=True,
            output_dir=str(unlearning_out / encoding),
        )
        unlearning_summary = run_unlearning_experiment(unlearn_config)
        _assert_num_layers(unlearning_summary, NUM_LAYERS)
        payload["unlearning"][encoding] = unlearning_summary

    _write(base_dir / "results.json", payload)


if __name__ == "__main__":
    main()
