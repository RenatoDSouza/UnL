from __future__ import annotations

from qfl.experiments.pipeline import TrainingExperimentConfig, _aggregate_runs, experiment_metadata, load_training_config


def test_aggregate_runs_computes_mean_and_std():
    summary = _aggregate_runs(
        [
            {"seed": 1, "metrics": {"num_clients": 4.0, "loss": 2.0}},
            {"seed": 2, "metrics": {"num_clients": 4.0, "loss": 4.0}},
        ]
    )

    assert summary["num_runs"] == 2
    assert summary["loss_mean"] == 3.0
    assert summary["loss_std"] > 0.0
    assert summary["loss_median"] == 3.0
    assert summary["loss_ci95_low"] <= summary["loss_mean"] <= summary["loss_ci95_high"]


def test_experiment_metadata_includes_seed_list():
    metadata = experiment_metadata(TrainingExperimentConfig(seed=7, seeds=[7, 8], num_clients=2, num_rounds=1))
    assert metadata["seed"] == 7
    assert metadata["seeds"] == [7, 8]
    assert metadata["num_clients"] == 2


def test_load_training_config_applies_defaults():
    config = load_training_config({"dataset_path": "data/sample.npz", "seeds": [1, 2]})
    assert config.dataset_path == "data/sample.npz"
    assert config.num_clients == 5
    assert config.num_rounds == 3
    assert config.seeds == [1, 2]


def test_load_training_config_accepts_encoding_modes():
    config = load_training_config({"encoding_modes": ["angle", "iqp"], "data_reuploads": 2})
    assert config.encoding_modes == ["angle", "iqp"]
    assert config.data_reuploads == 2
