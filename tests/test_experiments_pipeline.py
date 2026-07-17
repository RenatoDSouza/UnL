from __future__ import annotations

from pathlib import Path

import pytest

from qfl.experiments.pipeline import TrainingExperimentConfig, _aggregate_runs, _normalize_config, experiment_metadata, load_training_config
from qfl.utils.checkpoint import save_progress_checkpoint


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


def test_experiment_configuration_requires_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        _normalize_config(TrainingExperimentConfig(prefer_gpu=False))


def test_progress_checkpoint_is_human_readable(tmp_path: Path):
    checkpoint = tmp_path / "progress.txt"
    save_progress_checkpoint(
        checkpoint,
        {
            "status": "running",
            "label": "QFL training (angle, seed=7)",
            "current_step": 1,
            "total_steps": 5,
            "updated_at_utc": "2026-07-16T00:00:00+00:00",
            "eta_seconds": 120.0,
            "seed": 7,
            "encoding": "angle",
            "metrics": {"global_accuracy": 0.5},
        },
    )

    text = checkpoint.read_text(encoding="utf-8")
    assert "Progresso: 1/5" in text
    assert "global_accuracy" in text
