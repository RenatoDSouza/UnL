from __future__ import annotations

from pathlib import Path

from qfl.experiments import load_training_config, load_yaml_config, run_training_experiment
from qfl.utils.logging import configure_logging


def main() -> None:
    configure_logging()
    config_path = Path("experiments/qfl_training/configs/default.yaml")
    payload = load_yaml_config(config_path)
    config = load_training_config({**payload, "output_dir": "experiments/qfl_training/outputs"})
    run_training_experiment(config)


if __name__ == "__main__":
    main()
