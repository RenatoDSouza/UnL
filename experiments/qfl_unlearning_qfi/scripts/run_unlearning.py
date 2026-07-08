from __future__ import annotations

from pathlib import Path

from qfl.experiments import load_unlearning_config, load_yaml_config, run_unlearning_experiment
from qfl.utils.logging import configure_logging


def main() -> None:
    configure_logging()
    config_path = Path("experiments/qfl_unlearning_qfi/configs/default.yaml")
    payload = load_yaml_config(config_path)
    config = load_unlearning_config({**payload, "output_dir": "experiments/qfl_unlearning_qfi/outputs"})
    run_unlearning_experiment(config)


if __name__ == "__main__":
    main()
