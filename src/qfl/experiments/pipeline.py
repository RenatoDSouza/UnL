"""Shared experiment orchestration for training and unlearning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from statistics import mean, median, pstdev
import platform
import sys
import logging
import csv

import numpy as np
import yaml

from qfl.data.femnist import (
    FEMNISTSplit,
    compress_to_quadrants,
    load_femnist_backdoor_partitions,
    load_femnist_binary_partitions,
    load_femnist_noniid_partitions,
    load_femnist_partitions,
    load_femnist_source,
    normalize_images,
    partition_by_client,
)
from qfl.federated.client import FederatedClient
from qfl.federated.metrics import evaluate_accuracy
from qfl.federated.mia import membership_inference_auc
from qfl.federated.server import FederatedServer
from qfl.federated.strategy import FederatedTrainingRun
from qfl.federated.hybrid_unlearning import HybridSHAPQFIUnlearner
from qfl.federated.unlearning import QFIUnlearningRun
from qfl.utils.checkpoint import load_checkpoint, save_checkpoint, save_progress_checkpoint
from qfl.utils.io import ensure_dir, write_json
from qfl.utils.progress import ProgressTracker
from qfl.utils.seed import set_seed


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingExperimentConfig:
    dataset_path: str | None = None
    num_clients: int = 5
    num_rounds: int = 3
    seed: int = 42
    seeds: list[int] | None = None
    encoding_modes: list[str] | None = None
    data_reuploads: int = 1
    prefer_gpu: bool = True
    max_samples_per_client: int | None = None
    feature_mode: str = "quadrants"
    n_features: int = 8
    num_layers: int = 2
    train_epochs: int = 6
    train_lr: float = 0.2
    output_dir: str = "experiments/qfl_training/outputs"


@dataclass(frozen=True)
class UnlearningExperimentConfig:
    dataset_path: str | None = None
    num_clients: int = 5
    num_rounds: int = 3
    excluded_client_id: str = "client_0"
    seed: int = 42
    seeds: list[int] | None = None
    encoding_modes: list[str] | None = None
    data_reuploads: int = 1
    prefer_gpu: bool = True
    max_samples_per_client: int | None = None
    feature_mode: str = "quadrants"
    n_features: int = 8
    num_layers: int = 2
    train_epochs: int = 6
    train_lr: float = 0.2
    shap_permutations: int = 16
    unlearn_lr: float = 0.4
    unlearn_max_steps: int = 12
    mask_quantile: float = 0.5
    output_dir: str = "experiments/qfl_unlearning_qfi/outputs"


def experiment_metadata(config: TrainingExperimentConfig | UnlearningExperimentConfig) -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "seed": config.seed,
        "seeds": _seed_list(config.seed, config.seeds),
        "num_clients": config.num_clients,
        "num_rounds": config.num_rounds,
        "dataset_path": config.dataset_path,
        "prefer_gpu": config.prefer_gpu,
        "encoding_modes": _encoding_list(getattr(config, "encoding_modes", None)),
        "data_reuploads": getattr(config, "data_reuploads", 1),
        "num_layers": getattr(config, "num_layers", 2),
        "qfi_definition": "pure_state_qfim=4*fubini_study_metric; block-diagonal approximation",
        "gpu_policy": "lightning.gpu required when prefer_gpu=true; no CPU fallback",
    }


def _seed_list(seed: int, seeds: list[int] | None) -> list[int]:
    values = list(seeds or [])
    return values if values else [seed]


def _encoding_list(encoding_modes: list[str] | None) -> list[str]:
    values = list(encoding_modes or [])
    return values if values else ["angle"]


def _normalize_config(config: TrainingExperimentConfig | UnlearningExperimentConfig) -> TrainingExperimentConfig | UnlearningExperimentConfig:
    if config.num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if config.num_rounds <= 0:
        raise ValueError("num_rounds must be positive")
    if not config.prefer_gpu:
        raise ValueError("Experiment execution requires CUDA/lightning.gpu; prefer_gpu cannot be false")
    seeds = _seed_list(config.seed, config.seeds)
    if not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must contain integers")
    encodings = _encoding_list(getattr(config, "encoding_modes", None))
    valid = {"angle", "iqp", "reupload"}
    if any(mode not in valid for mode in encodings):
        raise ValueError(f"encoding_modes must be a subset of {sorted(valid)}")
    if getattr(config, "data_reuploads", 1) <= 0:
        raise ValueError("data_reuploads must be positive")
    return config


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"num_runs": 0, "runs": []}
    flattened: list[dict[str, Any]] = []
    for run in runs:
        if "metrics" in run and isinstance(run["metrics"], dict):
            flattened.append(run["metrics"])
        elif "summary" in run and isinstance(run["summary"], dict):
            if "main" in run["summary"] and isinstance(run["summary"]["main"], dict):
                flattened.append(run["summary"]["main"])
            else:
                flattened.append(run["summary"])
        else:
            flattened.append(run)
    numeric_keys = sorted({key for run in flattened for key, value in run.items() if isinstance(value, (int, float))})
    aggregates: dict[str, Any] = {"num_runs": len(runs), "runs": runs}
    for key in numeric_keys:
        values = [float(run[key]) for run in flattened if isinstance(run.get(key), (int, float))]
        values_sorted = sorted(values)
        aggregates[f"{key}_mean"] = mean(values)
        aggregates[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregates[f"{key}_median"] = median(values)
        aggregates[f"{key}_min"] = values_sorted[0]
        aggregates[f"{key}_max"] = values_sorted[-1]
        if len(values) > 1:
            stderr = aggregates[f"{key}_std"] / (len(values) ** 0.5)
            margin = 1.96 * stderr
            aggregates[f"{key}_ci95_low"] = aggregates[f"{key}_mean"] - margin
            aggregates[f"{key}_ci95_high"] = aggregates[f"{key}_mean"] + margin
        else:
            aggregates[f"{key}_ci95_low"] = values[0]
            aggregates[f"{key}_ci95_high"] = values[0]
    return aggregates


def _write_runs_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        seed = run.get("seed")
        encoding = run.get("encoding")
        if "metrics" in run and isinstance(run["metrics"], dict):
            for key, value in run["metrics"].items():
                rows.append({"seed": seed, "encoding": encoding, "scope": "metrics", "name": key, "value": value})
        if "summary" in run and isinstance(run["summary"], dict):
            for key, value in run["summary"].items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            rows.append({"seed": seed, "encoding": encoding, "scope": key, "name": subkey, "value": subvalue})
                elif isinstance(value, (int, float)):
                    rows.append({"seed": seed, "encoding": encoding, "scope": "summary", "name": key, "value": value})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "encoding", "scope", "name", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_runs_by_encoding(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run.get("encoding", "unknown")), []).append(run)
    return {encoding: _aggregate_runs(group) for encoding, group in grouped.items()}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return payload


def load_training_config(payload: dict[str, Any]) -> TrainingExperimentConfig:
    return _normalize_config(
        TrainingExperimentConfig(
            dataset_path=payload.get("dataset_path"),
            num_clients=int(payload.get("num_clients", 5)),
            num_rounds=int(payload.get("num_rounds", 3)),
            seed=int(payload.get("seed", 42)),
            seeds=[int(seed) for seed in payload.get("seeds", [])] or None,
            encoding_modes=[str(mode) for mode in payload.get("encoding_modes", [])] or None,
            data_reuploads=int(payload.get("data_reuploads", 1)),
            prefer_gpu=bool(payload.get("prefer_gpu", True)),
            max_samples_per_client=payload.get("max_samples_per_client"),
            feature_mode=str(payload.get("feature_mode", "quadrants")),
            n_features=int(payload.get("n_features", 8)),
            num_layers=int(payload.get("num_layers", 2)),
            train_epochs=int(payload.get("train_epochs", 6)),
            train_lr=float(payload.get("train_lr", 0.2)),
            output_dir=str(payload.get("output_dir", "experiments/qfl_training/outputs")),
        )
    )


def load_unlearning_config(payload: dict[str, Any]) -> UnlearningExperimentConfig:
    return _normalize_config(
        UnlearningExperimentConfig(
            dataset_path=payload.get("dataset_path"),
            num_clients=int(payload.get("num_clients", 5)),
            num_rounds=int(payload.get("num_rounds", 3)),
            excluded_client_id=str(payload.get("excluded_client_id", "client_0")),
            seed=int(payload.get("seed", 42)),
            seeds=[int(seed) for seed in payload.get("seeds", [])] or None,
            encoding_modes=[str(mode) for mode in payload.get("encoding_modes", [])] or None,
            data_reuploads=int(payload.get("data_reuploads", 1)),
            prefer_gpu=bool(payload.get("prefer_gpu", True)),
            max_samples_per_client=payload.get("max_samples_per_client"),
            feature_mode=str(payload.get("feature_mode", "quadrants")),
            n_features=int(payload.get("n_features", 8)),
            num_layers=int(payload.get("num_layers", 2)),
            train_epochs=int(payload.get("train_epochs", 6)),
            train_lr=float(payload.get("train_lr", 0.2)),
            shap_permutations=int(payload.get("shap_permutations", 16)),
            unlearn_lr=float(payload.get("unlearn_lr", 0.4)),
            unlearn_max_steps=int(payload.get("unlearn_max_steps", 12)),
            mask_quantile=float(payload.get("mask_quantile", 0.5)),
            output_dir=str(payload.get("output_dir", "experiments/qfl_unlearning_qfi/outputs")),
        )
    )

def _build_clients(num_clients: int, dataset_path: str | None = None, prefer_gpu: bool = True, encoding: str = "angle", data_reuploads: int = 1, max_samples_per_client: int | None = None, feature_mode: str = "quadrants", n_features: int = 8, train_epochs: int = 6, train_lr: float = 0.2, num_layers: int = 2, seed: int = 0):
    if feature_mode == "binary":
        client_splits = load_femnist_binary_partitions(num_clients=num_clients, n_features=n_features, seed=seed)
    elif feature_mode == "noniid":
        client_splits = load_femnist_noniid_partitions(num_clients=num_clients, n_features=n_features, seed=seed)
    elif feature_mode == "backdoor":
        client_splits = load_femnist_backdoor_partitions(num_clients=num_clients, n_features=n_features, seed=seed, per_client=max_samples_per_client or 120)
    elif dataset_path:
        dataset_file = Path(dataset_path)
        if dataset_file.exists():
            x, y = load_femnist_source(dataset_file)
        else:
            LOGGER.warning("dataset_path=%s not found; falling back to remote FEMNIST partitions", dataset_path)
            x = y = None
        if x is not None and y is not None:
            if x.ndim == 3:
                x = compress_to_quadrants(normalize_images(x))
            client_splits = partition_by_client(x, y, num_clients=num_clients)
        else:
            client_splits = load_femnist_partitions(num_clients=num_clients, feature_mode=feature_mode, n_features=n_features, seed=seed)
    else:
        client_splits = load_femnist_partitions(num_clients=num_clients, feature_mode=feature_mode, n_features=n_features, seed=seed)
    if max_samples_per_client:
        client_splits = [
            FEMNISTSplit(split.client_id, split.x[:max_samples_per_client], split.y[:max_samples_per_client], split.x_eval, split.y_eval)
            for split in client_splits
        ]
    clients = [
        FederatedClient(
            split.client_id,
            split.x,
            split.y,
            prefer_gpu=prefer_gpu,
            encoding=encoding,
            data_reuploads=data_reuploads,
            num_layers=num_layers,
            epochs=train_epochs,
            lr=train_lr,
            x_eval=split.x_eval,
            y_eval=split.y_eval,
        )
        for split in client_splits
    ]
    from qfl.quantum.model import QuantumClassifier

    weight_shape = QuantumClassifier.weight_shape(num_layers, client_splits[0].x.shape[1], encoding, data_reuploads)
    initial_weights = 0.1 * np.random.randn(*weight_shape)
    server = FederatedServer(initial_weights=initial_weights)
    return client_splits, clients, server


def _build_model_from_weights(num_wires: int, weights: np.ndarray, prefer_gpu: bool, encoding: str, data_reuploads: int, num_layers: int = 2):
    from qfl.quantum.model import QuantumClassifier

    model = QuantumClassifier(num_wires=num_wires, num_layers=num_layers, prefer_gpu=prefer_gpu, encoding=encoding, data_reuploads=data_reuploads)
    model.weights = np.asarray(weights, dtype=float).reshape(model.weights.shape)
    return model


def run_training_experiment(config: TrainingExperimentConfig, checkpoint_name: str = "training_state.json") -> dict[str, Any]:
    config = _normalize_config(config)
    output_dir = ensure_dir(config.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    runs: list[dict[str, Any]] = []
    for encoding in _encoding_list(config.encoding_modes):
        for seed in _seed_list(config.seed, config.seeds):
            set_seed(seed)
            LOGGER.info(
                "starting training seed=%s encoding=%s num_clients=%s num_rounds=%s",
                seed,
                encoding,
                config.num_clients,
                config.num_rounds,
            )
            _, clients, server = _build_clients(config.num_clients, config.dataset_path, config.prefer_gpu, encoding, config.data_reuploads, config.max_samples_per_client, config.feature_mode, config.n_features, config.train_epochs, config.train_lr, config.num_layers, seed)
            run = FederatedTrainingRun(server=server, clients=clients)
            progress = ProgressTracker(label=f"QFL training ({encoding}, seed={seed})", total_steps=config.num_rounds)
            seed_checkpoint_path = checkpoint_dir / f"{Path(checkpoint_name).stem}_{encoding}_seed{seed}.json"
            progress_checkpoint_path = seed_checkpoint_path.with_suffix(".txt")

            def save_round_progress(result) -> None:
                payload = progress.checkpoint_payload(
                    result.round_index + 1,
                    {"status": "running", "seed": seed, "encoding": encoding, "metrics": result.metrics},
                )
                save_checkpoint(seed_checkpoint_path, payload)
                save_progress_checkpoint(progress_checkpoint_path, payload)

            partial = run.run(num_rounds=config.num_rounds, on_round_complete=save_round_progress)
            round_metrics = [result.metrics for result in partial]
            for result in partial:
                LOGGER.info(
                    "training seed=%s encoding=%s round=%s metrics=%s",
                    seed,
                    encoding,
                    result.round_index,
                    result.metrics,
                )
            seed_summary = {
                "seed": seed,
                "encoding": encoding,
                "rounds": round_metrics,
                "metrics": round_metrics[-1] if round_metrics else {},
            }
            runs.append(seed_summary)
            final_payload = progress.checkpoint_payload(
                config.num_rounds,
                {"status": "completed", "seed": seed, "encoding": encoding, "metrics": seed_summary["metrics"], "summary": seed_summary},
            )
            save_checkpoint(seed_checkpoint_path, final_payload)
            save_progress_checkpoint(progress_checkpoint_path, final_payload)

    summary = {
        "metadata": experiment_metadata(config),
        "ablation": _aggregate_runs_by_encoding(runs),
        **_aggregate_runs(runs),
    }
    write_json(output_dir / "training_summary.json", summary)
    write_json(log_dir / "training_runs.json", {"runs": runs})
    _write_runs_csv(log_dir / "training_runs.csv", runs)
    return summary


def run_unlearning_experiment(config: UnlearningExperimentConfig, checkpoint_name: str = "unlearning_state.json") -> dict[str, Any]:
    config = _normalize_config(config)
    output_dir = ensure_dir(config.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    runs: list[dict[str, Any]] = []
    for encoding in _encoding_list(config.encoding_modes):
        for seed in _seed_list(config.seed, config.seeds):
            set_seed(seed)
            LOGGER.info(
                "starting unlearning seed=%s encoding=%s num_clients=%s num_rounds=%s excluded_client=%s",
                seed,
                encoding,
                config.num_clients,
                config.num_rounds,
                config.excluded_client_id,
            )
            _, clients, server = _build_clients(config.num_clients, config.dataset_path, config.prefer_gpu, encoding, config.data_reuploads, config.max_samples_per_client, config.feature_mode, config.n_features, config.train_epochs, config.train_lr, config.num_layers, seed)
            training_run = FederatedTrainingRun(server=server, clients=clients)
            unlearning_run = QFIUnlearningRun(
                training_run=training_run,
                excluded_client_id=config.excluded_client_id,
                prefer_gpu=config.prefer_gpu,
                encoding=encoding,
                data_reuploads=config.data_reuploads,
            )
            progress = ProgressTracker(label=f"QFI unlearning ({encoding}, seed={seed})", total_steps=config.num_rounds)
            seed_checkpoint_path = checkpoint_dir / f"{Path(checkpoint_name).stem}_{encoding}_seed{seed}.json"
            progress_checkpoint_path = seed_checkpoint_path.with_suffix(".txt")

            def save_round_progress(result) -> None:
                payload = progress.checkpoint_payload(
                    result.round_index + 1,
                    {"status": "running", "seed": seed, "encoding": encoding, "metrics": result.metrics},
                )
                save_checkpoint(seed_checkpoint_path, payload)
                save_progress_checkpoint(progress_checkpoint_path, payload)

            base_result = unlearning_run.run(num_rounds=config.num_rounds, on_round_complete=save_round_progress)
            final_weights = np.asarray(base_result["global_weights"], dtype=float)
            model = _build_model_from_weights(
                clients[0].x_train.shape[1],
                final_weights,
                config.prefer_gpu,
                encoding,
                config.data_reuploads,
                config.num_layers,
            )
            split = _split_clients(clients, config.excluded_client_id)
            baselines = _run_unlearning_baselines(
                model,
                split,
                config.num_rounds,
                config.prefer_gpu,
                encoding,
                config.data_reuploads,
                config.num_layers,
                shap_permutations=config.shap_permutations,
                unlearn_lr=config.unlearn_lr,
                unlearn_max_steps=config.unlearn_max_steps,
                mask_quantile=config.mask_quantile,
                seed=seed,
            )
            seed_result = {"main": base_result, "baselines": baselines}
            LOGGER.info("unlearning seed=%s encoding=%s result_keys=%s", seed, encoding, sorted(seed_result.keys()))
            seed_summary = {"seed": seed, "encoding": encoding, "summary": seed_result}
            runs.append(seed_summary)
            final_payload = progress.checkpoint_payload(
                config.num_rounds,
                {"status": "completed", "seed": seed, "encoding": encoding, "summary": seed_summary},
            )
            save_checkpoint(seed_checkpoint_path, final_payload)
            save_progress_checkpoint(progress_checkpoint_path, final_payload)

    summary = {
        "metadata": experiment_metadata(config),
        "ablation": _aggregate_runs_by_encoding(runs),
        "baselines": baselines_summary(runs),
        "baselines_table": baselines_table(runs),
        "baselines_aggregated": aggregate_baselines(runs),
        **_aggregate_runs(runs),
    }
    write_json(output_dir / "unlearning_summary.json", summary)
    write_json(log_dir / "unlearning_runs.json", {"runs": runs})
    _write_runs_csv(log_dir / "unlearning_runs.csv", runs)
    return summary


@dataclass(frozen=True)
class _ClientSplit:
    active_clients: list[FederatedClient]
    forget_x: np.ndarray
    forget_y: np.ndarray
    forget_x_eval: np.ndarray
    forget_y_eval: np.ndarray
    retain_x: np.ndarray
    retain_y: np.ndarray


def _split_clients(clients: list[FederatedClient], excluded_client_id: str) -> _ClientSplit:
    excluded = [c for c in clients if c.client_id == excluded_client_id]
    active = [c for c in clients if c.client_id != excluded_client_id]
    if len(excluded) != 1:
        raise ValueError(f"excluded_client_id must identify exactly one client; got {excluded_client_id!r}")
    if not active:
        raise ValueError("At least one active client is required")
    n_features = active[0].x_train.shape[1]
    return _ClientSplit(
        active_clients=active,
        forget_x=excluded[0].x_train if excluded else np.empty((0, n_features)),
        forget_y=excluded[0].y_train if excluded else np.empty((0,)),
        forget_x_eval=excluded[0].x_eval if excluded and excluded[0].x_eval is not None else np.empty((0, n_features)),
        forget_y_eval=excluded[0].y_eval if excluded and excluded[0].y_eval is not None else np.empty((0,)),
        retain_x=np.concatenate([c.x_train for c in active], axis=0),
        retain_y=np.concatenate([c.y_train for c in active], axis=0),
    )


def _run_unlearning_baselines(
    model,
    split: _ClientSplit,
    num_rounds: int,
    prefer_gpu: bool,
    encoding: str,
    data_reuploads: int,
    num_layers: int,
    shap_permutations: int = 16,
    unlearn_lr: float = 0.4,
    unlearn_max_steps: int = 12,
    mask_quantile: float = 0.5,
    seed: int = 0,
):
    from copy import deepcopy

    # The retrain reference is calculated first and passed into every mutating
    # variant. It is therefore a genuine stopping/evaluation reference, not a
    # post-hoc number reported after a majority-class proxy was used.
    set_seed(seed)
    retrained_model = _retrain_without_excluded(split.active_clients, num_rounds, prefer_gpu, encoding, data_reuploads, num_layers)
    forget_acc = evaluate_accuracy(retrained_model, split.forget_x, split.forget_y)
    retain_acc = evaluate_accuracy(retrained_model, split.retain_x, split.retain_y)
    mia_auc_val = membership_inference_auc(retrained_model, split.forget_x, split.forget_y, split.forget_x_eval, split.forget_y_eval)
    qfi_trace_val = retrained_model.qfi_trace(split.forget_x[: min(8, len(split.forget_x))]) if len(split.forget_x) else 0.0
    retrain_reference = {
        "forget_accuracy": forget_acc,
        "forget_accuracy_after": forget_acc,
        "retain_accuracy": retain_acc,
        "retain_accuracy_after": retain_acc,
        "mia_auc": mia_auc_val,
        "mia_auc_after": mia_auc_val,
        "qfi_trace": qfi_trace_val,
        "qfi_trace_after": qfi_trace_val,
    }

    baselines: dict[str, Any] = {}
    for mode in ("no_unlearning", "shap_only", "qfi_only", "shap_qfi"):
        baseline_model = deepcopy(model)
        report, _ = HybridSHAPQFIUnlearner(baseline_model).run(
            split.forget_x, split.forget_y, split.retain_x, split.retain_y,
            x_forget_eval=split.forget_x_eval, y_forget_eval=split.forget_y_eval,
            num_shap_permutations=shap_permutations, mask_quantile=mask_quantile,
            lr=unlearn_lr, max_steps=unlearn_max_steps, mode=mode, seed=seed,
            retrain_forget_accuracy=forget_acc, retrain_retain_accuracy=retain_acc,
        )
        baselines[mode] = {
            "forget_accuracy_before": report.forget_accuracy_before,
            "forget_accuracy_after": report.forget_accuracy_after,
            "forget_loss_before": report.forget_loss_before,
            "forget_loss_after": report.forget_loss_after,
            "retain_accuracy_before": report.retain_accuracy_before,
            "retain_accuracy_after": report.retain_accuracy_after,
            "retain_loss_before": report.retain_loss_before,
            "retain_loss_after": report.retain_loss_after,
            "mia_auc_before": report.mia_auc_before,
            "mia_auc_after": report.mia_auc_after,
            "qfi_trace_before": report.qfi_trace_before,
            "qfi_trace_after": report.qfi_trace_after,
            "retrain_forget_accuracy": report.retrain_forget_accuracy,
            "retrain_retain_accuracy": report.retrain_retain_accuracy,
            "qfi_condition_number": report.qfi_condition_number,
            "accepted_step_sizes": list(report.accepted_step_sizes),
            "unlearning_steps": float(report.unlearning_steps),
            "shap_drop_mean": report.shap_drop_mean,
        }
    baselines["retrain_complete"] = retrain_reference
    return baselines


def _retrain_without_excluded(
    active_clients: list[FederatedClient],
    num_rounds: int,
    prefer_gpu: bool,
    encoding: str,
    data_reuploads: int,
    num_layers: int,
):
    from qfl.quantum.model import QuantumClassifier

    weight_shape = QuantumClassifier.weight_shape(num_layers, active_clients[0].x_train.shape[1], encoding, data_reuploads)
    server = FederatedServer(initial_weights=0.1 * np.random.randn(*weight_shape))
    for client in active_clients:
        client.prefer_gpu = prefer_gpu
        client.encoding = encoding
        client.data_reuploads = data_reuploads
    run = FederatedTrainingRun(server=server, clients=active_clients)
    partial = run.run(num_rounds=num_rounds)
    model = QuantumClassifier(
        num_wires=active_clients[0].x_train.shape[1],
        num_layers=num_layers,
        prefer_gpu=prefer_gpu,
        encoding=encoding,
        data_reuploads=data_reuploads,
    )
    model.weights = np.asarray(partial[-1].global_weights, dtype=float).reshape(model.weights.shape)
    return model


def baselines_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for run in runs:
        summary[f'{run.get("encoding", "unknown")}-seed{run["seed"]}'] = run.get("summary", {}).get("baselines", {})
    return summary


def baselines_table(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        seed = run["seed"]
        encoding = run.get("encoding")
        baselines = run.get("summary", {}).get("baselines", {})
        for mode, metrics in baselines.items():
            if not isinstance(metrics, dict):
                continue
            row = {"seed": seed, "encoding": encoding, "mode": mode}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = float(value)
            rows.append(row)
    return rows


def aggregate_baselines(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean +/- std of every baseline metric across seeds, per (encoding, mode).

    This is the paper-facing comparison table: for each encoding and each
    unlearning variant it summarises the metrics over the repeated seeds.
    """
    grouped: dict[str, dict[str, list[dict[str, float]]]] = {}
    for row in baselines_table(runs):
        enc = str(row.get("encoding", "unknown"))
        mode = str(row.get("mode", "unknown"))
        metrics = {k: v for k, v in row.items() if k not in ("seed", "encoding", "mode")}
        grouped.setdefault(enc, {}).setdefault(mode, []).append(metrics)

    aggregated: dict[str, Any] = {}
    for enc, modes in grouped.items():
        aggregated[enc] = {}
        for mode, metric_dicts in modes.items():
            keys = sorted({k for d in metric_dicts for k in d})
            stats: dict[str, float] = {"num_seeds": float(len(metric_dicts))}
            for key in keys:
                values = [d[key] for d in metric_dicts if key in d]
                stats[f"{key}_mean"] = mean(values)
                stats[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
            aggregated[enc][mode] = stats
    return aggregated
