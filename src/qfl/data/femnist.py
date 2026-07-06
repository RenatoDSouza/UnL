"""FEMNIST loading and federated partitioning helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FEMNISTSplit:
    client_id: str
    x: np.ndarray
    y: np.ndarray


def load_femnist_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(path))
    return data["x"], data["y"]


def _reshape_leaf_image(image: list[float] | list[int]) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.size != 28 * 28:
        raise ValueError(f"Expected flattened 28x28 image, got {arr.size} values")
    return arr.reshape(28, 28)


def load_femnist_leaf_json(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load FEMNIST from LEAF JSON files and reconstruct 28x28 images.

    The LEAF preprocessing pipeline stores federated samples as JSON files with
    `users`, `num_samples`, and `user_data` fields. Each image is a flattened
    784-element list. This loader concatenates all available samples in file
    order and returns image and label arrays compatible with the current
    experiment pipeline.
    """

    root = Path(data_dir)
    if root.is_file():
        raise ValueError("Expected a directory containing LEAF JSON files, not a file")

    json_files = sorted(root.rglob("all_data*.json"))
    if not json_files:
        json_files = sorted(root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No LEAF JSON files found under {root}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        user_data = payload.get("user_data", {})
        for user in payload.get("users", []):
            sample = user_data.get(user, {})
            xs = sample.get("x", [])
            ys = sample.get("y", [])
            for image, label in zip(xs, ys):
                images.append(_reshape_leaf_image(image))
                labels.append(int(label))

    if not images:
        raise ValueError(f"No samples could be read from {root}")

    return np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)


def load_femnist_source(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load FEMNIST from either a .npz file or a LEAF dataset directory."""

    source = Path(path)
    if source.suffix == ".npz":
        return load_femnist_npz(source)
    return load_femnist_leaf_json(source)


def normalize_images(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / 255.0 if x.max() > 1.0 else x


def flatten_images(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def compress_to_quadrants(x: np.ndarray) -> np.ndarray:
    """Reduce 28x28 FEMNIST images to 4 normalized quadrant features."""
    if x.ndim != 3:
        raise ValueError("Expected images with shape (n_samples, height, width)")
    h_mid = x.shape[1] // 2
    w_mid = x.shape[2] // 2
    quadrants = [
        x[:, :h_mid, :w_mid].mean(axis=(1, 2)),
        x[:, :h_mid, w_mid:].mean(axis=(1, 2)),
        x[:, h_mid:, :w_mid].mean(axis=(1, 2)),
        x[:, h_mid:, w_mid:].mean(axis=(1, 2)),
    ]
    return np.stack(quadrants, axis=1).astype(np.float32)


def partition_by_client(
    x: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    client_prefix: str = "client",
) -> list[FEMNISTSplit]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    indices = np.array_split(np.arange(len(x)), num_clients)
    return [
        FEMNISTSplit(f"{client_prefix}_{idx}", x[split], y[split])
        for idx, split in enumerate(indices)
    ]


def select_active_clients(
    clients: Iterable[FEMNISTSplit],
    excluded_client_id: str | None = None,
) -> list[FEMNISTSplit]:
    return [client for client in clients if client.client_id != excluded_client_id]


def load_femnist_partitions(num_clients: int) -> list[FEMNISTSplit]:
    """Load partitions from flwr-datasets using NaturalIdPartitioner."""
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner

    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")}
    )

    splits = []
    for partition_id in range(num_clients):
        partition = fds.load_partition(partition_id=partition_id, split="train")

        # Convert PIL images to numpy array
        x_raw = np.array([np.asarray(row["image"]) for row in partition], dtype=np.float32)
        y_raw = np.array([row["character"] for row in partition], dtype=np.int64)

        # Apply preprocessing
        x = compress_to_quadrants(normalize_images(x_raw))
        y = (y_raw > 0).astype(int)

        splits.append(FEMNISTSplit(client_id=f"client_{partition_id}", x=x, y=y))
    return splits

