"""FEMNIST loading and federated partitioning helpers."""

from __future__ import annotations

from dataclasses import dataclass
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
