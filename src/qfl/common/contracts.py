"""Core type aliases and protocol-like contracts for the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ClientConfig:
    client_id: str
    num_samples: int
    is_active: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    num_clients: int = 5
    num_rounds: int = 1
    batch_size: int = 32
    seed: int = 42


Weights = Sequence[float]

