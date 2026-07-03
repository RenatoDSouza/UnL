"""Project-wide data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClientUpdate:
    client_id: str
    num_samples: int
    weights: list[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FederatedResult:
    round_index: int
    global_weights: list[float]
    metrics: dict[str, float]

