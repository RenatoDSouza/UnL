"""PennyLane device selection with GPU preference."""

from __future__ import annotations

import logging

import pennylane as qml

LOGGER = logging.getLogger(__name__)


def create_device(num_wires: int, prefer_gpu: bool = True) -> qml.Device:
    """Select the fastest available PennyLane device.

    When ``prefer_gpu`` is set we try ``lightning.gpu`` first, but a missing GPU
    backend is no longer fatal: we fall back to the CPU state-vector simulators
    with a warning. For the small (few-qubit), single-sample circuits used here
    ``lightning.qubit`` on CPU is typically faster than ``lightning.gpu``, whose
    per-call kernel-launch overhead dominates.
    """
    candidates = []
    if prefer_gpu:
        candidates.append("lightning.gpu")
    candidates.extend(["lightning.qubit", "default.qubit"])

    last_exc: Exception | None = None
    for name in candidates:
        try:
            device = qml.device(name, wires=num_wires)
            LOGGER.info("Using PennyLane device: %s", name)
            return device
        except Exception as exc:  # pragma: no cover - depends on installed backends
            LOGGER.warning("PennyLane device %s unavailable (%s); trying next fallback.", name, exc)
            last_exc = exc

    raise RuntimeError("No PennyLane device could be created") from last_exc
