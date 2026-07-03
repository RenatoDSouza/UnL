"""PennyLane device selection with GPU preference."""

from __future__ import annotations

import logging

import pennylane as qml

LOGGER = logging.getLogger(__name__)


def create_device(num_wires: int, prefer_gpu: bool = True) -> qml.Device:
    if prefer_gpu:
        for name in ("lightning.gpu", "lightning.qubit"):
            try:
                device = qml.device(name, wires=num_wires)
                LOGGER.info("Using PennyLane device: %s", name)
                return device
            except Exception:
                continue
    return qml.device("default.qubit", wires=num_wires)

