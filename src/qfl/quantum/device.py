"""PennyLane device selection with GPU preference."""

from __future__ import annotations

import logging

import pennylane as qml

LOGGER = logging.getLogger(__name__)


def create_device(num_wires: int, prefer_gpu: bool = True) -> qml.Device:
    if prefer_gpu:
        try:
            device = qml.device("lightning.gpu", wires=num_wires)
            LOGGER.info("Using PennyLane device: lightning.gpu")
            return device
        except Exception as exc:
            raise RuntimeError(
                "GPU execution was requested, but PennyLane lightning.gpu is not available. "
                "Install pennylane-lightning-gpu and use a CUDA/PyTorch stack compatible "
                "with the NVIDIA driver, or set prefer_gpu=false in the experiment config."
            ) from exc

    try:
        device = qml.device("lightning.qubit", wires=num_wires)
        LOGGER.info("Using PennyLane device: lightning.qubit")
        return device
    except Exception:
        LOGGER.info("Using PennyLane device: default.qubit")
        return qml.device("default.qubit", wires=num_wires)
