"""PennyLane device selection.

Production experiments use ``lightning.gpu`` exclusively.  Falling back to a
CPU simulator would change both the advertised execution environment and the
runtime characteristics, so it is an explicit opt-in reserved for tests.
"""

from __future__ import annotations

import logging

import pennylane as qml

LOGGER = logging.getLogger(__name__)


def create_device(num_wires: int, prefer_gpu: bool = True) -> qml.Device:
    """Create a PennyLane device, requiring CUDA when ``prefer_gpu`` is true."""
    if prefer_gpu:
        try:
            device = qml.device("lightning.gpu", wires=num_wires)
        except Exception as exc:  # pragma: no cover - depends on host CUDA setup
            raise RuntimeError(
                "CUDA execution is required, but PennyLane device 'lightning.gpu' "
                "could not be created. Install a CUDA-compatible pennylane-lightning "
                "build and expose the NVIDIA GPU (RTX 3080) to this process. "
                "Use prefer_gpu=False only for unit tests."
            ) from exc
        LOGGER.info("Using required CUDA PennyLane device: lightning.gpu")
        return device

    for name in ("lightning.qubit", "default.qubit"):
        try:
            device = qml.device(name, wires=num_wires)
            LOGGER.info("Using explicit CPU PennyLane device: %s", name)
            return device
        except Exception:
            LOGGER.debug("PennyLane device %s unavailable", name, exc_info=True)
    raise RuntimeError("No CPU PennyLane device could be created")
