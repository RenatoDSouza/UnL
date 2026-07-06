"""Checkpoint helpers for resumable experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qfl.utils.io import read_json, write_json


def checkpoint_exists(path: str | Path) -> bool:
    return Path(path).exists()


def load_checkpoint(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return read_json(path)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
