"""Checkpoint helpers for resumable experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

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


def save_progress_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a human-readable snapshot of an experiment's current progress."""

    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Status: {payload.get('status', 'running')}",
        f"Experimento: {payload.get('label', 'unknown')}",
        f"Progresso: {payload.get('current_step', 0)}/{payload.get('total_steps', 0)}",
        f"Atualizado (UTC): {payload.get('updated_at_utc', 'unknown')}",
    ]
    eta = payload.get("eta_seconds")
    if eta is not None:
        lines.append(f"ETA (segundos): {float(eta):.1f}")
    for key in ("seed", "encoding", "metrics"):
        if key in payload:
            value = payload[key]
            lines.append(f"{key}: {json.dumps(value, sort_keys=True)}")
    checkpoint.write_text("\n".join(lines) + "\n", encoding="utf-8")
