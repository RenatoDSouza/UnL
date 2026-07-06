"""I/O helpers for experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def update_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def print_progress(label: str, current: int, total: int) -> None:
    total = max(total, 1)
    current = min(max(current, 0), total)
    width = 30
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100 * current / total
    print(f"\r{label}: [{bar}] {percent:5.1f}% ({current}/{total})", end="", flush=True)


def finish_progress(label: str) -> None:
    print(f"\r{label}: [##############################] 100.0%")
