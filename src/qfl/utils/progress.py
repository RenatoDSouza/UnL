"""Progress tracking helpers for long-running experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic


@dataclass
class ProgressTracker:
    label: str
    total_steps: int
    started_at: float = field(default_factory=monotonic)

    def eta_seconds(self, current_step: int) -> float | None:
        if current_step <= 0:
            return None
        elapsed = monotonic() - self.started_at
        avg = elapsed / current_step
        remaining = max(self.total_steps - current_step, 0)
        return remaining * avg

    def progress_line(self, current_step: int) -> str:
        current_step = min(max(current_step, 0), self.total_steps)
        width = 30
        filled = int(width * current_step / max(self.total_steps, 1))
        bar = "#" * filled + "-" * (width - filled)
        percent = 100 * current_step / max(self.total_steps, 1)
        eta = self.eta_seconds(current_step)
        eta_part = "ETA: --" if eta is None else f"ETA: {eta:6.1f}s"
        return f"{self.label}: [{bar}] {percent:5.1f}% ({current_step}/{self.total_steps}) {eta_part}"

    def checkpoint_payload(self, current_step: int, extra: dict[str, object] | None = None) -> dict[str, object]:
        payload = {
            "label": self.label,
            "current_step": current_step,
            "total_steps": self.total_steps,
            "eta_seconds": self.eta_seconds(current_step),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        return payload
