"""Checkpoint management utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch # type: ignore


@dataclass
class CheckpointManager:
    """Manage saving and tracking checkpoints."""
    directory: Path

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path: Optional[Path] = None

    def save(self, state: Dict[str, Any], filename: str) -> Path:
        """Save a checkpoint file."""
        path = self.directory / filename
        torch.save(state, path)
        return path

    def save_best(self, model_state: Dict[str, Any], metrics: Dict[str, float], signature: Dict[str, Any]) -> Path:
        """Save the best-performing checkpoint."""
        state = {
            "model_state": model_state,
            "metrics": metrics,
            "signature": signature,
        }
        path = self.directory / "best.pt"
        torch.save(state, path)
        self.best_checkpoint_path = path
        return path