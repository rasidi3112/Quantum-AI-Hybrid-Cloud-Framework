"""Logging utilities for consistent experiment tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List


def create_logger(log_path: Path | None) -> logging.Logger:
    """Create a configured logger.

    Args:
        log_path: Optional path to a log file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger("quantum_ai_hybrid")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)

    return logger


def dump_metrics(metrics: Iterable[dict[str, Any]], path: Path) -> None:
    """Persist metrics to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(list(metrics), fp, indent=2)


def load_metrics_file(path: Path) -> List[dict[str, Any]]:
    """Load metrics JSON file."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)