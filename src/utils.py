"""Utility helpers for paths, logging, and persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logger formatting for the project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directories() -> None:
    """Create runtime directories required by the pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Serialize dictionary as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON content from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
