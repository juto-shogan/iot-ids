"""Utility helpers for paths, logging, configuration, and persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

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


def ensure_directories(models_dir: Path = MODELS_DIR, outputs_dir: Path = OUTPUTS_DIR) -> None:
    """Create runtime directories required by the pipeline."""
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Serialize dictionary as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON content from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file from disk."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_run_metadata(config: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a run metadata payload for reproducibility and auditability."""
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    if extra:
        payload["extra"] = extra
    return payload
