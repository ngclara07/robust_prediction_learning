from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds used by the project."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as formatted JSON."""
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
