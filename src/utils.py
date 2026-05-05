"""Common utilities: config loading, seeding, device, JSON IO."""
from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def save_yaml(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def append_csv_row(row: dict, path: str | Path) -> None:
    """Append a single row to a CSV file, writing header if file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    keys = list(row.keys())
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write(",".join(keys) + "\n")
        f.write(",".join(str(row[k]) for k in keys) + "\n")


def run_id_for(config: dict, seed: int) -> str:
    exp_id = config.get("experiment", {}).get("id", "exp")
    data = config.get("data", {})
    mode = data.get("partition", "dirichlet")
    if mode == "iid":
        ptag = "iid"
    elif mode == "dirichlet":
        alpha = data.get("alpha", "?")
        ptag = f"dir{alpha}"
    else:
        ptag = mode
    return f"{exp_id}__{ptag}__seed{seed}"
