"""Small JSON and output helpers for the steady 1D scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {path}; pass --overwrite to reuse it")
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in ("run_manifest.json", "history.json", "summary.json", "profiles.csv", "solution.npz", "solution.npz.tmp"):
            candidate = path / name
            if candidate.is_file():
                candidate.unlink()


def write_profiles_csv(path: Path, x: np.ndarray, profiles: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    names = list(profiles)
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["x", *names])
        for index, x_value in enumerate(x):
            writer.writerow([float(x_value), *(float(profiles[name][index]) for name in names)])
    temporary.replace(path)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(destination, **arrays)
    temporary.replace(path)
