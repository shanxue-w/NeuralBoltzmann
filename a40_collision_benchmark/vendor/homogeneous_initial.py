"""Exact initial-condition definitions copied from experiments/homogeneous/run_homogeneous.py."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class VelocityGrid:
    n: int
    L: float
    dv: float
    dv3: float
    v: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    v2: np.ndarray
    psi: np.ndarray

def make_grid(n: int, L: float) -> VelocityGrid:
    dv = 2.0 * L / n
    v = (np.arange(n, dtype=np.float64) - n // 2) * dv
    vx, vy, vz = np.meshgrid(v, v, v, indexing="ij")
    v2 = vx * vx + vy * vy + vz * vz
    psi = np.stack((np.ones_like(vx), vx / L, vy / L, vz / L, v2 / (L * L)), axis=-1)
    return VelocityGrid(n, L, dv, dv**3, v, vx, vy, vz, v2, psi)

def integrate(value: np.ndarray, grid: VelocityGrid) -> float:
    return float(grid.dv3 * value.sum())

def normalized_maxwellian(grid: VelocityGrid, center: tuple[float, float, float], temperature: float) -> np.ndarray:
    exponent = -((grid.vx - center[0]) ** 2 + (grid.vy - center[1]) ** 2 + (grid.vz - center[2]) ** 2) / (2.0 * temperature)
    value = np.exp(exponent) / (2.0 * np.pi * temperature) ** 1.5
    return value / integrate(value, grid)

def initial_condition(name: str, grid: VelocityGrid) -> tuple[np.ndarray, dict[str, Any]]:
    data = {
        "two_stream": ((0.5, 0.5), ((2.5, 0.0, 0.0), (-2.5, 0.0, 0.0))),
        "three_stream": ((0.4, 0.35, 0.25), ((2.5, 0.0, 0.0), (-1.6, 1.4, 0.0), (-1.76, -1.96, 0.0))),
    }
    weights, centers = data[name]
    temperature = 0.6
    value = sum(weight * normalized_maxwellian(grid, center, temperature) for weight, center in zip(weights, centers))
    return value, {"weights": list(weights), "centers": [list(center) for center in centers], "component_temperature": temperature}
