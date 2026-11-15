"""Couette, Fourier, and custom diffuse-wall configurations."""

from __future__ import annotations

from dataclasses import dataclass

from .numerics_1d import WallCondition


@dataclass(frozen=True)
class Case1D:
    name: str
    description: str
    left_wall: WallCondition
    right_wall: WallCondition


CASES = {
    "couette": Case1D("couette", "oppositely moving isothermal walls", WallCondition((0.0, -0.5, 0.0), 1.0), WallCondition((0.0, 0.5, 0.0), 1.0)),
    "fourier": Case1D("fourier", "stationary walls with a temperature difference", WallCondition((0.0, 0.0, 0.0), 1.0), WallCondition((0.0, 0.0, 0.0), 2.0)),
    "custom": Case1D("custom", "user-configured Maxwell walls", WallCondition((0.0, 0.0, 0.0), 1.0), WallCondition((0.0, 0.0, 0.0), 1.0)),
}


def case_preset(name: str) -> Case1D:
    return CASES[name]


def with_wall_overrides(case: Case1D, *, left_velocity: tuple[float, float, float] | None,
                        right_velocity: tuple[float, float, float] | None,
                        left_temperature: float | None, right_temperature: float | None) -> Case1D:
    left = WallCondition(left_velocity if left_velocity is not None else case.left_wall.velocity, left_temperature if left_temperature is not None else case.left_wall.temperature)
    right = WallCondition(right_velocity if right_velocity is not None else case.right_wall.velocity, right_temperature if right_temperature is not None else case.right_wall.temperature)
    return Case1D(case.name, case.description, left, right)
