"""Kinetic cavity initialization and the uniform-grid solver shared by both entry points."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from UnsteadyBoltzmann.runtime import run_fixed_time


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "boltzmann_tail_simplified_gate_fno_w36_l3_neuralop_10000_b8_s1234_bkw"
    / "best.pt"
)


class LidDrivenCavity2D:
    """Case-specific 2D finite-volume kinetic cavity solver."""

    def __init__(
        self,
        collision: Any,
        *,
        nx: int = 32,
        ny: int = 32,
        nv: int = 32,
        velocity_half_width: float = 10.0,
        kn: float = 0.1,
        wall_temperature: float = 1.0,
        lid_speed: float = 0.15,
        time_integrator: str = "euler",
        device: str | torch.device = "cuda",
    ) -> None:
        self.collision = collision
        self.nx = int(nx)
        self.ny = int(ny)
        self.nv = int(nv)
        self.velocity_half_width = float(velocity_half_width)
        self.kn = float(kn)
        self.wall_temperature = float(wall_temperature)
        self.lid_speed = float(lid_speed)
        self.time_integrator = time_integrator
        requested_device = torch.device(device)

        self.dx = 1.0 / self.nx
        self.dy = 1.0 / self.ny
        self.x = (torch.arange(self.nx, dtype=torch.float64, device=requested_device) + 0.5) * self.dx
        self.y = (torch.arange(self.ny, dtype=torch.float64, device=requested_device) + 0.5) * self.dy
        self.device = self.x.device

        self.dv = 2.0 * self.velocity_half_width / self.nv
        self.dv3 = self.dv**3
        self.v = (torch.arange(self.nv, dtype=torch.float64, device=self.device) - self.nv // 2) * self.dv
        self.vx, self.vy, self.vz = torch.meshgrid(self.v, self.v, self.v, indexing="ij")
        self.v2 = self.vx.square() + self.vy.square() + self.vz.square()
        self.positive_vx = self.vx.clamp_min(0.0)
        self.negative_vx = self.vx.clamp_max(0.0)
        self.positive_vy = self.vy.clamp_min(0.0)
        self.negative_vy = self.vy.clamp_max(0.0)

        self.initial_maxwellian = self._maxwellian(density=1.0, velocity=(0.0, 0.0, 0.0), temperature=1.0)
        self.stationary_wall_unit = self._maxwellian(density=1.0, velocity=(0.0, 0.0, 0.0), temperature=self.wall_temperature)
        self.top_wall_unit = self._maxwellian(density=1.0, velocity=(self.lid_speed, 0.0, 0.0), temperature=self.wall_temperature)
        self.left_incoming_flux_unit = self.dv3 * torch.sum(self.positive_vx * self.stationary_wall_unit)
        self.right_incoming_flux_unit = self.dv3 * torch.sum(self.negative_vx * self.stationary_wall_unit)
        self.bottom_incoming_flux_unit = self.dv3 * torch.sum(self.positive_vy * self.stationary_wall_unit)
        self.top_incoming_flux_unit = self.dv3 * torch.sum(self.negative_vy * self.top_wall_unit)
        self.transport_cfl = 0.5
        self.maximum_transport_speed = self.velocity_half_width
        self.cfl_time_step = self.transport_cfl / (self.maximum_transport_speed / self.dx + self.maximum_transport_speed / self.dy)
        self.last_nominal_dt: float | None = None
        self.last_number_of_steps = 0

    def _maxwellian(
        self,
        *,
        density: float,
        velocity: tuple[float, float, float],
        temperature: float,
    ) -> torch.Tensor:
        relative_speed_squared = (self.vx - velocity[0]).square() + (self.vy - velocity[1]).square() + (self.vz - velocity[2]).square()
        return density * torch.exp(-relative_speed_squared / (2.0 * temperature)) / (2.0 * math.pi * temperature) ** 1.5

    def initial_state(self) -> torch.Tensor:
        return self.initial_maxwellian[None, None].expand(self.nx, self.ny, self.nv, self.nv, self.nv).clone()

    def moments(self, state: torch.Tensor) -> torch.Tensor:
        """Return [rho, ux, uy, uz, T] on the [Nx,Ny] cell grid."""

        rho = self.dv3 * state.sum(dim=(-3, -2, -1))
        momentum = torch.stack(tuple(
            self.dv3 * torch.sum(state * component[None, None], dim=(-3, -2, -1))
            for component in (self.vx, self.vy, self.vz)), dim=-1)
        velocity = momentum / rho[..., None]
        energy = self.dv3 * torch.sum(state * self.v2[None, None], dim=(-3, -2, -1))
        temperature = (energy / rho - velocity.square().sum(dim=-1)) / 3.0
        return torch.cat((rho[..., None], velocity, temperature[..., None]), dim=-1)

    @staticmethod
    def _minmod(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.where(left * right > 0.0, torch.sign(left) * torch.minimum(left.abs(), right.abs()), 0.0)

    @staticmethod
    def _positive_trace(average: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        theta = torch.where(high < 0.0, average / (average - high), 1.0).clamp_(0.0, 1.0)
        return average + theta * (high - average)

    def _wall_reconstruction(self, state: torch.Tensor):
        near = state[0], state[-1], state[:, 0], state[:, -1]
        next_ = state[1], state[-2], state[:, 1], state[:, -2]
        third = state[2], state[-3], state[:, 2], state[:, -3]
        speed = self.negative_vx, self.positive_vx, self.negative_vy, self.positive_vy
        denominator = self.left_incoming_flux_unit, self.right_incoming_flux_unit, self.bottom_incoming_flux_unit, self.top_incoming_flux_unit
        unit = self.stationary_wall_unit, self.stationary_wall_unit, self.stationary_wall_unit, self.top_wall_unit
        outgoing = tuple(self._positive_trace(a, (11.0 * a - 7.0 * b + 2.0 * c) / 6.0) for a, b, c in zip(near, next_, third))
        density = tuple(-self.dv3 * torch.sum(v[None] * f, dim=(-3, -2, -1)) / d for v, f, d in zip(speed, outgoing, denominator))
        wall = tuple(rho[..., None, None, None] * maxwellian[None] for rho, maxwellian in zip(density, unit))
        incoming = tuple(self._positive_trace(a, 1.25 * a + 0.25 * b - 0.5 * g) for a, b, g in zip(near, next_, wall))
        return dict(zip(("left", "right", "bottom", "top"), density)), outgoing, incoming, wall

    def wall_densities(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._wall_reconstruction(state)[0]

    def transport_interface_fluxes(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        slope_x = torch.zeros_like(state)
        slope_x[1:-1] = self._minmod(state[1:-1] - state[:-2], state[2:] - state[1:-1])
        slope_y = torch.zeros_like(state)
        slope_y[:, 1:-1] = self._minmod(state[:, 1:-1] - state[:, :-2], state[:, 2:] - state[:, 1:-1])

        wall_density, outgoing, incoming, wall = self._wall_reconstruction(state)
        left_out, right_out, bottom_out, top_out = outgoing
        left_in, right_in, bottom_in, top_in = incoming
        left_wall, right_wall, bottom_wall, top_wall = wall

        x_left_trace = state[:-1] + 0.5 * slope_x[:-1]
        x_right_trace = state[1:] - 0.5 * slope_x[1:]
        x_left_trace[0] = torch.where(self.vx > 0.0, left_in, x_left_trace[0])
        x_right_trace[-1] = torch.where(self.vx < 0.0, right_in, x_right_trace[-1])
        x_interior_flux = self.positive_vx[None, None] * x_left_trace + self.negative_vx[None, None] * x_right_trace

        y_bottom_trace = state[:, :-1] + 0.5 * slope_y[:, :-1]
        y_top_trace = state[:, 1:] - 0.5 * slope_y[:, 1:]
        y_bottom_trace[:, 0] = torch.where(self.vy > 0.0, bottom_in, y_bottom_trace[:, 0])
        y_top_trace[:, -1] = torch.where(self.vy < 0.0, top_in, y_top_trace[:, -1])
        y_interior_flux = self.positive_vy[None, None] * y_bottom_trace + self.negative_vy[None, None] * y_top_trace

        left_flux = self.positive_vx[None] * left_wall + self.negative_vx[None] * left_out
        right_flux = self.positive_vx[None] * right_out + self.negative_vx[None] * right_wall
        bottom_flux = self.positive_vy[None] * bottom_wall + self.negative_vy[None] * bottom_out
        top_flux = self.positive_vy[None] * top_out + self.negative_vy[None] * top_wall

        flux_x = torch.cat((left_flux[None], x_interior_flux, right_flux[None]), dim=0)
        flux_y = torch.cat((bottom_flux[:, None], y_interior_flux, top_flux[:, None]), dim=1)
        return flux_x, flux_y, wall_density

    def transport_rhs(self, state: torch.Tensor) -> torch.Tensor:
        flux_x, flux_y, _ = self.transport_interface_fluxes(state)
        return -((flux_x[1:] - flux_x[:-1]) / self.dx + (flux_y[:, 1:] - flux_y[:, :-1]) / self.dy)

    def rhs(self, state: torch.Tensor) -> torch.Tensor:
        """Return the unsplit 2D transport-plus-collision right-hand side."""
        flat_state = state.reshape(self.nx * self.ny, self.nv, self.nv, self.nv)
        flat_collision = self.collision.evaluate(flat_state)
        collision = flat_collision.reshape_as(state)
        return self.transport_rhs(state) + collision / self.kn

    def _forward_euler(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        flat = state.reshape(self.nx * self.ny, self.nv, self.nv, self.nv)
        base = state + (dt / self.kn) * self.collision.evaluate(flat).reshape_as(state)

        flux = self.positive_vx[None, None] * state[:-1] + self.negative_vx[None, None] * state[1:]
        base[:-1] -= (dt / self.dx) * flux
        base[1:] += (dt / self.dx) * flux
        base[0] += (dt / self.dx) * self.negative_vx[None] * state[0]
        base[-1] -= (dt / self.dx) * self.positive_vx[None] * state[-1]
        flux = self.positive_vy[None, None] * state[:, :-1] + self.negative_vy[None, None] * state[:, 1:]
        base[:, :-1] -= (dt / self.dy) * flux
        base[:, 1:] += (dt / self.dy) * flux
        base[:, 0] += (dt / self.dy) * self.negative_vy[None] * state[:, 0]
        base[:, -1] -= (dt / self.dy) * self.positive_vy[None] * state[:, -1]
        minimum = float(base.min())
        if minimum < -1.0e-13:
            raise FloatingPointError(f"positive low-order Euler baseline failed: min(f)={minimum:.6e}; reduce --dt")
        base.clamp_min_(0.0)

        near = state[0], state[-1], state[:, 0], state[:, -1]
        next_ = state[1], state[-2], state[:, 1], state[:, -2]
        third = state[2], state[-3], state[:, 2], state[:, -3]
        outgoing = tuple(self._positive_trace(a, (11.0 * a - 7.0 * b + 2.0 * c) / 6.0) for a, b, c in zip(near, next_, third))
        factors = (dt / self.dx * self.negative_vx, -dt / self.dx * self.positive_vx,
                   dt / self.dy * self.negative_vy, -dt / self.dy * self.positive_vy)
        boundary = base[0], base[-1], base[:, 0], base[:, -1]
        corrections = tuple(factor[None] * (high - average) for factor, high, average in zip(factors, outgoing, near))
        for values, correction in zip(boundary, corrections):
            values.add_(correction.clamp_min(0.0))
        limited_outgoing = []
        for values, correction, high, average in zip(boundary, corrections, outgoing, near):
            theta = torch.where(correction < 0.0, (values / -correction).clamp_(0.0, 1.0), 1.0)
            values.add_(theta * correction.clamp_max(0.0))
            limited_outgoing.append(average + theta * (high - average))

        speed = self.negative_vx, self.positive_vx, self.negative_vy, self.positive_vy
        denominator = self.left_incoming_flux_unit, self.right_incoming_flux_unit, self.bottom_incoming_flux_unit, self.top_incoming_flux_unit
        unit = self.stationary_wall_unit, self.stationary_wall_unit, self.stationary_wall_unit, self.top_wall_unit
        density = tuple(-self.dv3 * torch.sum(v[None] * f, dim=(-3, -2, -1)) / d for v, f, d in zip(speed, limited_outgoing, denominator))
        wall = tuple(rho[..., None, None, None] * maxwellian[None] for rho, maxwellian in zip(density, unit))
        base[0] += (dt / self.dx) * self.positive_vx[None] * wall[0]
        base[-1] -= (dt / self.dx) * self.negative_vx[None] * wall[1]
        base[:, 0] += (dt / self.dy) * self.positive_vy[None] * wall[2]
        base[:, -1] -= (dt / self.dy) * self.negative_vy[None] * wall[3]
        incoming = tuple(self._positive_trace(a, 1.25 * a + 0.25 * b - 0.5 * g) for a, b, g in zip(near, next_, wall))

        slope = torch.zeros_like(state)
        slope[1:-1] = self._minmod(state[1:-1] - state[:-2], state[2:] - state[1:-1])
        correction = 0.5 * (self.positive_vx[None, None] * slope[:-1] - self.negative_vx[None, None] * slope[1:])
        correction[0] += self.positive_vx[None] * (incoming[0] - state[0])
        correction[-1] += self.negative_vx[None] * (incoming[1] - state[-1])
        correction.mul_(dt / self.dx)
        slope.zero_()
        slope[:-1] += correction.clamp_min(0.0)
        slope[1:] += (-correction).clamp_min(0.0)
        slope.clamp_min_(torch.finfo(state.dtype).tiny).reciprocal_().mul_(base).clamp_(0.0, 1.0)
        correction.mul_(torch.where(correction > 0.0, slope[:-1], slope[1:]))
        base[:-1] -= correction
        base[1:] += correction

        slope.zero_()
        slope[:, 1:-1] = self._minmod(state[:, 1:-1] - state[:, :-2], state[:, 2:] - state[:, 1:-1])
        correction = 0.5 * (self.positive_vy[None, None] * slope[:, :-1] - self.negative_vy[None, None] * slope[:, 1:])
        correction[:, 0] += self.positive_vy[None] * (incoming[2] - state[:, 0])
        correction[:, -1] += self.negative_vy[None] * (incoming[3] - state[:, -1])
        correction.mul_(dt / self.dy)
        slope.zero_()
        slope[:, :-1] += correction.clamp_min(0.0)
        slope[:, 1:] += (-correction).clamp_min(0.0)
        slope.clamp_min_(torch.finfo(state.dtype).tiny).reciprocal_().mul_(base).clamp_(0.0, 1.0)
        correction.mul_(torch.where(correction > 0.0, slope[:, :-1], slope[:, 1:]))
        base[:, :-1] -= correction
        base[:, 1:] += correction
        minimum = float(base.min())
        if minimum < -1.0e-13:
            raise FloatingPointError(f"positivity flux correction failed: min(f)={minimum:.6e}")
        return base.clamp_min_(0.0)

    def advance(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        first = self._forward_euler(state, dt)
        if self.time_integrator == "euler":
            return first
        return 0.5 * state + 0.5 * self._forward_euler(first, dt)

    def run(
        self,
        *,
        t_end: float = 1.0,
        dt: float | None = None,
        progress: bool = True,
        checkpoint_path: Path | None = None,
        checkpoint_interval_seconds: float = 1800.0,
        checkpoint_time_interval: float | None = None,
        resume: bool = False,
        checkpoint_signature: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        nominal_dt = self.cfl_time_step if dt is None else min(dt, self.cfl_time_step)
        self.last_nominal_dt = nominal_dt
        state, self.last_number_of_steps = run_fixed_time(
            initial_state=self.initial_state,
            advance=self.advance,
            t_end=t_end,
            nominal_dt=nominal_dt,
            device=self.device,
            case="lid_driven_cavity_2d",
            progress=progress,
            checkpoint_path=checkpoint_path,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
            resume_path=checkpoint_path if resume else None,
            signature=checkpoint_signature or {},
            checkpoint_time_interval=checkpoint_time_interval,
        )
        return state


def build_parser(*, output_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2D kinetic lid-driven cavity")
    parser.add_argument("--Nx", "--nx", dest="nx", type=int, default=32)
    parser.add_argument("--Ny", "--ny", dest="ny", type=int, default=32)
    parser.add_argument("--Nv", "--nv", dest="nv", type=int, default=32)
    parser.add_argument("--T", "--t-end", "--final-time", dest="t_end", type=float, default=1.0)
    parser.add_argument("--Kn", "--kn", dest="kn", type=float, default=0.1)
    parser.add_argument("--velocity-half-width", type=float, default=10.0)
    parser.add_argument("--wall-temperature", type=float, default=1.0)
    parser.add_argument("--lid-speed", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--time-integrator", choices=("euler", "ssprk2"), default="euler")
    parser.add_argument("--collision-backend", choices=("gate", "spectral"), default="gate")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0)
    parser.add_argument("--checkpoint-time-interval", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gate-batch-chunk", type=int, default=32)
    parser.add_argument("--spectral-R", type=float, default=9.0)
    parser.add_argument("--spectral-angular-accuracy", type=int, default=13)
    parser.add_argument("--spectral-angular-chunk", type=int, default=8)
    parser.add_argument("--spectral-batch-chunk", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser
