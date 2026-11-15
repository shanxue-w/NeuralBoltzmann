"""High-particle-count 1D-x/3D-v DSMC reference for the SOD shock case.

The collision kernel matches the Maxwell-molecule normalization used by the
spectral reference: B = 1/(4*pi), hence the total angular collision rate is
one. Particle transport is exact between collision steps, the two boundaries
inject the prescribed half-range Maxwellian fluxes, and cell-local collisions
use Bird's no-time-counter pair count with isotropic post-collision directions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

try:
    from numba import njit, prange, set_num_threads
except ImportError as error:  # pragma: no cover - exercised only without Numba
    raise RuntimeError("run_dsmc.py requires numba in the active environment") from error


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UnsteadyBoltzmann.runtime import filename_float, guard_output_directory, json_ready

X_LEFT = -35.0
X_RIGHT = 35.0
CHECKPOINT_NAME = "checkpoint_latest.npz"


@njit(cache=False)
def _compact_inside(
    positions: np.ndarray,
    velocities: np.ndarray,
    particle_count: int,
    x_left: float,
    x_right: float,
) -> int:
    write = 0
    for read in range(particle_count):
        x_value = positions[read]
        if x_left <= x_value < x_right:
            if write != read:
                positions[write] = x_value
                velocities[write, 0] = velocities[read, 0]
                velocities[write, 1] = velocities[read, 1]
                velocities[write, 2] = velocities[read, 2]
            write += 1
    return write


@njit(cache=False)
def _cell_counting_order(
    positions: np.ndarray,
    particle_count: int,
    x_left: float,
    inverse_dx: float,
    nx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(nx, dtype=np.int64)
    for particle in range(particle_count):
        cell = int((positions[particle] - x_left) * inverse_dx)
        if cell < 0:
            cell = 0
        elif cell >= nx:
            cell = nx - 1
        counts[cell] += 1
    offsets = np.empty(nx + 1, dtype=np.int64)
    offsets[0] = 0
    for cell in range(nx):
        offsets[cell + 1] = offsets[cell] + counts[cell]
    cursor = offsets[:-1].copy()
    order = np.empty(particle_count, dtype=np.int64)
    for particle in range(particle_count):
        cell = int((positions[particle] - x_left) * inverse_dx)
        if cell < 0:
            cell = 0
        elif cell >= nx:
            cell = nx - 1
        order[cursor[cell]] = particle
        cursor[cell] += 1
    return order, offsets, counts


@njit(cache=False, parallel=True)
def _apply_isotropic_collisions(
    velocities: np.ndarray,
    pair_first: np.ndarray,
    pair_second: np.ndarray,
    directions: np.ndarray,
    collision_offsets: np.ndarray,
) -> None:
    for cell in prange(collision_offsets.size - 1):
        for event in range(collision_offsets[cell], collision_offsets[cell + 1]):
            first = pair_first[event]
            second = pair_second[event]
            v1x = velocities[first, 0]
            v1y = velocities[first, 1]
            v1z = velocities[first, 2]
            v2x = velocities[second, 0]
            v2y = velocities[second, 1]
            v2z = velocities[second, 2]
            center_x = 0.5 * (v1x + v2x)
            center_y = 0.5 * (v1y + v2y)
            center_z = 0.5 * (v1z + v2z)
            relative_x = v1x - v2x
            relative_y = v1y - v2y
            relative_z = v1z - v2z
            relative_speed = math.sqrt(relative_x * relative_x + relative_y * relative_y + relative_z * relative_z)
            half_speed = 0.5 * relative_speed
            new_x = half_speed * directions[event, 0]
            new_y = half_speed * directions[event, 1]
            new_z = half_speed * directions[event, 2]
            velocities[first, 0] = center_x + new_x
            velocities[first, 1] = center_y + new_y
            velocities[first, 2] = center_z + new_z
            velocities[second, 0] = center_x - new_x
            velocities[second, 1] = center_y - new_y
            velocities[second, 2] = center_z - new_z


def parameterized_output_directory(args: argparse.Namespace) -> Path:
    dx = (X_RIGHT - X_LEFT) / float(args.nx)
    rho_ratio = (args.gamma + 1.0) * args.mach**2 / ((args.gamma - 1.0) * args.mach**2 + 2.0)
    rho_plus = args.rho_minus * rho_ratio
    transport_dt = args.transport_cfl * dx / args.velocity_half_width
    collision_dt = args.collision_cfl * args.kn / rho_plus
    automatic_dt = min(transport_dt, collision_dt)
    effective_dt = automatic_dt if args.dt is None else min(args.dt, automatic_dt)
    parts = [
        f"nx{args.nx}",
        f"kn{filename_float(args.kn)}",
        f"tend{filename_float(args.t_end)}",
        f"dt{filename_float(effective_dt)}",
        f"L{filename_float(args.velocity_half_width)}",
        f"Ma{filename_float(args.mach)}",
        f"gam{filename_float(args.gamma)}",
        f"rhom{filename_float(args.rho_minus)}",
        f"Tm{filename_float(args.temperature_minus)}",
        "Tstd",
        "dsmc",
        f"ppc{args.particles_per_upstream_cell}",
        f"seed{args.seed}",
        f"avg{filename_float(args.sample_start)}",
        f"ds{filename_float(args.sample_interval)}",
        f"vb{args.velocity_bins}",
    ]
    if args.full_distribution_nv > 0:
        parts.append(f"fNv{args.full_distribution_nv}")
    return args.output_dir / "_".join(parts)


class SodShockDSMC:
    """One-off particle solver for the prescribed stationary shock states."""

    def __init__(
        self,
        *,
        nx: int = 100,
        kn: float = 1.0,
        gamma: float = 5.0 / 3.0,
        mach: float = 2.0,
        rho_minus: float = 1.0,
        temperature_minus: float = 1.0,
        particles_per_upstream_cell: int = 20_000,
        velocity_half_width: float = 10.0,
        velocity_bins: int = 128,
        full_distribution_nv: int = 0,
        transport_cfl: float = 0.5,
        collision_cfl: float = 0.25,
        sample_start: float = 50.0,
        sample_interval: float = 1.0,
        seed: int = 1234,
        threads: int = 0,
    ) -> None:
        if threads > 0:
            set_num_threads(threads)

        self.nx = int(nx)
        self.kn = float(kn)
        self.gamma = float(gamma)
        self.mach = float(mach)
        self.rho_minus = float(rho_minus)
        self.temperature_minus = float(temperature_minus)
        self.particles_per_upstream_cell = int(particles_per_upstream_cell)
        self.velocity_half_width = float(velocity_half_width)
        self.velocity_bins = int(velocity_bins)
        self.full_distribution_nv = int(full_distribution_nv)
        self.transport_cfl = float(transport_cfl)
        self.collision_cfl = float(collision_cfl)
        self.sample_start = float(sample_start)
        self.sample_interval = float(sample_interval)
        self.seed = int(seed)
        self.threads = int(threads)
        self.x_left = X_LEFT
        self.x_right = X_RIGHT
        self.dx = (self.x_right - self.x_left) / self.nx
        self.inverse_dx = 1.0 / self.dx
        self.x_centers = self.x_left + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dx

        density_ratio = (self.gamma + 1.0) * self.mach**2 / ((self.gamma - 1.0) * self.mach**2 + 2.0)
        velocity_ratio = ((self.gamma - 1.0) * self.mach**2 + 2.0) / ((self.gamma + 1.0) * self.mach**2)
        temperature_ratio = (
            ((self.gamma - 1.0) * self.mach**2 + 2.0)
            * (2.0 * self.gamma * self.mach**2 - self.gamma + 1.0)
            / ((self.gamma + 1.0) ** 2 * self.mach**2)
        )
        self.rho_plus = self.rho_minus * density_ratio
        self.u_minus = self.mach * math.sqrt(self.gamma * self.temperature_minus)
        self.u_plus = self.u_minus * velocity_ratio
        self.temperature_plus = self.temperature_minus * temperature_ratio
        self.particle_weight = self.rho_minus * self.dx / self.particles_per_upstream_cell
        self.transport_time_step = self.transport_cfl * self.dx / self.velocity_half_width
        self.collision_time_step = self.collision_cfl * self.kn / self.rho_plus
        self.automatic_time_step = min(self.transport_time_step, self.collision_time_step)

        self.rng = np.random.default_rng(self.seed)
        self.positions = np.empty(0, dtype=np.float64)
        self.velocities = np.empty((0, 3), dtype=np.float64)
        self.particle_count = 0
        self.capacity = 0
        self.sample_times: list[float] = []
        self.sample_moments: list[np.ndarray] = []
        self.vx_marginal_sum = np.zeros((self.nx, self.velocity_bins), dtype=np.float64)
        if self.full_distribution_nv > 0:
            shape = (self.nx, self.full_distribution_nv, self.full_distribution_nv, self.full_distribution_nv)
            self.full_distribution_sum: np.ndarray | None = np.zeros(shape, dtype=np.float64)
        else:
            self.full_distribution_sum = None
        self.velocity_tail_particle_fraction_sum = 0.0
        self.total_collision_events = 0
        self.total_injected_left = 0
        self.total_injected_right = 0
        self.last_nominal_dt = math.nan
        self.last_completed_steps = 0

        self.left_flux_grid, self.left_flux_cdf, self.left_number_flux = self._incoming_flux_table(
            density=self.rho_minus, velocity_x=self.u_minus,
            temperature=self.temperature_minus, incoming_from_left=True)
        self.right_flux_grid, self.right_flux_cdf, self.right_number_flux = self._incoming_flux_table(
            density=self.rho_plus, velocity_x=self.u_plus,
            temperature=self.temperature_plus, incoming_from_left=False)

    @property
    def sample_count(self) -> int:
        return len(self.sample_times)

    def _incoming_flux_table(
        self,
        *,
        density: float,
        velocity_x: float,
        temperature: float,
        incoming_from_left: bool,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        thermal_width = math.sqrt(temperature)
        if incoming_from_left:
            lower = 0.0
            upper = max(velocity_x + 9.0 * thermal_width, 9.0 * thermal_width)
        else:
            lower = min(velocity_x - 9.0 * thermal_width, -9.0 * thermal_width)
            upper = 0.0
        grid = np.linspace(lower, upper, 65_537, dtype=np.float64)
        one_dimensional = density * np.exp(-(grid - velocity_x) ** 2 / (2.0 * temperature)) / math.sqrt(2.0 * math.pi * temperature)
        normal_speed = grid if incoming_from_left else -grid
        density_flux = normal_speed * one_dimensional
        increments = 0.5 * (density_flux[1:] + density_flux[:-1]) * np.diff(grid)
        cumulative = np.empty_like(grid)
        cumulative[0] = 0.0
        cumulative[1:] = np.cumsum(increments)
        number_flux = float(cumulative[-1])
        cumulative /= number_flux
        cumulative[-1] = 1.0
        return grid, cumulative, number_flux

    def _ensure_capacity(self, required: int) -> None:
        if required <= self.capacity:
            return
        new_capacity = max(required, int(1.25 * max(1, self.capacity)) + 1024)
        positions = np.empty(new_capacity, dtype=np.float64)
        velocities = np.empty((new_capacity, 3), dtype=np.float64)
        if self.particle_count:
            positions[: self.particle_count] = self.positions[: self.particle_count]
            velocities[: self.particle_count] = self.velocities[: self.particle_count]
        self.positions = positions
        self.velocities = velocities
        self.capacity = new_capacity

    def _append_particles(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        count = int(positions.size)
        if count == 0:
            return
        stop = self.particle_count + count
        self._ensure_capacity(stop)
        self.positions[self.particle_count : stop] = positions
        self.velocities[self.particle_count : stop] = velocities
        self.particle_count = stop

    def initialize_particles(self) -> None:
        cell_density = np.where(self.x_centers < 0.0, self.rho_minus, self.rho_plus)
        expected = cell_density * self.dx / self.particle_weight
        cumulative = np.rint(np.cumsum(expected)).astype(np.int64)
        counts = np.diff(np.concatenate((np.zeros(1, dtype=np.int64), cumulative)))
        total = int(counts.sum())
        self.positions = np.empty(total, dtype=np.float64)
        self.velocities = np.empty((total, 3), dtype=np.float64)
        self.capacity = total
        self.particle_count = total
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
        for cell in range(self.nx):
            begin = int(offsets[cell])
            stop = int(offsets[cell + 1])
            count = stop - begin
            self.positions[begin:stop] = self.x_left + cell * self.dx + self.dx * self.rng.random(count)
            if self.x_centers[cell] < 0.0:
                mean_x = self.u_minus
                temperature = self.temperature_minus
            else:
                mean_x = self.u_plus
                temperature = self.temperature_plus
            deviation = math.sqrt(temperature)
            self.velocities[begin:stop] = self.rng.normal(loc=(mean_x, 0.0, 0.0), scale=deviation, size=(count, 3))

    def _sample_boundary_velocities(
        self,
        count: int,
        *,
        incoming_from_left: bool,
    ) -> np.ndarray:
        if count == 0:
            return np.empty((0, 3), dtype=np.float64)
        if incoming_from_left:
            grid = self.left_flux_grid
            cdf = self.left_flux_cdf
            temperature = self.temperature_minus
        else:
            grid = self.right_flux_grid
            cdf = self.right_flux_cdf
            temperature = self.temperature_plus
        result = np.empty((count, 3), dtype=np.float64)
        result[:, 0] = np.interp(self.rng.random(count), cdf, grid)
        result[:, 1:] = self.rng.normal(0.0, math.sqrt(temperature), size=(count, 2))
        return result

    def _inject_boundaries(self, dt: float) -> None:
        left_count = int(self.rng.poisson(self.left_number_flux * dt / self.particle_weight))
        right_count = int(self.rng.poisson(self.right_number_flux * dt / self.particle_weight))
        left_velocity = self._sample_boundary_velocities(left_count, incoming_from_left=True)
        right_velocity = self._sample_boundary_velocities(right_count, incoming_from_left=False)
        left_position = self.x_left + left_velocity[:, 0] * dt * self.rng.random(left_count)
        right_position = self.x_right + right_velocity[:, 0] * dt * self.rng.random(right_count)
        self._append_particles(left_position, left_velocity)
        self._append_particles(right_position, right_velocity)
        self.total_injected_left += left_count
        self.total_injected_right += right_count

    def _collision_schedule(
        self,
        order: np.ndarray,
        offsets: np.ndarray,
        counts: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        particle_pairs = 0.5 * counts.astype(np.float64) * np.maximum(counts - 1, 0)
        expected = particle_pairs * self.particle_weight * dt / (self.kn * self.dx)
        collision_counts = np.floor(expected + self.rng.random(self.nx)).astype(np.int64)
        collision_counts = np.where(counts >= 2, collision_counts, 0)
        collision_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(collision_counts, dtype=np.int64)))
        total = int(collision_offsets[-1])
        first = np.empty(total, dtype=np.int64)
        second = np.empty(total, dtype=np.int64)
        directions = np.empty((total, 3), dtype=np.float64)
        for cell in range(self.nx):
            event_begin = int(collision_offsets[cell])
            event_stop = int(collision_offsets[cell + 1])
            events = event_stop - event_begin
            cell_count = int(counts[cell])
            if events == 0:
                continue
            local_first = self.rng.integers(0, cell_count, size=events)
            local_second = self.rng.integers(0, cell_count - 1, size=events)
            local_second += local_second >= local_first
            cell_begin = int(offsets[cell])
            first[event_begin:event_stop] = order[cell_begin + local_first]
            second[event_begin:event_stop] = order[cell_begin + local_second]
        if total:
            cosine = 2.0 * self.rng.random(total) - 1.0
            azimuth = 2.0 * math.pi * self.rng.random(total)
            radial = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
            directions[:, 0] = radial * np.cos(azimuth)
            directions[:, 1] = radial * np.sin(azimuth)
            directions[:, 2] = cosine
        return first, second, directions, collision_offsets

    def advance(self, dt: float) -> tuple[np.ndarray, int]:
        self.positions[: self.particle_count] += dt * self.velocities[: self.particle_count, 0]
        self.particle_count = _compact_inside(
            self.positions, self.velocities, self.particle_count, self.x_left, self.x_right)
        self._inject_boundaries(dt)
        order, offsets, counts = _cell_counting_order(
            self.positions, self.particle_count, self.x_left, self.inverse_dx, self.nx)
        first, second, directions, collision_offsets = self._collision_schedule(order, offsets, counts, dt)
        _apply_isotropic_collisions(self.velocities, first, second, directions, collision_offsets)
        collision_events = int(first.size)
        self.total_collision_events += collision_events
        return counts, collision_events

    def instantaneous_moments(self) -> np.ndarray:
        positions = self.positions[: self.particle_count]
        velocities = self.velocities[: self.particle_count]
        cell = np.floor((positions - self.x_left) * self.inverse_dx).astype(np.int64)
        np.clip(cell, 0, self.nx - 1, out=cell)
        counts = np.bincount(cell, minlength=self.nx).astype(np.float64)
        density = self.particle_weight * counts / self.dx
        momentum = np.stack(tuple(
            np.bincount(cell, weights=velocities[:, axis], minlength=self.nx)
            for axis in range(3)), axis=1)
        energy_sum = np.bincount(cell, weights=np.einsum("ij,ij->i", velocities, velocities), minlength=self.nx)
        safe_counts = np.maximum(counts, 1.0)
        velocity = momentum / safe_counts[:, None]
        temperature = (energy_sum / safe_counts - np.einsum("ij,ij->i", velocity, velocity)) / 3.0
        if np.any(counts == 0.0):
            velocity[counts == 0.0] = np.nan
            temperature[counts == 0.0] = np.nan
        return np.column_stack((density, velocity, temperature))

    def collect_sample(self, time_value: float) -> None:
        moments = self.instantaneous_moments()
        if not np.isfinite(moments).all():
            raise FloatingPointError("DSMC sample contains an empty or nonfinite cell")
        self.sample_times.append(float(time_value))
        self.sample_moments.append(moments)

        positions = self.positions[: self.particle_count]
        velocities = self.velocities[: self.particle_count]
        cell = np.floor((positions - self.x_left) * self.inverse_dx).astype(np.int64)
        np.clip(cell, 0, self.nx - 1, out=cell)
        dv = 2.0 * self.velocity_half_width / self.velocity_bins
        vx_bin = np.floor((velocities[:, 0] + self.velocity_half_width) / dv).astype(np.int64)
        inside_vx = (vx_bin >= 0) & (vx_bin < self.velocity_bins)
        flat_vx = cell[inside_vx] * self.velocity_bins + vx_bin[inside_vx]
        marginal = np.bincount(flat_vx, minlength=self.nx * self.velocity_bins).reshape(self.nx, self.velocity_bins)
        self.vx_marginal_sum += self.particle_weight / (self.dx * dv) * marginal
        self.velocity_tail_particle_fraction_sum += 1.0 - (
            float(np.count_nonzero(inside_vx)) / self.particle_count
        )

        if self.full_distribution_sum is not None:
            nv = self.full_distribution_nv
            full_dv = 2.0 * self.velocity_half_width / nv
            velocity_bin = np.floor((velocities + self.velocity_half_width) / full_dv).astype(np.int64)
            inside = np.all((velocity_bin >= 0) & (velocity_bin < nv), axis=1)
            selected = velocity_bin[inside]
            flat = (
                ((cell[inside] * nv + selected[:, 0]) * nv + selected[:, 1])
                * nv
                + selected[:, 2]
            )
            histogram = np.bincount(flat, minlength=self.nx * nv**3).reshape(self.nx, nv, nv, nv)
            self.full_distribution_sum += self.particle_weight / (self.dx * full_dv**3) * histogram

    def save_checkpoint(
        self,
        path: Path,
        *,
        time_value: float,
        completed_steps: int,
        nominal_dt: float,
        signature: dict[str, Any],
    ) -> None:
        history = np.stack(self.sample_moments) if self.sample_moments else np.empty((0, self.nx, 5), dtype=np.float64)
        full_sum = self.full_distribution_sum if self.full_distribution_sum is not None else np.empty(0, dtype=np.float64)
        payload = {
            "format_version": np.array(1, dtype=np.int64),
            "case": np.array("sod_shock_1d_dsmc"),
            "time": np.array(time_value, dtype=np.float64),
            "completed_steps": np.array(completed_steps, dtype=np.int64),
            "nominal_dt": np.array(nominal_dt, dtype=np.float64),
            "signature_json": np.array(json.dumps(json_ready(signature), sort_keys=True)),
            "rng_state_json": np.array(json.dumps(json_ready(self.rng.bit_generator.state))),
            "positions": self.positions[: self.particle_count],
            "velocities": self.velocities[: self.particle_count],
            "sample_times": np.asarray(self.sample_times, dtype=np.float64),
            "sample_moments": history,
            "vx_marginal_sum": self.vx_marginal_sum,
            "full_distribution_sum": full_sum,
            "velocity_tail_particle_fraction_sum": np.array(
                self.velocity_tail_particle_fraction_sum, dtype=np.float64
            ),
            "total_collision_events": np.array(
                self.total_collision_events, dtype=np.int64
            ),
            "total_injected_left": np.array(self.total_injected_left, dtype=np.int64),
            "total_injected_right": np.array(self.total_injected_right, dtype=np.int64),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **payload)
        os.replace(temporary, path)

    def load_checkpoint(
        self,
        path: Path,
        *,
        t_end: float,
        nominal_dt: float,
        signature: dict[str, Any],
    ) -> tuple[float, int]:
        with np.load(path, allow_pickle=False) as payload:
            if json.loads(str(payload["signature_json"])) != json_ready(signature):
                raise ValueError("resume checkpoint parameters do not match this run")
            saved_dt = float(payload["nominal_dt"])
            if not math.isclose(saved_dt, nominal_dt, rel_tol=1.0e-14):
                raise ValueError("checkpoint time step does not match this run")
            time_value = float(payload["time"])
            completed_steps = int(payload["completed_steps"])
            self.positions = payload["positions"].astype(np.float64, copy=True)
            self.velocities = payload["velocities"].astype(np.float64, copy=True)
            self.sample_times = payload["sample_times"].astype(np.float64).tolist()
            history = np.asarray(payload["sample_moments"], dtype=np.float64)
            self.sample_moments = [item.copy() for item in history]
            self.vx_marginal_sum = payload["vx_marginal_sum"].astype(np.float64, copy=True)
            if self.full_distribution_sum is not None:
                self.full_distribution_sum = payload["full_distribution_sum"].astype(np.float64, copy=True)
            self.velocity_tail_particle_fraction_sum = float(payload["velocity_tail_particle_fraction_sum"])
            self.total_collision_events = int(payload["total_collision_events"])
            self.total_injected_left = int(payload["total_injected_left"])
            self.total_injected_right = int(payload["total_injected_right"])
            rng_state = json.loads(str(payload["rng_state_json"]))
        self.capacity = self.particle_count = self.positions.size
        self.rng.bit_generator.state = rng_state
        return min(time_value, t_end), completed_steps

    def _sample_due_at_current_time(self, time_value: float) -> bool:
        tolerance = 1.0e-11 * max(1.0, abs(time_value))
        if time_value < self.sample_start - tolerance:
            return False
        index = round((time_value - self.sample_start) / self.sample_interval)
        scheduled = self.sample_start + index * self.sample_interval
        if abs(time_value - scheduled) > tolerance:
            return False
        if not self.sample_times:
            return True
        return time_value > self.sample_times[-1] + tolerance

    def _next_sample_time(self, time_value: float) -> float:
        tolerance = 1.0e-11 * max(1.0, abs(time_value))
        if time_value < self.sample_start - tolerance:
            return self.sample_start
        index = math.floor((time_value - self.sample_start) / self.sample_interval + 1.0e-11) + 1
        return self.sample_start + index * self.sample_interval

    def run(
        self,
        *,
        t_end: float,
        dt: float | None,
        progress: bool,
        checkpoint_path: Path,
        checkpoint_interval_seconds: float,
        signature: dict[str, Any],
        resume: bool = False,
        resume_from_path: Path | None = None,
    ) -> np.ndarray:
        nominal_dt = self.automatic_time_step if dt is None else min(float(dt), self.automatic_time_step)
        self.last_nominal_dt = nominal_dt
        if resume:
            source = checkpoint_path if resume_from_path is None else resume_from_path
            time_value, completed_steps = self.load_checkpoint(source, t_end=t_end, nominal_dt=nominal_dt, signature=signature)
            if progress:
                print(f"resumed {source} at t={time_value:.10g}, particles={self.particle_count:,}", flush=True)
        else:
            self.initialize_particles()
            time_value = 0.0
            completed_steps = 0

        def checkpoint() -> None:
            self.save_checkpoint(checkpoint_path, time_value=time_value, completed_steps=completed_steps,
                                 nominal_dt=nominal_dt, signature=signature)
        if self._sample_due_at_current_time(time_value):
            self.collect_sample(time_value)
        checkpoint()

        estimated_steps = max(1, int(math.ceil(max(0.0, t_end - time_value) / nominal_dt)))
        report_stride = max(1, estimated_steps // 100)
        local_steps = 0
        last_checkpoint_wall_time = time.monotonic()
        next_sample = self._next_sample_time(time_value)
        start_wall_time = time.monotonic()
        try:
            while time_value < t_end - 1.0e-13 * max(1.0, t_end):
                step_dt = min(nominal_dt, t_end - time_value)
                if next_sample > time_value:
                    step_dt = min(step_dt, next_sample - time_value)
                counts, collision_events = self.advance(step_dt)
                time_value += step_dt
                completed_steps += 1
                local_steps += 1
                if self._sample_due_at_current_time(time_value):
                    self.collect_sample(time_value)
                    next_sample = self._next_sample_time(time_value)
                report = local_steps == 1 or local_steps % report_stride == 0 or time_value >= t_end - 1.0e-13 * max(1.0, t_end)
                if progress and report:
                    elapsed = time.monotonic() - start_wall_time
                    density_min = self.particle_weight * float(counts.min()) / self.dx
                    density_max = self.particle_weight * float(counts.max()) / self.dx
                    print(
                        f"step={completed_steps}, t={time_value:.10g}/{t_end:.10g}, "
                        f"particles={self.particle_count:,}, collisions={collision_events:,}, "
                        f"rho_cells=[{density_min:.5g},{density_max:.5g}], "
                        f"samples={self.sample_count}, wall={elapsed:.1f}s",
                        flush=True,
                    )
                checkpoint_due = checkpoint_interval_seconds > 0.0 and time.monotonic() - last_checkpoint_wall_time >= checkpoint_interval_seconds
                if checkpoint_due:
                    checkpoint()
                    last_checkpoint_wall_time = time.monotonic()
                    if progress:
                        print(f"checkpoint saved at t={time_value:.10g}: {checkpoint_path}", flush=True)
        except KeyboardInterrupt:
            checkpoint()
            print(f"interrupted; checkpoint saved at t={time_value:.10g}: {checkpoint_path}", file=sys.stderr, flush=True)
            raise
        checkpoint()
        self.last_completed_steps = completed_steps
        return self.instantaneous_moments()


def latest_compatible_checkpoint(
    root: Path,
    *,
    target: Path,
    t_end: float,
    nominal_dt: float,
    signature: dict[str, Any],
) -> tuple[Path, float] | None:
    expected_signature = json_ready(signature)
    candidates = set(root.glob(f"*/{CHECKPOINT_NAME}"))
    if target.is_file():
        candidates.add(target)
    tolerance = 1.0e-12 * max(1.0, abs(t_end))
    best: tuple[tuple[float, bool, int], Path, float] | None = None
    for path in candidates:
        try:
            with np.load(path, allow_pickle=False) as payload:
                valid = (int(payload["format_version"]) == 1
                         and str(payload["case"]) == "sod_shock_1d_dsmc"
                         and json.loads(str(payload["signature_json"])) == expected_signature)
                if not valid:
                    continue
                saved_dt = float(payload["nominal_dt"])
                saved_time = float(payload["time"])
            if not math.isclose(saved_dt, nominal_dt, rel_tol=1.0e-14, abs_tol=0.0):
                continue
            if (
                not math.isfinite(saved_time)
                or saved_time < 0.0
                or saved_time > t_end + tolerance
            ):
                continue
            key = (min(saved_time, t_end), path == target, path.stat().st_mtime_ns)
        except Exception:
            continue
        if best is None or key > best[0]:
            best = (key, path, min(saved_time, t_end))
    if best is None:
        return None
    return best[1], best[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="1D normal-shock DSMC reference")
    parser.add_argument("--Nx", "--nx", dest="nx", type=int, default=100)
    parser.add_argument("--Kn", "--kn", dest="kn", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--Ma", "--ma", "--mach", dest="mach", type=float, default=2.0)
    parser.add_argument("--rho-minus", type=float, default=1.0)
    parser.add_argument("--temperature-minus", type=float, default=1.0)
    parser.add_argument("--T", "--t-end", "--final-time", dest="t_end", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument(
        "--particles-per-upstream-cell", "--particles-per-cell",
        dest="particles_per_upstream_cell", type=int, default=20_000,
    )
    parser.add_argument("--velocity-half-width", type=float, default=10.0)
    parser.add_argument("--velocity-bins", type=int, default=128)
    parser.add_argument("--full-distribution-nv", type=int, default=0)
    parser.add_argument("--transport-cfl", type=float, default=0.5)
    parser.add_argument("--collision-cfl", type=float, default=0.25)
    parser.add_argument("--sample-start", type=float, default=50.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _checkpoint_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "case": "sod_shock_1d_dsmc",
        "Nx": args.nx,
        "Kn": args.kn,
        "gamma": args.gamma,
        "Mach": args.mach,
        "rho_minus": args.rho_minus,
        "temperature_minus": args.temperature_minus,
        "temperature_convention": "standard",
        "particles_per_upstream_cell": args.particles_per_upstream_cell,
        "velocity_half_width": args.velocity_half_width,
        "velocity_bins": args.velocity_bins,
        "full_distribution_nv": args.full_distribution_nv,
        "transport_cfl": args.transport_cfl,
        "collision_cfl": args.collision_cfl,
        "sample_start": args.sample_start,
        "sample_interval": args.sample_interval,
        "seed": args.seed,
        "collision_kernel": "Maxwell_B_1_over_4pi",
        "boundary": "half_range_Maxwellian_inflow",
    }


from UnsteadyBoltzmann.sod_shock_1d.postprocess_dsmc import save_results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_start > args.t_end and not args.quiet:
        print("warning: sample-start is after T; PDFs will use the instantaneous profile", file=sys.stderr, flush=True)
    output_dir = parameterized_output_directory(args)
    guard_output_directory(output_dir, overwrite=args.overwrite, resume=args.resume)

    solver = SodShockDSMC(
        nx=args.nx,
        kn=args.kn,
        gamma=args.gamma,
        mach=args.mach,
        rho_minus=args.rho_minus,
        temperature_minus=args.temperature_minus,
        particles_per_upstream_cell=args.particles_per_upstream_cell,
        velocity_half_width=args.velocity_half_width,
        velocity_bins=args.velocity_bins,
        full_distribution_nv=args.full_distribution_nv,
        transport_cfl=args.transport_cfl,
        collision_cfl=args.collision_cfl,
        sample_start=args.sample_start,
        sample_interval=args.sample_interval,
        seed=args.seed,
        threads=args.threads,
    )
    nominal_dt = solver.automatic_time_step if args.dt is None else min(float(args.dt), solver.automatic_time_step)
    signature = _checkpoint_signature(args)
    checkpoint_path = output_dir / CHECKPOINT_NAME
    resume_source: Path | None = None
    resume_source_time: float | None = None
    if args.resume:
        selected = latest_compatible_checkpoint(
            args.output_dir, target=checkpoint_path, t_end=args.t_end,
            nominal_dt=nominal_dt, signature=signature)
        if selected is None:
            raise FileNotFoundError(
                "no compatible DSMC checkpoint exists at or below "
                f"T={args.t_end:.10g} under {args.output_dir}"
            )
        resume_source, resume_source_time = selected

    mean_density_ratio = 0.5 * (1.0 + solver.rho_plus / solver.rho_minus)
    expected_initial_particles = round(args.nx * args.particles_per_upstream_cell * mean_density_ratio)
    if not args.quiet:
        print("DSMC kernel=Maxwell B=1/(4pi), Bird NTC, isotropic scattering", flush=True)
        states = (f"minus=({solver.rho_minus:.8g},{solver.u_minus:.8g},{solver.temperature_minus:.8g}), "
                  f"plus=({solver.rho_plus:.8g},{solver.u_plus:.8g},{solver.temperature_plus:.8g})")
        print(f"states: {states}", flush=True)
        print(f"particle weight={solver.particle_weight:.8g}, initial particles~{expected_initial_particles:,}, "
              f"upstream particles/cell={solver.particles_per_upstream_cell:,}", flush=True)
        print(
            f"dt={nominal_dt:.10g} "
            f"(transport cap={solver.transport_time_step:.10g}, "
            f"collision cap={solver.collision_time_step:.10g}), "
            f"sample t>={solver.sample_start:g} every {solver.sample_interval:g}",
            flush=True,
        )
        print(f"result directory: {output_dir}", flush=True)
        print(f"rolling checkpoint: {checkpoint_path} (every {args.checkpoint_minutes:g} wall-clock minutes)", flush=True)
        if resume_source is not None:
            print(f"resume source: {resume_source} (saved t={resume_source_time:.10g})", flush=True)

    instantaneous = solver.run(
        t_end=args.t_end,
        dt=args.dt,
        progress=not args.quiet,
        checkpoint_path=checkpoint_path,
        checkpoint_interval_seconds=60.0 * args.checkpoint_minutes,
        signature=signature,
        resume=args.resume,
        resume_from_path=resume_source,
    )
    tail_fraction = solver.velocity_tail_particle_fraction_sum / solver.sample_count if solver.sample_count else math.nan
    run_configuration = {
        **signature,
        "T_end": args.t_end,
        "requested_dt": args.dt,
        "nominal_dt": nominal_dt,
        "completed_steps": solver.last_completed_steps,
        "x_domain": [solver.x_left, solver.x_right],
        "particle_weight": solver.particle_weight,
        "final_particle_count": solver.particle_count,
        "sample_count": solver.sample_count,
        "sample_times": solver.sample_times,
        "velocity_tail_particle_fraction_mean": tail_fraction,
        "total_collision_events": solver.total_collision_events,
        "total_injected_left": solver.total_injected_left,
        "total_injected_right": solver.total_injected_right,
        "checkpoint": checkpoint_path,
        "checkpoint_minutes": args.checkpoint_minutes,
        "resumed": args.resume,
        "resume_source": resume_source,
        "resume_source_time": resume_source_time,
        "threads": args.threads,
    }
    save_results(output_dir, solver=solver, instantaneous_moments=instantaneous, run_configuration=run_configuration)
    print(f"saved DSMC profiles, sampling history, and PDFs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
