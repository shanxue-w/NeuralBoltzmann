"""DSMC transport, collisions, and checkpoint helpers shared by both cavity meshes."""

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
except ImportError as error:  # pragma: no cover
    raise RuntimeError("run_dsmc.py requires numba") from error

from UnsteadyBoltzmann.runtime import json_ready


CHECKPOINT_NAME = "checkpoint_latest.npz"
MAX_WALL_HITS = 4


@njit(cache=False)
def _cell_counting_order(
    positions: np.ndarray, nx: int, ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = nx * ny
    counts = np.zeros(cells, dtype=np.int64)
    for particle in range(positions.shape[0]):
        ix = min(nx - 1, max(0, int(positions[particle, 0] * nx)))
        iy = min(ny - 1, max(0, int(positions[particle, 1] * ny)))
        counts[ix * ny + iy] += 1
    offsets = np.empty(cells + 1, dtype=np.int64)
    offsets[0] = 0
    for cell in range(cells):
        offsets[cell + 1] = offsets[cell] + counts[cell]
    cursor = offsets[:-1].copy()
    order = np.empty(positions.shape[0], dtype=np.int64)
    for particle in range(positions.shape[0]):
        ix = min(nx - 1, max(0, int(positions[particle, 0] * nx)))
        iy = min(ny - 1, max(0, int(positions[particle, 1] * ny)))
        cell = ix * ny + iy
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
            v1x, v1y, v1z = velocities[first]
            v2x, v2y, v2z = velocities[second]
            center_x = 0.5 * (v1x + v2x)
            center_y = 0.5 * (v1y + v2y)
            center_z = 0.5 * (v1z + v2z)
            relative_x = v1x - v2x
            relative_y = v1y - v2y
            relative_z = v1z - v2z
            half_speed = 0.5 * math.sqrt(
                relative_x * relative_x + relative_y * relative_y + relative_z * relative_z
            )
            new_x = half_speed * directions[event, 0]
            new_y = half_speed * directions[event, 1]
            new_z = half_speed * directions[event, 2]
            velocities[first, 0] = center_x + new_x
            velocities[first, 1] = center_y + new_y
            velocities[first, 2] = center_z + new_z
            velocities[second, 0] = center_x - new_x
            velocities[second, 1] = center_y - new_y
            velocities[second, 2] = center_z - new_z


@njit(cache=False, parallel=True)
def _reflect_crossing_particles(
    positions: np.ndarray,
    velocities: np.ndarray,
    crossing: np.ndarray,
    dt: float,
    random_values: np.ndarray,
    wall_temperature: float,
    lid_speed: float,
) -> None:
    thermal_deviation = math.sqrt(wall_temperature)
    for row in prange(crossing.size):
        particle = crossing[row]
        vx, vy, vz = velocities[particle]
        x = positions[particle, 0] - dt * vx
        y = positions[particle, 1] - dt * vy
        remaining = dt
        for impact in range(random_values.shape[1]):
            tx = 1.0e300
            ty = 1.0e300
            if vx < 0.0:
                tx = -x / vx
            elif vx > 0.0:
                tx = (1.0 - x) / vx
            if vy < 0.0:
                ty = -y / vy
            elif vy > 0.0:
                ty = (1.0 - y) / vy
            hit_time = min(tx, ty)
            if hit_time > remaining:
                x += remaining * vx
                y += remaining * vy
                remaining = 0.0
                break
            x += hit_time * vx
            y += hit_time * vy
            remaining -= hit_time

            r0 = min(random_values[row, impact, 0], 1.0 - 1.0e-15)
            r1 = max(random_values[row, impact, 1], 1.0e-300)
            angle = 2.0 * math.pi * random_values[row, impact, 2]
            normal_speed = math.sqrt(-2.0 * wall_temperature * math.log(1.0 - r0))
            radius = math.sqrt(-2.0 * math.log(r1))
            tangent0 = thermal_deviation * radius * math.cos(angle)
            tangent1 = thermal_deviation * radius * math.sin(angle)
            if tx <= ty:
                vx = normal_speed if vx < 0.0 else -normal_speed
                vy = tangent0
                vz = tangent1
            else:
                bottom = vy < 0.0
                vx = tangent0 if bottom else lid_speed + tangent0
                vy = normal_speed if bottom else -normal_speed
                vz = tangent1
            if remaining <= 1.0e-15:
                remaining = 0.0
                break
        if remaining > 0.0:
            x += remaining * vx
            y += remaining * vy
        positions[particle, 0] = min(1.0, max(0.0, x))
        positions[particle, 1] = min(1.0, max(0.0, y))
        velocities[particle, 0] = vx
        velocities[particle, 1] = vy
        velocities[particle, 2] = vz


def automatic_time_step(args: argparse.Namespace) -> float:
    cell_width = min(1.0 / args.nx, 1.0 / args.ny)
    reference_speed = abs(args.lid_speed) + 4.0 * math.sqrt(max(1.0, args.wall_temperature))
    transport_dt = args.transport_cfl * cell_width / reference_speed
    collision_dt = args.collision_cfl * args.kn
    return min(transport_dt, collision_dt)


class LidDrivenCavityDSMC:
    """Closed-cavity DSMC with diffuse moving-wall reflection."""

    def __init__(
        self,
        *,
        nx: int = 32,
        ny: int = 32,
        kn: float = 0.1,
        wall_temperature: float = 1.0,
        lid_speed: float = 0.15,
        particles_per_cell: int = 2_000,
        transport_cfl: float = 0.5,
        collision_cfl: float = 0.25,
        sample_start: float = 200.0,
        sample_interval: float = 1.0,
        seed: int = 1234,
        threads: int = 0,
    ) -> None:
        if threads > 0:
            set_num_threads(threads)
        self.nx = int(nx)
        self.ny = int(ny)
        self.kn = float(kn)
        self.wall_temperature = float(wall_temperature)
        self.lid_speed = float(lid_speed)
        self.particles_per_cell = int(particles_per_cell)
        self.transport_cfl = float(transport_cfl)
        self.collision_cfl = float(collision_cfl)
        self.sample_start = float(sample_start)
        self.sample_interval = float(sample_interval)
        self.seed = int(seed)
        self.threads = int(threads)
        self.dx = 1.0 / self.nx
        self.dy = 1.0 / self.ny
        self.cell_area = self.dx * self.dy
        self.cells = self.nx * self.ny
        self.particle_weight = self.cell_area / self.particles_per_cell
        self.particle_count = self.cells * self.particles_per_cell
        self.x = (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dx
        self.y = (np.arange(self.ny, dtype=np.float64) + 0.5) * self.dy
        self.reference_speed = abs(self.lid_speed) + 4.0 * math.sqrt(max(1.0, self.wall_temperature))
        self.transport_time_step = self.transport_cfl * min(self.dx, self.dy) / self.reference_speed
        self.collision_time_step = self.collision_cfl * self.kn
        self.automatic_time_step = min(self.transport_time_step, self.collision_time_step)
        self.rng = np.random.default_rng(self.seed)
        self.positions = np.empty((0, 2), dtype=np.float64)
        self.velocities = np.empty((0, 3), dtype=np.float64)
        self.sample_times: list[float] = []
        self.sample_moments: list[np.ndarray] = []
        self.sample_statistics_sum = np.zeros((self.cells, 5), dtype=np.float64)
        self.total_collision_events = 0
        self.last_nominal_dt = math.nan
        self.last_completed_steps = 0

    @property
    def sample_count(self) -> int:
        return len(self.sample_times)

    def initialize_particles(self) -> None:
        cell = np.repeat(np.arange(self.cells, dtype=np.int64), self.particles_per_cell)
        ix, iy = cell // self.ny, cell % self.ny
        self.positions = np.empty((self.particle_count, 2), dtype=np.float64)
        self.positions[:, 0] = (ix + self.rng.random(self.particle_count)) * self.dx
        self.positions[:, 1] = (iy + self.rng.random(self.particle_count)) * self.dy
        self.velocities = self.rng.normal(0.0, 1.0, size=(self.particle_count, 3))

    def transport(self, dt: float) -> int:
        self.positions += dt * self.velocities[:, :2]
        outside = np.any((self.positions < 0.0) | (self.positions > 1.0), axis=1)
        crossing = np.flatnonzero(outside).astype(np.int64, copy=False)
        if crossing.size:
            random_values = self.rng.random((crossing.size, MAX_WALL_HITS, 3))
            _reflect_crossing_particles(
                self.positions, self.velocities, crossing, dt, random_values,
                self.wall_temperature, self.lid_speed,
            )
        return int(crossing.size)

    def _collision_schedule(
        self, order: np.ndarray, offsets: np.ndarray, counts: np.ndarray, dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pairs = 0.5 * counts.astype(np.float64) * np.maximum(counts - 1, 0)
        expected = pairs * self.particle_weight * dt / (self.kn * self.cell_area)
        collision_counts = np.floor(expected + self.rng.random(self.cells)).astype(np.int64)
        collision_counts[counts < 2] = 0
        collision_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(collision_counts)))
        event_cells = np.repeat(np.arange(self.cells, dtype=np.int64), collision_counts)
        total = event_cells.size
        if total == 0:
            return (
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty((0, 3), dtype=np.float64), collision_offsets,
            )
        local_count = counts[event_cells]
        first_local = np.floor(self.rng.random(total) * local_count).astype(np.int64)
        second_local = np.floor(self.rng.random(total) * (local_count - 1)).astype(np.int64)
        second_local += second_local >= first_local
        begin = offsets[event_cells]
        first = order[begin + first_local]
        second = order[begin + second_local]
        cosine = 2.0 * self.rng.random(total) - 1.0
        azimuth = 2.0 * math.pi * self.rng.random(total)
        radial = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
        directions = np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth), cosine))
        return first, second, directions, collision_offsets

    def advance(self, dt: float) -> tuple[np.ndarray, int, int]:
        wall_hits = self.transport(dt)
        order, offsets, counts = _cell_counting_order(self.positions, self.nx, self.ny)
        first, second, directions, collision_offsets = self._collision_schedule(order, offsets, counts, dt)
        _apply_isotropic_collisions(self.velocities, first, second, directions, collision_offsets)
        collision_events = int(first.size)
        self.total_collision_events += collision_events
        return counts, collision_events, wall_hits

    def cell_statistics(self) -> np.ndarray:
        ix = np.minimum(self.nx - 1, (self.positions[:, 0] * self.nx).astype(np.int64))
        iy = np.minimum(self.ny - 1, (self.positions[:, 1] * self.ny).astype(np.int64))
        cell = ix * self.ny + iy
        statistics = np.empty((self.cells, 5), dtype=np.float64)
        statistics[:, 0] = np.bincount(cell, minlength=self.cells)
        for component in range(3):
            statistics[:, component + 1] = np.bincount(
                cell, weights=self.velocities[:, component], minlength=self.cells
            )
        speed_squared = np.einsum("ij,ij->i", self.velocities, self.velocities)
        statistics[:, 4] = np.bincount(cell, weights=speed_squared, minlength=self.cells)
        return statistics

    def moments_from_statistics(self, statistics: np.ndarray) -> np.ndarray:
        counts = statistics[:, 0]
        safe = np.maximum(counts, 1.0)
        velocity = statistics[:, 1:4] / safe[:, None]
        temperature = (
            statistics[:, 4] / safe - np.einsum("ij,ij->i", velocity, velocity)
        ) / 3.0
        density = self.particle_weight * counts / self.cell_area
        return np.column_stack((density, velocity, temperature)).reshape(self.nx, self.ny, 5)

    def conservative_moments_from_statistics(self, statistics: np.ndarray) -> np.ndarray:
        scale = self.particle_weight / self.cell_area
        conservative = np.empty_like(statistics)
        conservative[:, :4] = scale * statistics[:, :4]
        conservative[:, 4] = 0.5 * scale * statistics[:, 4]
        return conservative.reshape(self.nx, self.ny, 5)

    def instantaneous_moments(self) -> np.ndarray:
        return self.moments_from_statistics(self.cell_statistics())

    def instantaneous_conservative_moments(self) -> np.ndarray:
        return self.conservative_moments_from_statistics(self.cell_statistics())

    def averaged_moments(self) -> np.ndarray:
        if not self.sample_count:
            return self.instantaneous_moments()
        return self.moments_from_statistics(self.sample_statistics_sum / self.sample_count)

    def collect_sample(self, time_value: float) -> None:
        statistics = self.cell_statistics()
        moments = self.moments_from_statistics(statistics)
        self.sample_statistics_sum += statistics
        self.sample_times.append(float(time_value))
        self.sample_moments.append(moments)

    def save_checkpoint(
        self, path: Path, *, time_value: float, completed_steps: int,
        nominal_dt: float, signature: dict[str, Any],
    ) -> None:
        history = np.stack(self.sample_moments) if self.sample_moments else np.empty(
            (0, self.nx, self.ny, 5), dtype=np.float64
        )
        payload = {
            "format_version": np.array(1, dtype=np.int64),
            "case": np.array("lid_driven_cavity_2d_dsmc"),
            "time": np.array(time_value),
            "completed_steps": np.array(completed_steps, dtype=np.int64),
            "nominal_dt": np.array(nominal_dt),
            "signature_json": np.array(json.dumps(json_ready(signature), sort_keys=True)),
            "rng_state_json": np.array(json.dumps(json_ready(self.rng.bit_generator.state))),
            "positions": self.positions,
            "velocities": self.velocities,
            "sample_times": np.asarray(self.sample_times),
            "sample_moments": history,
            "sample_statistics_sum": self.sample_statistics_sum,
            "total_collision_events": np.array(self.total_collision_events, dtype=np.int64),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **payload)
        os.replace(temporary, path)

    def load_checkpoint(
        self, path: Path, *, nominal_dt: float, signature: dict[str, Any],
    ) -> tuple[float, int]:
        with np.load(path, allow_pickle=False) as payload:
            if json.loads(str(payload["signature_json"])) != json_ready(signature):
                raise ValueError("resume checkpoint parameters do not match this run")
            if not math.isclose(float(payload["nominal_dt"]), nominal_dt, rel_tol=1.0e-14):
                raise ValueError("checkpoint time step does not match this run")
            time_value = float(payload["time"])
            completed_steps = int(payload["completed_steps"])
            self.positions = payload["positions"].astype(np.float64, copy=True)
            self.velocities = payload["velocities"].astype(np.float64, copy=True)
            self.sample_times = payload["sample_times"].astype(np.float64).tolist()
            self.sample_moments = [item.copy() for item in np.asarray(payload["sample_moments"])]
            self.sample_statistics_sum = payload["sample_statistics_sum"].astype(np.float64, copy=True)
            self.total_collision_events = int(payload["total_collision_events"])
            rng_state = json.loads(str(payload["rng_state_json"]))
        self.particle_count = self.positions.shape[0]
        self.rng.bit_generator.state = rng_state
        return time_value, completed_steps

    def _sample_due(self, time_value: float) -> bool:
        tolerance = 1.0e-11 * max(1.0, abs(time_value))
        if time_value < self.sample_start - tolerance:
            return False
        index = round((time_value - self.sample_start) / self.sample_interval)
        scheduled = self.sample_start + index * self.sample_interval
        return abs(time_value - scheduled) <= tolerance and (
            not self.sample_times or time_value > self.sample_times[-1] + tolerance
        )

    def _next_sample_time(self, time_value: float) -> float:
        if time_value < self.sample_start:
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
        checkpoint_time_interval: float | None = None,
        resume_from: Path | None = None,
    ) -> np.ndarray:
        nominal_dt = self.automatic_time_step if dt is None else min(float(dt), self.automatic_time_step)
        self.last_nominal_dt = nominal_dt
        if resume_from is None:
            self.initialize_particles()
            time_value, completed_steps = 0.0, 0
        else:
            time_value, completed_steps = self.load_checkpoint(
                resume_from, nominal_dt=nominal_dt, signature=signature
            )
            if progress:
                print(f"resumed {resume_from} at t={time_value:.10g}", flush=True)
        if self._sample_due(time_value):
            self.collect_sample(time_value)

        def checkpoint() -> None:
            self.save_checkpoint(
                checkpoint_path, time_value=time_value, completed_steps=completed_steps,
                nominal_dt=nominal_dt, signature=signature,
            )

        checkpoint()
        remaining_steps = max(1, math.ceil(max(0.0, t_end - time_value) / nominal_dt))
        report_stride = max(1, remaining_steps // 100)
        local_steps = 0
        next_sample = self._next_sample_time(time_value)
        next_time_checkpoint = math.inf
        if checkpoint_time_interval is not None and checkpoint_time_interval > 0.0:
            next_time_checkpoint = (
                math.floor(time_value / checkpoint_time_interval + 1.0e-12) + 1
            ) * checkpoint_time_interval
        last_checkpoint = time.monotonic()
        start = time.monotonic()
        try:
            while time_value < t_end - 1.0e-13 * max(1.0, t_end):
                step_dt = min(nominal_dt, t_end - time_value)
                if next_sample > time_value:
                    step_dt = min(step_dt, next_sample - time_value)
                counts, collision_events, wall_hits = self.advance(step_dt)
                time_value += step_dt
                completed_steps += 1
                local_steps += 1
                if self._sample_due(time_value):
                    self.collect_sample(time_value)
                    next_sample = self._next_sample_time(time_value)
                if progress and (local_steps == 1 or local_steps % report_stride == 0 or time_value >= t_end):
                    density = counts / self.particles_per_cell
                    print(
                        f"step={completed_steps}, t={time_value:.10g}/{t_end:.10g}, "
                        f"collisions={collision_events:,}, wall_hits={wall_hits:,}, "
                        f"rho_cells=[{density.min():.5g},{density.max():.5g}], "
                        f"samples={self.sample_count}, wall={time.monotonic() - start:.1f}s",
                        flush=True,
                    )
                time_checkpoint_due = time_value + 1.0e-12 >= next_time_checkpoint
                wall_checkpoint_due = (
                    checkpoint_interval_seconds > 0.0
                    and time.monotonic() - last_checkpoint >= checkpoint_interval_seconds
                )
                if time_checkpoint_due or wall_checkpoint_due:
                    checkpoint()
                    last_checkpoint = time.monotonic()
                    if time_checkpoint_due:
                        while time_value + 1.0e-12 >= next_time_checkpoint:
                            next_time_checkpoint += checkpoint_time_interval
                    if progress:
                        print(f"checkpoint saved at t={time_value:.10g}: {checkpoint_path}", flush=True)
        except KeyboardInterrupt:
            checkpoint()
            print(f"interrupted; checkpoint saved at t={time_value:.10g}: {checkpoint_path}", file=sys.stderr)
            raise
        checkpoint()
        self.last_completed_steps = completed_steps
        return self.instantaneous_moments()


def latest_compatible_checkpoint(
    root: Path, *, target: Path, t_end: float, nominal_dt: float,
    signature: dict[str, Any],
) -> tuple[Path, float] | None:
    expected = json_ready(signature)
    candidates = set(root.glob(f"*/{CHECKPOINT_NAME}"))
    if target.is_file():
        candidates.add(target)
    best: tuple[tuple[float, bool, int], Path, float] | None = None
    for path in candidates:
        try:
            with np.load(path, allow_pickle=False) as payload:
                valid = (
                    int(payload["format_version"]) == 1
                    and str(payload["case"]) == "lid_driven_cavity_2d_dsmc"
                    and json.loads(str(payload["signature_json"])) == expected
                    and math.isclose(float(payload["nominal_dt"]), nominal_dt, rel_tol=1.0e-14)
                )
                saved_time = float(payload["time"])
            if not valid or not 0.0 <= saved_time <= t_end:
                continue
            key = (saved_time, path == target, path.stat().st_mtime_ns)
        except Exception:
            continue
        if best is None or key > best[0]:
            best = (key, path, saved_time)
    return None if best is None else (best[1], best[2])
