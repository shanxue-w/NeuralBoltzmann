"""DSMC on a statically refined mesh; particle weight uses the base cell width."""
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
from numba import njit

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from UnsteadyBoltzmann.runtime import filename_float, guard_output_directory, json_ready
from UnsteadyBoltzmann.sod_shock_1d.spatial_grid import grid_edges, add_mesh_arguments
from UnsteadyBoltzmann.sod_shock_1d.run_dsmc import (
    SodShockDSMC, X_LEFT, X_RIGHT, CHECKPOINT_NAME, _compact_inside,
    _apply_isotropic_collisions, latest_compatible_checkpoint,
    build_parser as base_parser, _checkpoint_signature as base_signature,
)


def refined_edges(nx, refine_factor=4, refine_center=0.0, refine_width=10.0,
                  *, mesh="sinh", stretch=5.0):
    return grid_edges(nx, mesh=mesh, stretch=stretch, refine_factor=refine_factor,
                      refine_center=refine_center, refine_width=refine_width)


def add_refinement_arguments(parser):
    parser.add_argument("--refine-factor", type=int, default=4)
    parser.add_argument("--refine-center", type=float, default=0.0)
    parser.add_argument("--refine-width", type=float, default=10.0,
                        help="full window width; selects base-cell centers")
    return add_mesh_arguments(parser)


def refinement_signature(args):
    if args.mesh == "sinh":
        return {"mesh_scheme": "sinh_dsmc_v1", "stretch": args.stretch,
                "particle_weight_rule": "upstream_base_cell"}
    return {"mesh_scheme": "static_refined_dsmc_v1", "refine_factor": args.refine_factor,
            "refine_center": args.refine_center, "refine_width": args.refine_width,
            "particle_weight_rule": "upstream_base_cell"}


@njit(cache=False)
def counting_order(positions, particle_count, edges):
    nx = len(edges) - 1
    cells = np.searchsorted(edges, positions[:particle_count], side="right") - 1
    cells = np.maximum(0, np.minimum(nx - 1, cells))
    counts = np.zeros(nx, dtype=np.int64)
    for cell in cells:
        counts[cell] += 1
    offsets = np.empty(nx + 1, dtype=np.int64)
    offsets[0] = 0
    for cell in range(nx):
        offsets[cell + 1] = offsets[cell] + counts[cell]
    cursor = offsets[:-1].copy()
    order = np.empty(particle_count, dtype=np.int64)
    for particle, cell in enumerate(cells):
        order[cursor[cell]] = particle
        cursor[cell] += 1
    return order, offsets, counts


class RefinedShockDSMC(SodShockDSMC):
    def __init__(self, *, refine_factor=4, refine_center=0.0, refine_width=10.0,
                 mesh="sinh", stretch=5.0, **kwargs):
        super().__init__(**kwargs)
        self.base_nx = self.nx
        self.edges = refined_edges(self.base_nx, refine_factor, refine_center, refine_width,
                                   mesh=mesh, stretch=stretch)
        self.widths = np.diff(self.edges)
        self.nx = len(self.widths)
        self.x_centers = (self.edges[:-1] + self.edges[1:]) / 2
        # Scalar dx remains the base width used to define the global particle weight.
        self.transport_time_step = self.transport_cfl * self.widths.min() / self.velocity_half_width
        self.automatic_time_step = min(self.transport_time_step, self.collision_time_step)
        self.vx_marginal_sum = np.zeros((self.nx, self.velocity_bins))
        if self.full_distribution_nv > 0:
            self.full_distribution_sum = np.zeros((self.nx,) + (self.full_distribution_nv,) * 3)

    def cell_indices(self, positions):
        return np.clip(np.searchsorted(self.edges, positions, side="right") - 1, 0, self.nx - 1)

    def initialize_particles(self):
        # Split a cell crossing zero to retain the prescribed initial jump location.
        segments = np.unique(np.append(self.edges, 0.0))
        center = (segments[:-1] + segments[1:]) / 2
        density = np.where(center < 0, self.rho_minus, self.rho_plus)
        expected = density * np.diff(segments) / self.particle_weight
        offsets = np.concatenate(([0], np.rint(np.cumsum(expected)).astype(np.int64)))
        self.particle_count = self.capacity = int(offsets[-1])
        self.positions = np.empty(self.particle_count)
        self.velocities = np.empty((self.particle_count, 3))
        for i, c in enumerate(center):
            begin, end = offsets[i:i+2]
            count = end - begin
            self.positions[begin:end] = segments[i] + (segments[i+1] - segments[i]) * self.rng.random(count)
            u = self.u_minus if c < 0 else self.u_plus
            temperature = self.temperature_minus if c < 0 else self.temperature_plus
            self.velocities[begin:end] = self.rng.normal((u, 0, 0), math.sqrt(temperature), (count, 3))

    def _collision_schedule(
        self,
        order: np.ndarray,
        offsets: np.ndarray,
        counts: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        particle_pairs = 0.5 * counts.astype(np.float64) * np.maximum(counts - 1, 0)
        expected = particle_pairs * self.particle_weight * dt / (self.kn * self.widths)
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
        order, offsets, counts = counting_order(self.positions, self.particle_count, self.edges)
        first, second, directions, collision_offsets = self._collision_schedule(order, offsets, counts, dt)
        _apply_isotropic_collisions(self.velocities, first, second, directions, collision_offsets)
        collision_events = int(first.size)
        self.total_collision_events += collision_events
        return counts, collision_events

    def instantaneous_moments(self) -> np.ndarray:
        positions = self.positions[: self.particle_count]
        velocities = self.velocities[: self.particle_count]
        cell = self.cell_indices(positions)
        np.clip(cell, 0, self.nx - 1, out=cell)
        counts = np.bincount(cell, minlength=self.nx).astype(np.float64)
        density = self.particle_weight * counts / self.widths
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
        cell = self.cell_indices(positions)
        np.clip(cell, 0, self.nx - 1, out=cell)
        dv = 2.0 * self.velocity_half_width / self.velocity_bins
        vx_bin = np.floor((velocities[:, 0] + self.velocity_half_width) / dv).astype(np.int64)
        inside_vx = (vx_bin >= 0) & (vx_bin < self.velocity_bins)
        flat_vx = cell[inside_vx] * self.velocity_bins + vx_bin[inside_vx]
        marginal = np.bincount(flat_vx, minlength=self.nx * self.velocity_bins).reshape(self.nx, self.velocity_bins)
        self.vx_marginal_sum += self.particle_weight / (self.widths[:, None] * dv) * marginal
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
            self.full_distribution_sum += self.particle_weight / (self.widths[:, None, None, None] * full_dv**3) * histogram

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
        if not math.isfinite(t_end) or t_end < 0 or (dt is not None and (not math.isfinite(dt) or dt <= 0)):
            raise ValueError("nonnegative finite terminal time and positive finite dt required")
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
                    density_min = float((self.particle_weight * counts / self.widths).min())
                    density_max = float((self.particle_weight * counts / self.widths).max())
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


def save_results(
    output_dir: Path,
    *,
    solver: Any,
    instantaneous_moments: np.ndarray,
    run_configuration: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if solver.sample_moments:
        history = np.stack(solver.sample_moments)
        averaged = history.mean(axis=0)
        standard_error = history.std(axis=0, ddof=1) / math.sqrt(history.shape[0]) if history.shape[0] > 1 else np.full_like(averaged, np.nan)
        averaged_vx = solver.vx_marginal_sum / history.shape[0]
    else:
        history = np.empty((0, solver.nx, 5), dtype=np.float64)
        averaged = instantaneous_moments.copy()
        standard_error = np.full_like(averaged, np.nan)
        averaged_vx = np.zeros_like(solver.vx_marginal_sum)

    dv = 2.0 * solver.velocity_half_width / solver.velocity_bins
    vx_centers = -solver.velocity_half_width + (np.arange(solver.velocity_bins, dtype=np.float64) + 0.5) * dv
    positions = solver.positions[: solver.particle_count]
    velocities = solver.velocities[: solver.particle_count]
    cell = solver.cell_indices(positions)
    np.clip(cell, 0, solver.nx - 1, out=cell)
    vx_bin = np.floor((velocities[:, 0] + solver.velocity_half_width) / dv).astype(np.int64)
    inside = (vx_bin >= 0) & (vx_bin < solver.velocity_bins)
    flat = cell[inside] * solver.velocity_bins + vx_bin[inside]
    final_vx = np.bincount(flat, minlength=solver.nx * solver.velocity_bins).reshape(solver.nx, solver.velocity_bins)
    final_vx = final_vx * (solver.particle_weight / (solver.widths[:, None] * dv))

    np.save(output_dir / "x_edges.npy", solver.edges)
    np.save(output_dir / "x.npy", solver.x_centers)
    np.save(output_dir / "final_moments.npy", instantaneous_moments)
    np.save(output_dir / "averaged_moments.npy", averaged)
    np.save(output_dir / "moment_standard_error.npy", standard_error)
    np.save(output_dir / "vx_centers.npy", vx_centers)
    np.save(output_dir / "final_vx_marginal.npy", final_vx)
    np.save(output_dir / "averaged_vx_marginal.npy", averaged_vx)
    np.savez(output_dir / "sampling_history.npz", times=np.asarray(solver.sample_times, dtype=np.float64), moments=history)
    if solver.full_distribution_sum is not None and solver.sample_count:
        np.save(output_dir / "averaged_f.npy", solver.full_distribution_sum / solver.sample_count)
        full_dv = 2.0 * solver.velocity_half_width / solver.full_distribution_nv
        centers = -solver.velocity_half_width + (np.arange(solver.full_distribution_nv, dtype=np.float64) + 0.5) * full_dv
        np.save(output_dir / "velocity_centers.npy", centers)

    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for column, ylabel, title, filename in (
        (0, r"$\rho$", "Final DSMC density", "rho_x.pdf"),
        (1, r"$u_x$", "Final DSMC velocity", "u_x.pdf"),
        (4, r"$T$", "Final DSMC temperature", "T_x.pdf"),
    ):
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot(solver.x_centers, instantaneous_moments[:, column], linewidth=1.8)
        axis.set(xlabel=r"$x$", ylabel=ylabel, title=title)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(figure)

    (output_dir / "run_config.json").write_text(
        json.dumps(json_ready(run_configuration), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser():
    parser = add_refinement_arguments(base_parser())
    parser.set_defaults(output_dir=CASE_DIR / "output_dsmc_refined")
    parser.description = __doc__
    return parser


def _checkpoint_signature(args):
    return {**base_signature(args), **refinement_signature(args)}


def parameterized_output_directory(args):
    edges = refined_edges(args.nx, args.refine_factor, args.refine_center, args.refine_width,
                           mesh=args.mesh, stretch=args.stretch)
    ratio = (args.gamma + 1) * args.mach**2 / ((args.gamma - 1) * args.mach**2 + 2)
    auto_dt = min(args.transport_cfl * np.diff(edges).min() / args.velocity_half_width,
                  args.collision_cfl * args.kn / (args.rho_minus * ratio))
    dt = auto_dt if args.dt is None else min(args.dt, auto_dt)
    import hashlib
    digest = hashlib.sha256(json.dumps(_checkpoint_signature(args), sort_keys=True).encode()).hexdigest()[:12]
    return args.output_dir / (f"nx{len(edges)-1}_kn{filename_float(args.kn)}_Ma{filename_float(args.mach)}"
        f"_tend{filename_float(args.t_end)}_dt{filename_float(dt)}_dsmc_refined_{digest}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_start > args.t_end and not args.quiet:
        print("warning: sample-start is after T; PDFs will use the instantaneous profile", file=sys.stderr, flush=True)
    output_dir = parameterized_output_directory(args)
    guard_output_directory(output_dir, overwrite=args.overwrite, resume=args.resume)

    solver = RefinedShockDSMC(
        mesh=args.mesh, stretch=args.stretch,
        refine_factor=args.refine_factor, refine_center=args.refine_center, refine_width=args.refine_width,
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
              f"upstream particles/base cell={solver.particles_per_upstream_cell:,}", flush=True)
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
        "base_nx": args.nx, "Nx": solver.nx, "dx_min": float(solver.widths.min()),
        "base_dx": solver.dx, "upstream_particles_per_smallest_cell": solver.rho_minus * solver.widths.min() / solver.particle_weight,
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
