"""Fixed-time DSMC on the same stretched mesh as run.py.

Reuse the original Maxwell-molecule collision kernel and diffuse-wall
particle transport. Particles have equal weight; initial cell populations
are proportional to cell area, with unbiased integer rounding. Thus
--particles-per-cell denotes the average population, not the lid population.
Collision selection and all densities use the actual cell areas.

The original DSMC transport-then-collision update is retained. Its time step
is capped using the smallest cell and the collision scale; the CSL solver's
large-step property does not transfer to DSMC. Outputs are snapshots at T,
without time averaging. Use independent --seed values for repeat samples.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
from numba import njit

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UnsteadyBoltzmann.lid_cavacity_2d._uniform_dsmc import (
    CHECKPOINT_NAME, LidDrivenCavityDSMC as UniformDSMC,
    _apply_isotropic_collisions, latest_compatible_checkpoint,
)
from UnsteadyBoltzmann.runtime import clipped_time_step, filename_float, guard_output_directory, json_ready, snapshot_targets
from UnsteadyBoltzmann.lid_cavacity_2d.snapshots import SECOND_COMPONENTS, fields_from_raw_moments, save_moment_snapshot


@njit(cache=True)
def _cell_indices(positions, x_faces, y_faces):
    nx, ny = len(x_faces) - 1, len(y_faces) - 1
    cell = np.empty(len(positions), dtype=np.int64)
    for p in range(len(positions)):
        ix = min(nx - 1, max(0, np.searchsorted(x_faces, positions[p, 0], side="right") - 1))
        iy = min(ny - 1, max(0, np.searchsorted(y_faces, positions[p, 1], side="right") - 1))
        cell[p] = ix * ny + iy
    return cell


@njit(cache=True)
def _cell_counting_order(positions, x_faces, y_faces):
    cell = _cell_indices(positions, x_faces, y_faces)
    cells = (len(x_faces) - 1) * (len(y_faces) - 1)
    counts = np.bincount(cell, minlength=cells)
    offsets = np.empty(cells + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)
    cursor = offsets[:-1].copy()
    order = np.empty(len(cell), dtype=np.int64)
    for p in range(len(cell)):
        order[cursor[cell[p]]] = p
        cursor[cell[p]] += 1
    return order, offsets, counts


class LidDrivenCavityDSMC(UniformDSMC):
    def __init__(self, *, nx=32, ny=48, stretch=2.0, kn=0.1,
                 wall_temperature=1.0, lid_speed=0.15, particles_per_cell=2000,
                 transport_cfl=0.5, collision_cfl=0.25, seed=1234, threads=0):
        if any(int(n) != n or n < 1 for n in (nx, ny, particles_per_cell)):
            raise ValueError("Nx, Ny and particles-per-cell must be positive integers")
        if not math.isfinite(stretch) or stretch < 0:
            raise ValueError("stretch must be finite and nonnegative")
        if not all(math.isfinite(v) and v > 0 for v in (kn, wall_temperature, transport_cfl, collision_cfl)):
            raise ValueError("Kn, wall temperature and time-step factors must be finite and positive")
        if not math.isfinite(lid_speed) or threads < 0:
            raise ValueError("lid speed must be finite; threads must be nonnegative")
        super().__init__(nx=nx, ny=ny, kn=kn, wall_temperature=wall_temperature,
                         lid_speed=lid_speed, particles_per_cell=particles_per_cell,
                         transport_cfl=transport_cfl, collision_cfl=collision_cfl,
                         sample_start=math.inf, seed=seed, threads=threads)
        self.stretch = float(stretch)
        s = np.linspace(0.0, 1.0, self.ny + 1)
        y_faces = s if stretch == 0 else np.expm1(-stretch*s) / np.expm1(-stretch)
        self.edges = (np.linspace(0.0, 1.0, self.nx + 1), y_faces)
        widths_x, widths_y = (np.diff(edges) for edges in self.edges)
        if np.any(widths_y <= 0):
            raise ValueError("stretch is too large to resolve distinct cell faces")
        self.dx, self.dy = widths_x, widths_y
        self.cell_area = (widths_x[:, None] * widths_y[None, :]).ravel()
        self.particle_weight = 1.0 / self.particle_count
        self.x, self.y = ((edges[:-1] + edges[1:])/2 for edges in self.edges)
        self.transport_time_step = transport_cfl * min(widths_x.min(), widths_y.min()) / self.reference_speed
        self.automatic_time_step = min(self.transport_time_step, self.collision_time_step)

    def initialize_particles(self):
        # One random shift rounds all cumulative populations without bias,
        # keeps their total exact, and gives each cell floor(N A) or ceil(N A).
        cumulative = np.r_[0.0, np.cumsum(self.cell_area)] * self.particle_count
        cumulative[-1] = self.particle_count
        counts = np.diff(np.floor(cumulative + self.rng.random()).astype(np.int64))
        cell = np.repeat(np.arange(self.cells), counts)
        ix, iy = cell // self.ny, cell % self.ny
        self.positions = np.empty((self.particle_count, 2))
        for axis, indices in enumerate((ix, iy)):
            edges = self.edges[axis]
            self.positions[:, axis] = edges[indices] + self.rng.random(self.particle_count) * np.diff(edges)[indices]
        self.velocities = self.rng.normal(size=(self.particle_count, 3))

    def advance(self, dt):
        wall_hits = self.transport(dt)
        order, offsets, counts = _cell_counting_order(self.positions, *self.edges)
        # The inherited scheduler broadcasts its 1/cell_area against counts:
        # E[Ncoll,K] = N_K (N_K-1)/2 * weight * dt / (Kn * area_K).
        first, second, directions, collision_offsets = self._collision_schedule(order, offsets, counts, dt)
        _apply_isotropic_collisions(self.velocities, first, second, directions, collision_offsets)
        self.total_collision_events += len(first)
        return counts, len(first), wall_hits

    def cell_statistics(self):
        cell = _cell_indices(self.positions, *self.edges)
        statistics = np.empty((self.cells, 5))
        statistics[:, 0] = np.bincount(cell, minlength=self.cells)
        for k in range(3):
            statistics[:, k + 1] = np.bincount(cell, weights=self.velocities[:, k], minlength=self.cells)
        statistics[:, 4] = np.bincount(cell, weights=np.sum(self.velocities**2, axis=1), minlength=self.cells)
        return statistics

    def moments_from_statistics(self, statistics):
        moments = super().moments_from_statistics(statistics)
        # Density is zero in an empty cell; velocity and temperature are undefined.
        moments.reshape(-1, 5)[statistics[:, 0] == 0, 1:] = np.nan
        return moments

    def conservative_moments_from_statistics(self, statistics):
        conservative = statistics * (self.particle_weight / self.cell_area)[:, None]
        conservative[:, 4] *= 0.5  # Energy = integral |v|^2 f / 2, as in the original DSMC.
        return conservative.reshape(self.nx, self.ny, 5)

    def snapshot_fields(self):
        cell = _cell_indices(self.positions, *self.edges)
        raw = np.empty((self.cells, 13))
        raw[:, 0] = np.bincount(cell, minlength=self.cells)
        for i in range(3):
            raw[:, i + 1] = np.bincount(cell, weights=self.velocities[:, i], minlength=self.cells)
        for index, (i, j) in enumerate(SECOND_COMPONENTS, 4):
            raw[:, index] = np.bincount(cell, weights=self.velocities[:, i] * self.velocities[:, j],
                                       minlength=self.cells)
        energy = 0.5 * np.sum(self.velocities**2, axis=-1)
        for i in range(3):
            raw[:, 10 + i] = np.bincount(cell, weights=energy * self.velocities[:, i], minlength=self.cells)
        raw *= (self.particle_weight / self.cell_area)[:, None]
        return fields_from_raw_moments(raw.reshape(self.nx, self.ny, 13))

    def checkpoint_signature(self):
        return dict(case="lid_cavacity_2d_dsmc", Nx=self.nx, Ny=self.ny, Kn=self.kn,
                    stretch=self.stretch, x_faces=self.edges[0].tolist(), y_faces=self.edges[1].tolist(),
                    particles_per_cell=self.particles_per_cell, particle_weight=self.particle_weight,
                    seed=self.seed, wall_temperature=self.wall_temperature, lid_speed=self.lid_speed,
                    transport_cfl=self.transport_cfl, collision_cfl=self.collision_cfl,
                    collision_kernel="Maxwell_B_1_over_4pi", splitting="transport-collision",
                    boundary="particle_conserving_Maxwell_diffuse_reflection",
                    initialization="equal_weight_area_stratified_random_rounding_v1",
                    temperature_convention="standard", time_averaging=False)

    def run(self, *, t_end=1.0, dt=0.005, progress=True, checkpoint_path,
            checkpoint_interval_seconds=0.0, checkpoint_time_interval=None, resume_from=None,
            snapshot_interval=0.5, snapshot_dir=None, save_checkpoints=False):
        """Save moments on schedule; full particle checkpoints are opt-in.

        When enabled, particles are saved only at completion or handled Ctrl-C.
        The checkpoint interval arguments remain accepted for compatibility
        with existing launch scripts, but do not trigger periodic writes.
        """
        if not math.isfinite(t_end) or t_end < 0 or (dt is not None and (not math.isfinite(dt) or dt <= 0)):
            raise ValueError("T must be finite and nonnegative; dt must be finite and positive")
        if not math.isfinite(checkpoint_interval_seconds) or checkpoint_interval_seconds < 0:
            raise ValueError("checkpoint interval in seconds must be finite and nonnegative")
        if checkpoint_time_interval is not None and (
            not math.isfinite(checkpoint_time_interval) or checkpoint_time_interval <= 0
        ):
            raise ValueError("checkpoint-time-interval must be finite and positive")
        nominal_dt = self.automatic_time_step if dt is None else min(dt, self.automatic_time_step)
        self.last_nominal_dt = nominal_dt
        targets = snapshot_targets(t_end, snapshot_interval)
        snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else Path(checkpoint_path).parent / "snapshots"
        signature = self.checkpoint_signature()
        if resume_from is None:
            self.initialize_particles()
            time_value, steps = 0.0, 0
            self.total_collision_events = 0
        else:
            time_value, steps = self.load_checkpoint(resume_from, nominal_dt=nominal_dt, signature=signature)
            if time_value > t_end:
                raise ValueError("checkpoint time exceeds requested T")

        def checkpoint():
            if save_checkpoints:
                self.save_checkpoint(checkpoint_path, time_value=time_value, completed_steps=steps,
                                     nominal_dt=nominal_dt, signature=signature)

        def snapshot():
            fields = self.snapshot_fields()
            save_moment_snapshot(snapshot_dir, time_value=time_value, **fields, x=self.x, y=self.y,
                                 x_faces=self.edges[0], y_faces=self.edges[1],
                                 cell_area=self.cell_area.reshape(self.nx, self.ny),
                                 metadata=dict(signature, T_end=time_value, completed_steps=steps,
                                               nominal_dt=nominal_dt, snapshot_interval=snapshot_interval,
                                               save_checkpoints=save_checkpoints,
                                               reference_title="DSMC instantaneous snapshot",
                                               reference_source="instantaneous_13_moments"))

        if time_value in targets:
            snapshot()
        future_targets = [target for target in targets if target > time_value]
        target_index = 0
        started = time.monotonic()
        report_stride = max(1, math.ceil((t_end - time_value)/nominal_dt) // 100)
        interrupted = False

        def defer_interrupt(*_):
            nonlocal interrupted
            interrupted = True

        previous_handler = signal.signal(signal.SIGINT, defer_interrupt)
        try:
            while time_value < t_end:
                target = future_targets[target_index]
                step_dt, next_time = clipped_time_step(time_value, nominal_dt, target)
                counts, collisions, wall_hits = self.advance(step_dt)
                time_value = next_time
                steps += 1
                snapshot_due = time_value == target
                if progress and (steps == 1 or steps % report_stride == 0 or time_value >= t_end):
                    rho = counts * self.particle_weight / self.cell_area
                    print(f"step={steps}, t={time_value:.10g}/{t_end:g}, collisions={collisions:,}, "
                          f"wall_hits={wall_hits:,}, rho=[{rho.min():.5g},{rho.max():.5g}], "
                          f"wall={time.monotonic()-started:.1f}s", flush=True)
                if snapshot_due:
                    snapshot()
                    target_index += 1
                if interrupted:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            checkpoint()
            if save_checkpoints:
                print(f"interrupted; completed-step checkpoint at t={time_value:.10g}: {checkpoint_path}", file=sys.stderr)
            else:
                print(f"interrupted at completed step t={time_value:.10g}; full particle checkpoint saving is disabled",
                      file=sys.stderr)
            raise
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        checkpoint()
        self.last_completed_steps = steps
        return self.instantaneous_moments()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in (("Nx", 32), ("Ny", 48)):
        parser.add_argument(f"--{name}", f"--{name.lower()}", dest=name.lower(), type=int, default=default)
    parser.add_argument("--Kn", "--kn", dest="kn", type=float, default=0.1)
    parser.add_argument("--T", "--t-end", dest="t_end", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.005, help="requested step; capped by DSMC mesh/collision scales")
    parser.add_argument("--stretch", type=float, default=2.0)
    parser.add_argument("--wall-temperature", type=float, default=1.0)
    parser.add_argument("--lid-speed", type=float, default=0.15)
    parser.add_argument("--particles-per-cell", type=int, default=2000, help="average population; total = Nx*Ny*this value")
    parser.add_argument("--transport-cfl", type=float, default=0.5)
    parser.add_argument("--collision-cfl", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--checkpoint-minutes", type=float, default=0.0,
                        help="compatibility option; particle checkpoint is saved only on completion or handled Ctrl-C")
    parser.add_argument("--checkpoint-time-interval", type=float, default=None,
                        help="compatibility option; does not enable periodic particle checkpoints")
    parser.add_argument("--snapshot-interval", type=float, default=0.5,
                        help="exact simulation-time spacing of 13-moment snapshots; also save t=0 and T")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="save full particle latest on completion or handled Ctrl-C; disabled by default")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path,
                        help="resume an explicit full particle checkpoint, including when extending T")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def save_results(output, solver, moments, configuration):
    output.mkdir(parents=True, exist_ok=True)
    counts = solver.cell_statistics()[:, 0].reshape(solver.nx, solver.ny).astype(np.int64)
    fields = solver.snapshot_fields()
    for name, array in dict(x=solver.x, y=solver.y, x_faces=solver.edges[0], y_faces=solver.edges[1],
                            cell_area=solver.cell_area.reshape(solver.nx, solver.ny), cell_particle_counts=counts,
                            final_moments=moments, reference_moments=moments,
                            final_conservative_moments=solver.instantaneous_conservative_moments(),
                            final_pressure_tensor=fields["pressure_tensor"], final_stress_tensor=fields["stress_tensor"],
                            final_heat_flux=fields["heat_flux"],
                            final_raw_moments=fields["raw_moments"],
                            final_density=moments[..., 0], final_velocity=moments[..., 1:4],
                            final_temperature=moments[..., 4]).items():
        np.save(output / f"{name}.npy", array)
    configuration.update(method="DSMC_Bird_NTC", reference_source="terminal_instantaneous_moments",
                         moment_columns=["rho", "ux", "uy", "uz", "temperature"],
                         conservative_columns=["rho", "rho_ux", "rho_uy", "rho_uz", "energy"],
                         mass=float(np.dot(moments[..., 0].ravel(), solver.cell_area)),
                         empty_cells=int(np.count_nonzero(counts == 0)), empty_cell_velocity_temperature="NaN")
    (output / "run_config.json").write_text(json.dumps(json_ready(configuration), indent=2) + "\n")
    from UnsteadyBoltzmann.lid_cavacity_2d.postprocess import save_moment_plots
    streamfunction = save_moment_plots(output, x=solver.x, y=solver.y, moments=moments,
                                     time_value=configuration["T_end"], source_title="DSMC terminal state",
                                     source_key="terminal_instantaneous_moments")
    np.save(output / "reference_streamfunction.npy", streamfunction)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.dt) or args.dt <= 0 or not math.isfinite(args.t_end) or args.t_end < 0:
        raise ValueError("dt must be finite and positive; T must be finite and nonnegative")
    if not math.isfinite(args.checkpoint_minutes) or args.checkpoint_minutes < 0:
        raise ValueError("checkpoint-minutes must be finite and nonnegative")
    if args.checkpoint_time_interval is not None and (
        not math.isfinite(args.checkpoint_time_interval) or args.checkpoint_time_interval <= 0
    ):
        raise ValueError("checkpoint-time-interval must be finite and positive")
    keys = ("nx", "ny", "stretch", "kn", "wall_temperature", "lid_speed", "particles_per_cell",
            "transport_cfl", "collision_cfl", "seed", "threads")
    solver = LidDrivenCavityDSMC(**{key: getattr(args, key) for key in keys})
    nominal_dt = min(args.dt, solver.automatic_time_step)
    name = "_".join((f"nx{args.nx}", f"ny{args.ny}", f"kn{filename_float(args.kn)}",
                     f"tend{filename_float(args.t_end)}", f"dt{filename_float(nominal_dt)}",
                     f"stretch{filename_float(args.stretch)}", f"Ulid{filename_float(args.lid_speed)}",
                     f"Tw{filename_float(args.wall_temperature)}", "dsmc",
                     f"ppc{args.particles_per_cell}", f"seed{args.seed}"))
    output = args.output_dir / name
    guard_output_directory(output, overwrite=args.overwrite, resume=args.resume or args.resume_from is not None)
    checkpoint = output / CHECKPOINT_NAME
    resume_source = args.resume_from
    if args.resume and resume_source is None:
        selected = latest_compatible_checkpoint(args.output_dir, target=checkpoint, t_end=args.t_end,
                                               nominal_dt=nominal_dt, signature=solver.checkpoint_signature())
        if selected is None:
            raise FileNotFoundError("no compatible stretched DSMC checkpoint at or before T")
        resume_source = selected[0]
    if not args.quiet:
        expected = solver.particle_count * solver.cell_area
        print(f"grid={args.nx}x{args.ny}, stretch={args.stretch:g}, particles={solver.particle_count:,}, "
              f"initial expected particles/cell=[{expected.min():.1f},{expected.max():.1f}]", flush=True)
        print(f"requested dt={args.dt:g}; effective dt={nominal_dt:.10g}; "
              f"transport cap={solver.transport_time_step:.10g}; collision cap={solver.collision_time_step:g}", flush=True)
        print(f"fixed-time snapshot T={args.t_end:g}; output={output}", flush=True)
        policy = "on completion or handled Ctrl-C only" if args.save_checkpoints else "disabled (enable with --save-checkpoints)"
        print(f"full particle checkpoint: {policy}", flush=True)
        if resume_source is not None:
            print(f"resume source: {resume_source}", flush=True)
    moments = solver.run(t_end=args.t_end, dt=args.dt, progress=not args.quiet, checkpoint_path=checkpoint,
                         checkpoint_interval_seconds=60*args.checkpoint_minutes,
                         checkpoint_time_interval=args.checkpoint_time_interval, resume_from=resume_source,
                         snapshot_interval=args.snapshot_interval, snapshot_dir=output/"snapshots",
                         save_checkpoints=args.save_checkpoints)
    save_results(output, solver, moments, dict(vars(args), **solver.checkpoint_signature(),
                 T_end=args.t_end, requested_dt=args.dt, nominal_dt=solver.last_nominal_dt,
                 particle_count=solver.particle_count, completed_steps=solver.last_completed_steps,
                 total_collision_events=solver.total_collision_events, resume_source=resume_source))
    print(f"saved DSMC snapshot, cell populations, centerlines and PDFs to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
