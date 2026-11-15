"""Independent fixed-time DSMC ensemble for the stretched cavity."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UnsteadyBoltzmann.lid_cavacity_2d._uniform_dsmc import (
    CHECKPOINT_NAME, LidDrivenCavityDSMC, automatic_time_step as uniform_time_step,
)
from UnsteadyBoltzmann.lid_cavacity_2d.postprocess import save_reference_fields_and_plots
from UnsteadyBoltzmann.lid_cavacity_2d.run_dsmc import LidDrivenCavityDSMC as StretchedDSMC
from UnsteadyBoltzmann.lid_cavacity_2d.snapshots import (
    fields_from_raw_moments, moment_snapshot_path, save_moment_snapshot,
)
from UnsteadyBoltzmann.runtime import filename_float, json_ready, snapshot_targets


def _build_parser(output_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-time DSMC ensemble for the lid-driven cavity")
    parser.add_argument("--Nx", "--nx", dest="nx", type=int, default=32)
    parser.add_argument("--Ny", "--ny", dest="ny", type=int, default=32)
    parser.add_argument("--stretch", type=float, default=None, help="use the new stretched-grid DSMC; omitted keeps the original solver")
    parser.add_argument("--Kn", "--kn", dest="kn", type=float, default=0.1)
    parser.add_argument("--T", "--t-end", dest="t_end", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--wall-temperature", type=float, default=1.0)
    parser.add_argument("--lid-speed", type=float, default=1.0)
    parser.add_argument("--particles-per-cell", type=int, default=2_000)
    parser.add_argument("--realizations", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--parallel-runs", type=int, default=4)
    parser.add_argument("--threads-per-run", type=int, default=4)
    parser.add_argument("--transport-cfl", type=float, default=0.5)
    parser.add_argument("--collision-cfl", type=float, default=0.25)
    parser.add_argument("--checkpoint-time-interval", type=float, default=0.1)
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0)
    parser.add_argument("--snapshot-interval", type=float, default=None,
                        help="exact 13-moment snapshot spacing for the stretched solver")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="retain full particle latest for continuation; stretched DSMC saves only on completion or handled Ctrl-C")
    parser.add_argument("--resume-from", type=Path,
                        help="previous ensemble result directory, for extending T with its per-seed latest checkpoints")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = _build_parser(CASE_DIR / "output")
    parser.description = __doc__
    parser.set_defaults(ny=48, stretch=2.0, t_end=1.0, dt=0.005,
                        realizations=200, snapshot_interval=0.5, checkpoint_minutes=0.0,
                        checkpoint_time_interval=None)
    for action in parser._actions:
        if action.dest in ("checkpoint_minutes", "checkpoint_time_interval"):
            action.help = "compatibility option; particle checkpoint is saved only on completion or handled Ctrl-C"
    return parser


def make_solver(args: argparse.Namespace, seed: int):
    common = dict(nx=args.nx, ny=args.ny, kn=args.kn, wall_temperature=args.wall_temperature,
                  lid_speed=args.lid_speed, particles_per_cell=args.particles_per_cell,
                  transport_cfl=args.transport_cfl, collision_cfl=args.collision_cfl,
                  seed=seed, threads=args.threads_per_run)
    if args.stretch is not None:
        return StretchedDSMC(**common, stretch=args.stretch)
    return LidDrivenCavityDSMC(**common, sample_start=args.t_end + 1.0, sample_interval=1.0)


def automatic_time_step(args: argparse.Namespace) -> float:
    if args.stretch is None:
        return uniform_time_step(args)
    return make_solver(args, args.base_seed).automatic_time_step


def parameterized_output_directory(args: argparse.Namespace) -> Path:
    parts = (
        f"nx{args.nx}", f"ny{args.ny}", f"kn{filename_float(args.kn)}",
        f"tend{filename_float(args.t_end)}", f"dt{filename_float(min(args.dt, automatic_time_step(args)))}",
        f"Ulid{filename_float(args.lid_speed)}", f"Tw{filename_float(args.wall_temperature)}",
        "Tstd",
        "dsmc", f"ensemble{args.realizations}", f"ppc{args.particles_per_cell}", f"seed{args.base_seed}",
    )
    if args.stretch is not None:
        parts = parts[:5] + (f"stretch{filename_float(args.stretch)}",) + parts[5:]
    return args.output_dir / "_".join(parts)


def realization_signature(args: argparse.Namespace, seed: int) -> dict:
    return {
        "case": "lid_driven_cavity_2d_dsmc_transient_ensemble",
        "Nx": args.nx, "Ny": args.ny, "Kn": args.kn,
        "wall_temperature": args.wall_temperature, "lid_speed": args.lid_speed,
        "temperature_convention": "standard",
        "particles_per_cell": args.particles_per_cell,
        "transport_cfl": args.transport_cfl, "collision_cfl": args.collision_cfl,
        "sample_start": args.t_end + 1.0, "sample_interval": 1.0, "seed": seed,
        "collision_kernel": "Maxwell_B_1_over_4pi",
        "boundary": "particle_conserving_Maxwell_diffuse_reflection",
    }


def run_realization(task: tuple[argparse.Namespace, int, Path]) -> np.ndarray:
    args, seed, output_dir = task
    result_dir = output_dir / "realizations" / f"seed{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = result_dir / CHECKPOINT_NAME
    terminal = result_dir / "terminal_conservative_moments.npy"
    if terminal.is_file() and not args.overwrite:
        if args.snapshot_interval is None:
            if args.stretch is None and not args.save_checkpoints:
                checkpoint.unlink(missing_ok=True)
        else:
            missing = [t for t in snapshot_targets(args.t_end, args.snapshot_interval)
                       if not moment_snapshot_path(result_dir / "snapshots", t).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"completed seed {seed} lacks requested moment snapshots; "
                    "use --overwrite to recompute its history")
        if args.save_checkpoints:
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"completed seed {seed} has no full particle checkpoint; "
                    "use --overwrite --save-checkpoints to recompute with continuation enabled")
            with np.load(checkpoint, allow_pickle=False) as data:
                if float(data["time"]) != args.t_end:
                    raise ValueError(
                        f"completed seed {seed} has no full checkpoint at T={args.t_end:g}; "
                        "use --overwrite --save-checkpoints to recompute")
        return np.load(terminal)
    solver = make_solver(args, seed)
    resume_from = checkpoint if checkpoint.is_file() and not args.overwrite else None
    if resume_from is None and args.resume_from is not None:
        previous = args.resume_from / "realizations" / f"seed{seed}"
        resume_from = previous / CHECKPOINT_NAME
        if not resume_from.is_file():
            raise FileNotFoundError(
                f"no resume checkpoint for seed {seed}: {resume_from}; "
                "the previous run must have been saved with --save-checkpoints")
        if args.snapshot_interval is not None:
            for time_value in snapshot_targets(args.t_end, args.snapshot_interval):
                source = moment_snapshot_path(previous / "snapshots", time_value)
                destination = moment_snapshot_path(result_dir / "snapshots", time_value)
                if source.is_file() and not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + ".tmp")
                    shutil.copy2(source, temporary)
                    os.replace(temporary, destination)
    # The stretched solver constructs and checks its own mesh/RNG signature.
    extra = {"signature": realization_signature(args, seed)} if args.stretch is None else {}
    if args.stretch is not None:
        extra["save_checkpoints"] = args.save_checkpoints
    if args.snapshot_interval is not None:
        extra.update(snapshot_interval=args.snapshot_interval, snapshot_dir=result_dir / "snapshots")
    solver.run(
        t_end=args.t_end, dt=args.dt, progress=False, checkpoint_path=checkpoint,
        checkpoint_interval_seconds=60.0 * args.checkpoint_minutes,
        checkpoint_time_interval=args.checkpoint_time_interval, **extra,
        resume_from=resume_from,
    )
    conservative = solver.instantaneous_conservative_moments()
    np.save(terminal, conservative)
    if args.stretch is None and not args.save_checkpoints:
        checkpoint.unlink()
    return conservative


def save_ensemble_snapshots(args, output_dir, seeds, mesh):
    for time_value in snapshot_targets(args.t_end, args.snapshot_interval):
        raw_stack, moment_stack, pressure_stack, heat_stack = [], [], [], []
        for seed in seeds:
            path = moment_snapshot_path(output_dir / "realizations" / f"seed{seed}" / "snapshots", time_value)
            with np.load(path, allow_pickle=False) as data:
                if float(data["time"]) != time_value:
                    raise ValueError(f"snapshot time does not match requested time: {path}")
                raw_stack.append(data["raw_moments"])
                moment_stack.append(data["moments"])
                pressure_stack.append(data["pressure_tensor"])
                heat_stack.append(data["heat_flux"])
        fields = fields_from_raw_moments(np.mean(raw_stack, axis=0))

        def standard_error(stack):
            return np.zeros_like(stack[0]) if len(seeds) == 1 else np.std(stack, axis=0, ddof=1) / math.sqrt(len(seeds))

        save_moment_snapshot(output_dir / "snapshots", time_value=time_value, **fields,
                             x=mesh.x, y=mesh.y, x_faces=mesh.edges[0], y_faces=mesh.edges[1],
                             cell_area=mesh.cell_area.reshape(args.nx, args.ny), seeds=seeds,
                             moments_standard_error=standard_error(moment_stack),
                             pressure_tensor_standard_error=standard_error(pressure_stack),
                             heat_flux_standard_error=standard_error(heat_stack),
                             metadata=dict(case="lid_cavacity_2d_dsmc_ensemble", T_end=time_value,
                                           Nx=args.nx, Ny=args.ny, Kn=args.kn, stretch=args.stretch,
                                           wall_temperature=args.wall_temperature, lid_speed=args.lid_speed,
                                           snapshot_interval=args.snapshot_interval, realizations=args.realizations,
                                           time_averaging=False, reference_title="DSMC ensemble snapshot",
                                           reference_source="ensemble_raw_13_moment_mean",
                                           ensemble_rule="average all 13 raw moments before centering"))
        if time_value == args.t_end:
            for name in ("pressure_tensor", "stress_tensor", "heat_flux", "raw_moments"):
                np.save(output_dir / f"final_{name}.npy", fields[name])


def moments_from_conservative(conservative: np.ndarray) -> np.ndarray:
    rho = conservative[..., 0]
    velocity = conservative[..., 1:4] / rho[..., None]
    temperature = (
        2.0 * conservative[..., 4] / rho - np.einsum("...i,...i->...", velocity, velocity)
    ) / 3.0
    return np.concatenate((rho[..., None], velocity, temperature[..., None]), axis=-1)


def main(
    argv: list[str] | None = None, *, parser: argparse.ArgumentParser | None = None,
    save_uniform_reference=save_reference_fields_and_plots,
) -> int:
    args = (parser or build_parser()).parse_args(argv)
    if args.snapshot_interval is not None:
        snapshot_targets(args.t_end, args.snapshot_interval)
        if args.stretch is None:
            raise ValueError("13-moment snapshots require --stretch (use 0 for a uniform mesh)")
    output_dir = parameterized_output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = np.arange(args.base_seed, args.base_seed + args.realizations, dtype=np.int64)
    tasks = [(args, int(seed), output_dir) for seed in seeds]
    if not args.quiet:
        print(
            f"fixed-time DSMC ensemble: R={args.realizations}, t={args.t_end:g}, "
            f"grid={args.nx}x{args.ny}, ppc={args.particles_per_cell:,}", flush=True,
        )
        if args.stretch is not None:
            checkpoint_policy = ("on completion or handled Ctrl-C only" if args.save_checkpoints else
                                 "disabled (enable with --save-checkpoints)")
        else:
            checkpoint_policy = f"dt={args.checkpoint_time_interval:g}"
        print(
            f"parallel runs={args.parallel_runs}, threads/run={args.threads_per_run}, "
            f"particle checkpoint: {checkpoint_policy}; output={output_dir}", flush=True,
        )
    if args.parallel_runs == 1:
        conservative_stack = np.stack(list(map(run_realization, tasks)))
    else:
        with ProcessPoolExecutor(
            max_workers=args.parallel_runs, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            conservative_stack = np.stack(list(executor.map(run_realization, tasks)))

    conservative = conservative_stack.mean(axis=0)
    realization_moments = np.stack([moments_from_conservative(values) for values in conservative_stack])
    moments = moments_from_conservative(conservative)
    standard_error = np.zeros_like(moments) if args.realizations == 1 else realization_moments.std(
        axis=0, ddof=1
    ) / math.sqrt(args.realizations)
    arrays = {
        "seeds": seeds, "realization_conservative_moments": conservative_stack,
        "realization_moments": realization_moments, "moments_standard_error": standard_error,
        "final_conservative_moments": conservative, "final_moments": moments,
        "final_density": moments[..., 0], "final_velocity": moments[..., 1:4],
        "final_temperature": moments[..., 4],
    }
    for name, values in arrays.items():
        np.save(output_dir / f"{name}.npy", values)
    x = (np.arange(args.nx, dtype=np.float64) + 0.5) / args.nx
    y = (np.arange(args.ny, dtype=np.float64) + 0.5) / args.ny
    save_reference = save_uniform_reference
    if args.stretch is not None:
        save_reference = save_reference_fields_and_plots
        mesh = make_solver(args, args.base_seed)
        x, y = mesh.x, mesh.y
        np.save(output_dir / "x_faces.npy", mesh.edges[0])
        np.save(output_dir / "y_faces.npy", mesh.edges[1])
        np.save(output_dir / "cell_area.npy", mesh.cell_area.reshape(args.nx, args.ny))
        if args.snapshot_interval is not None:
            save_ensemble_snapshots(args, output_dir, seeds, mesh)
    save_reference(
        output_dir, x=x, y=y, moments=moments, time_value=args.t_end,
        source_key="terminal_ensemble_conservative_mean",
        source_title=f"{args.realizations}-realization terminal ensemble mean",
    )
    config = {key: value for key, value in vars(args).items() if key not in ("output_dir", "overwrite", "quiet")}
    config.update({
        "temperature_convention": "standard",
        "seeds": seeds.tolist(),
        "ensemble_rule": "average terminal rho, rho*u and E at identical T, then reconstruct u and T",
        "time_averaging": False,
        "per_seed_output": (("13-moment snapshots and one full latest checkpoint per seed"
                             if args.save_checkpoints else
                             "13-moment snapshots; full particle checkpoint saving disabled")
                            if args.snapshot_interval is not None else
                            "terminal conservative five moments; full latest retained" if args.save_checkpoints else
                            "terminal conservative five moments; checkpoints retained only while incomplete"),
        "global_mass": float(moments[..., 0].mean()),
        "density_range": [float(moments[..., 0].min()), float(moments[..., 0].max())],
    })
    if args.stretch is not None:
        config.update(case="lid_cavacity_2d_dsmc_ensemble", T_end=args.t_end,
                      requested_dt=args.dt, nominal_dt=min(args.dt, mesh.automatic_time_step),
                      reference_title=f"DSMC {args.realizations}-realization terminal ensemble mean",
                      reference_source="terminal_ensemble_conservative_mean",
                      global_mass=float(np.dot(moments[..., 0].ravel(), mesh.cell_area)))
    (output_dir / "run_config.json").write_text(
        json.dumps(json_ready(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"saved terminal ensemble: mass={config['global_mass']:.16g}, "
        f"rho=[{config['density_range'][0]:.6g},{config['density_range'][1]:.6g}]",
        flush=True,
    )
    print(f"saved five moments and PDFs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
