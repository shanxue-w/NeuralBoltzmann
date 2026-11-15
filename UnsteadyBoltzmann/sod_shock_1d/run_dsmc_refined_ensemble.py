"""Fixed-time DSMC ensemble on the same local mesh as run_strang.py."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UnsteadyBoltzmann.runtime import filename_float
from UnsteadyBoltzmann.sod_shock_1d.run_dsmc_refined import (
    CHECKPOINT_NAME, RefinedShockDSMC, X_LEFT, X_RIGHT,
    refined_edges, add_refinement_arguments, refinement_signature,
)
from UnsteadyBoltzmann.sod_shock_1d.run_dsmc_ensemble import (
    conservative_from_moments, moments_from_conservative, save_profiles,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-time DSMC ensemble for the 1D normal shock")
    parser.add_argument("--Nx", "--nx", dest="nx", type=int, default=100)
    parser.add_argument("--Kn", "--kn", dest="kn", type=float, default=0.1)
    parser.add_argument("--T", "--t-end", dest="t_end", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--Ma", "--ma", "--mach", dest="mach", type=float, default=2.0)
    parser.add_argument("--rho-minus", type=float, default=1.0)
    parser.add_argument("--temperature-minus", type=float, default=1.0)
    parser.add_argument("--velocity-half-width", type=float, default=10.0)
    parser.add_argument("--particles-per-upstream-cell", "--particles-per-cell", dest="particles_per_cell", type=int, default=2_000)
    parser.add_argument("--realizations", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--parallel-runs", type=int, default=4)
    parser.add_argument("--threads-per-run", type=int, default=4)
    parser.add_argument("--transport-cfl", type=float, default=0.5)
    parser.add_argument("--collision-cfl", type=float, default=0.25)
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "output_dsmc_refined")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return add_refinement_arguments(parser)


def automatic_time_step(args: argparse.Namespace) -> float:
    density_ratio = (args.gamma + 1.0) * args.mach**2 / ((args.gamma - 1.0) * args.mach**2 + 2.0)
    edges = refined_edges(args.nx, args.refine_factor, args.refine_center, args.refine_width, mesh=args.mesh, stretch=args.stretch)
    transport_dt = args.transport_cfl * np.diff(edges).min() / args.velocity_half_width
    collision_dt = args.collision_cfl * args.kn / (args.rho_minus * density_ratio)
    return min(transport_dt, collision_dt)


def parameterized_output_directory(args: argparse.Namespace) -> Path:
    dt = automatic_time_step(args) if args.dt is None else min(args.dt, automatic_time_step(args))
    parts = (
        f"nx{args.nx}", f"kn{filename_float(args.kn)}", f"tend{filename_float(args.t_end)}",
        f"dt{filename_float(dt)}", f"L{filename_float(args.velocity_half_width)}",
        f"Ma{filename_float(args.mach)}", f"gam{filename_float(args.gamma)}",
        f"rhom{filename_float(args.rho_minus)}", f"Tm{filename_float(args.temperature_minus)}",
        "Tstd", "dsmc", f"ensemble{args.realizations}", f"ppc{args.particles_per_cell}", f"seed{args.base_seed}",
    )
    import hashlib
    digest = hashlib.sha256(json.dumps(refinement_signature(args), sort_keys=True).encode()).hexdigest()[:12]
    return args.output_dir / ("_".join(parts) + "_refined_" + digest)


def realization_signature(args: argparse.Namespace, seed: int) -> dict:
    return {
        "case": "sod_shock_1d_dsmc_terminal_ensemble",
        **refinement_signature(args),
        "Nx": args.nx, "Kn": args.kn, "gamma": args.gamma, "Mach": args.mach,
        "rho_minus": args.rho_minus, "temperature_minus": args.temperature_minus,
        "temperature_convention": "standard", "particles_per_upstream_cell": args.particles_per_cell,
        "velocity_half_width": args.velocity_half_width, "transport_cfl": args.transport_cfl,
        "collision_cfl": args.collision_cfl, "sample_start": args.t_end + 1.0,
        "sample_interval": 1.0, "seed": seed, "collision_kernel": "Maxwell_B_1_over_4pi",
        "boundary": "half_range_Maxwellian_inflow",
    }


def run_realization(task: tuple[argparse.Namespace, int, Path]) -> np.ndarray:
    args, seed, output_dir = task
    result_dir = output_dir / "realizations" / f"seed{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = result_dir / CHECKPOINT_NAME
    terminal = result_dir / "terminal_conservative_moments.npy"
    if terminal.is_file() and not args.overwrite:
        cached = np.load(terminal)
        mesh = refined_edges(args.nx, args.refine_factor, args.refine_center, args.refine_width, mesh=args.mesh, stretch=args.stretch)
        if cached.shape != (len(mesh)-1, 5) or not np.isfinite(cached).all():
            raise ValueError(f"invalid cached terminal moments: {terminal}")
        checkpoint.unlink(missing_ok=True)
        return cached
    solver = RefinedShockDSMC(
        mesh=args.mesh, stretch=args.stretch,
        refine_factor=args.refine_factor, refine_center=args.refine_center, refine_width=args.refine_width,
        nx=args.nx, kn=args.kn, gamma=args.gamma, mach=args.mach,
        rho_minus=args.rho_minus, temperature_minus=args.temperature_minus,
        particles_per_upstream_cell=args.particles_per_cell,
        velocity_half_width=args.velocity_half_width, velocity_bins=1,
        transport_cfl=args.transport_cfl, collision_cfl=args.collision_cfl,
        sample_start=args.t_end + 1.0, sample_interval=1.0,
        seed=seed, threads=args.threads_per_run,
    )
    resume = checkpoint.is_file() and not args.overwrite
    moments = solver.run(
        t_end=args.t_end, dt=args.dt, progress=False, checkpoint_path=checkpoint,
        checkpoint_interval_seconds=60.0 * args.checkpoint_minutes,
        signature=realization_signature(args, seed), resume=resume,
        resume_from_path=checkpoint if resume else None,
    )
    if not np.isfinite(moments).all():
        raise FloatingPointError("empty or nonfinite terminal cell; increase particles-per-cell")
    conservative = conservative_from_moments(moments)
    temporary = terminal.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, conservative)
    temporary.replace(terminal)
    checkpoint.unlink(missing_ok=True)
    return conservative


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.realizations < 1 or args.parallel_runs < 1 or args.particles_per_cell < 1:
        raise ValueError("realizations, parallel-runs and particles-per-cell must be positive")
    edges = refined_edges(args.nx, args.refine_factor, args.refine_center, args.refine_width, mesh=args.mesh, stretch=args.stretch)
    x = (edges[:-1] + edges[1:]) / 2
    output_dir = parameterized_output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = np.arange(args.base_seed, args.base_seed + args.realizations, dtype=np.int64)
    tasks = [(args, int(seed), output_dir) for seed in seeds]
    if not args.quiet:
        print(f"fixed-time SOD DSMC ensemble: R={args.realizations}, t={args.t_end:g}, Nx={len(x)}, ppc={args.particles_per_cell:,}", flush=True)
        print(f"parallel runs={args.parallel_runs}, threads/run={args.threads_per_run}; output={output_dir}", flush=True)
    if args.parallel_runs == 1:
        conservative_stack = np.stack(list(map(run_realization, tasks)))
    else:
        with ProcessPoolExecutor(max_workers=args.parallel_runs, mp_context=multiprocessing.get_context("spawn")) as executor:
            conservative_stack = np.stack(list(executor.map(run_realization, tasks)))

    conservative = conservative_stack.mean(axis=0)
    realization_moments = np.stack([moments_from_conservative(values) for values in conservative_stack])
    moments = moments_from_conservative(conservative)
    divisor = math.sqrt(args.realizations)
    conservative_error = np.zeros_like(conservative) if args.realizations == 1 else conservative_stack.std(axis=0, ddof=1) / divisor
    moment_error = np.zeros_like(moments) if args.realizations == 1 else realization_moments.std(axis=0, ddof=1) / divisor
    arrays = {
        "seeds": seeds, "x": x, "x_edges": edges,
        "realization_conservative_moments": conservative_stack, "realization_moments": realization_moments,
        "conservative_moments_standard_error": conservative_error, "moments_standard_error": moment_error,
        "final_conservative_moments": conservative, "final_moments": moments,
    }
    for name, values in arrays.items():
        np.save(output_dir / f"{name}.npy", values)
    save_profiles(output_dir, arrays["x"], moments, args.t_end)
    config = {key: value for key, value in vars(args).items() if key not in ("output_dir", "overwrite", "quiet")}
    config.update({
        "temperature_convention": "standard", "seeds": seeds.tolist(),
        **refinement_signature(args), "base_nx": args.nx, "Nx": len(x),
        "dx_min": float(np.diff(edges).min()),
        "nominal_dt": automatic_time_step(args) if args.dt is None else min(args.dt, automatic_time_step(args)),
        "upstream_particles_per_smallest_cell": args.particles_per_cell * float(np.diff(edges).min()) / ((X_RIGHT-X_LEFT)/args.nx),
        "ensemble_rule": "average terminal rho, rho*u and E at identical T, then reconstruct u and T",
        "time_averaging": False,
        "per_seed_output": "terminal conservative five moments; checkpoints retained only while incomplete",
        "mean_density": float(np.average(moments[:, 0], weights=np.diff(edges))),
        "density_range": [float(moments[:, 0].min()), float(moments[:, 0].max())],
    })
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved terminal ensemble: mean rho={config['mean_density']:.16g}, rho=[{config['density_range'][0]:.6g},{config['density_range'][1]:.6g}]", flush=True)
    print(f"saved five moments and three PDFs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
