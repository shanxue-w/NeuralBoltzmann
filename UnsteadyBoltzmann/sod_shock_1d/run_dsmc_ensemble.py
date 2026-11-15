"""Independent fixed-time DSMC ensemble for the 1D normal shock."""

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
from UnsteadyBoltzmann.sod_shock_1d.run_dsmc import CHECKPOINT_NAME, SodShockDSMC, X_LEFT, X_RIGHT


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
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def automatic_time_step(args: argparse.Namespace) -> float:
    density_ratio = (args.gamma + 1.0) * args.mach**2 / ((args.gamma - 1.0) * args.mach**2 + 2.0)
    transport_dt = args.transport_cfl * (X_RIGHT - X_LEFT) / (args.nx * args.velocity_half_width)
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
    return args.output_dir / "_".join(parts)


def realization_signature(args: argparse.Namespace, seed: int) -> dict:
    return {
        "case": "sod_shock_1d_dsmc_terminal_ensemble",
        "Nx": args.nx, "Kn": args.kn, "gamma": args.gamma, "Mach": args.mach,
        "rho_minus": args.rho_minus, "temperature_minus": args.temperature_minus,
        "temperature_convention": "standard", "particles_per_upstream_cell": args.particles_per_cell,
        "velocity_half_width": args.velocity_half_width, "transport_cfl": args.transport_cfl,
        "collision_cfl": args.collision_cfl, "sample_start": args.t_end + 1.0,
        "sample_interval": 1.0, "seed": seed, "collision_kernel": "Maxwell_B_1_over_4pi",
        "boundary": "half_range_Maxwellian_inflow",
    }


def conservative_from_moments(moments: np.ndarray) -> np.ndarray:
    rho, velocity, temperature = moments[:, 0], moments[:, 1:4], moments[:, 4]
    conservative = np.empty_like(moments)
    conservative[:, 0] = rho
    conservative[:, 1:4] = rho[:, None] * velocity
    conservative[:, 4] = 0.5 * rho * (np.einsum("ij,ij->i", velocity, velocity) + 3.0 * temperature)
    return conservative


def moments_from_conservative(conservative: np.ndarray) -> np.ndarray:
    rho = conservative[:, 0]
    velocity = conservative[:, 1:4] / rho[:, None]
    temperature = (2.0 * conservative[:, 4] / rho - np.einsum("ij,ij->i", velocity, velocity)) / 3.0
    return np.column_stack((rho, velocity, temperature))


def run_realization(task: tuple[argparse.Namespace, int, Path]) -> np.ndarray:
    args, seed, output_dir = task
    result_dir = output_dir / "realizations" / f"seed{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = result_dir / CHECKPOINT_NAME
    terminal = result_dir / "terminal_conservative_moments.npy"
    if terminal.is_file() and not args.overwrite:
        checkpoint.unlink(missing_ok=True)
        return np.load(terminal)
    solver = SodShockDSMC(
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
    conservative = conservative_from_moments(moments)
    np.save(terminal, conservative)
    checkpoint.unlink(missing_ok=True)
    return conservative


def save_profiles(output_dir: Path, x: np.ndarray, moments: np.ndarray, t_end: float) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for column, label, name in ((0, r"$\rho$", "rho_x.pdf"), (1, r"$u_x$", "u_x.pdf"), (4, r"$T$", "T_x.pdf")):
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot(x, moments[:, column], linewidth=1.8)
        axis.set(xlabel=r"$x$", ylabel=label, title=f"Terminal DSMC ensemble at t={t_end:g}")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / name, bbox_inches="tight")
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = parameterized_output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = np.arange(args.base_seed, args.base_seed + args.realizations, dtype=np.int64)
    tasks = [(args, int(seed), output_dir) for seed in seeds]
    if not args.quiet:
        print(f"fixed-time SOD DSMC ensemble: R={args.realizations}, t={args.t_end:g}, Nx={args.nx}, ppc={args.particles_per_cell:,}", flush=True)
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
        "seeds": seeds, "x": X_LEFT + (np.arange(args.nx) + 0.5) * (X_RIGHT - X_LEFT) / args.nx,
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
        "ensemble_rule": "average terminal rho, rho*u and E at identical T, then reconstruct u and T",
        "time_averaging": False,
        "per_seed_output": "terminal conservative five moments; checkpoints retained only while incomplete",
        "mean_density": float(moments[:, 0].mean()),
        "density_range": [float(moments[:, 0].min()), float(moments[:, 0].max())],
    })
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved terminal ensemble: mean rho={config['mean_density']:.16g}, rho=[{config['density_range'][0]:.6g},{config['density_range'][1]:.6g}]", flush=True)
    print(f"saved five moments and three PDFs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
