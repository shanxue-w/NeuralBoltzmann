"""Run neural, neural-then-spectral, and pure-BGK Couette Kn sweeps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from SteadyBoltzmann.plot_couette_kn_sweep_1d import (
    KN_VALUES,
    kn_directory_name,
    plot_from_sweep,
)
from SteadyBoltzmann.io_1d import read_json, write_json
from SteadyBoltzmann.run_1d import DEFAULT_CHECKPOINT
from SteadyBoltzmann.numerics_1d import (
    DEFAULT_GATE_INNER_RELATIVE_TOLERANCE,
    MAX_INNER_ITERATIONS,
)


LEFT_VELOCITY = (0.0, -0.5, 0.0)
RIGHT_VELOCITY = (0.0, 0.5, 0.0)
WALL_TEMPERATURE = 1.0
X_LEFT = -0.5
X_RIGHT = 0.5
SWEEP_VARIANTS = (
    ("neural", "neural", False),
    ("neural_then_spectral", "neural", True),
    ("bgk", "bgk", False),
)


def _velocity_argument(value: tuple[float, float, float]) -> str:
    return ",".join(f"{component:g}" for component in value)


def neural_outer_tolerance(args: argparse.Namespace, kn: float) -> float:
    return args.kn01_neural_outer_tolerance if abs(kn - 0.1) < 1.0e-12 else args.outer_tolerance


def direction_outer_tolerance(
    args: argparse.Namespace,
    kn: float,
    collision_backend: str,
) -> float:
    return (
        neural_outer_tolerance(args, kn)
        if collision_backend == "neural"
        else args.outer_tolerance
    )


def build_case_command(
    args: argparse.Namespace,
    *,
    kn: float,
    case_dir: Path,
    final_spectral_refinement: bool,
    collision_backend: str = "neural",
) -> list[str]:
    if collision_backend == "bgk" and final_spectral_refinement:
        raise ValueError("the BGK sweep must remain BGK from start to finish")
    max_outer_iterations = (
        args.max_outer_iterations
        if collision_backend == "neural"
        else args.bgk_max_outer_iterations
    )
    command = [sys.executable, "-m", "SteadyBoltzmann.run_1d"]
    options = (
        ("--case", "couette"), ("--left-u", _velocity_argument(LEFT_VELOCITY)), ("--right-u", _velocity_argument(RIGHT_VELOCITY)),
        ("--left-temperature", WALL_TEMPERATURE), ("--right-temperature", WALL_TEMPERATURE), ("--x-left", X_LEFT), ("--x-right", X_RIGHT),
        ("--nx", args.nx), ("--nv", args.nv), ("--kn", kn), ("--device", args.device),
        ("--collision-backend", collision_backend),
        ("--outer-tolerance", direction_outer_tolerance(args, kn, collision_backend)),
        ("--final-outer-tolerance", args.outer_tolerance), ("--inner-tolerance", args.inner_tolerance),
        ("--max-outer-iterations", max_outer_iterations), ("--max-inner-iterations", args.max_inner_iterations),
        ("--spectral-switch-relative", args.spectral_switch_relative), ("--spectral-switch-outer-iteration", args.spectral_switch_outer_iteration),
        ("--R", args.R),
        ("--spectral-angular-chunk", args.spectral_angular_chunk), ("--spectral-batch-chunk", args.spectral_batch_chunk),
        ("--final-spectral-angular-accuracy", args.final_spectral_angular_accuracy), ("--final-inner-tolerance", args.final_inner_tolerance),
        ("--final-max-outer-iterations", args.final_max_outer_iterations), ("--final-max-inner-iterations", args.final_max_inner_iterations),
        ("--warmup", args.warmup), ("--output-dir", case_dir),
    )
    for flag, value in options:
        command.extend((flag, str(value)))
    if collision_backend == "neural":
        command.extend(("--checkpoint", str(args.checkpoint.resolve())))
        command.extend(("--gate-batch-chunk", str(args.gate_batch_chunk)))
    else:
        command.extend(("--velocity-L", str(args.velocity_L)))
    command.extend(("--overwrite", "--require-inner-convergence" if args.require_inner_convergence else "--no-require-inner-convergence", "--final-spectral-refinement" if final_spectral_refinement else "--no-final-spectral-refinement"))
    return command


def _completed_case_matches(
    case_dir: Path,
    args: argparse.Namespace,
    *,
    kn: float,
    final_spectral_refinement: bool,
    collision_backend: str = "neural",
) -> bool:
    summary_path = case_dir / "summary.json"
    manifest_path = case_dir / "run_manifest.json"
    solution_path = case_dir / "solution.npz"
    if not all(path.is_file() for path in (summary_path, manifest_path, solution_path)):
        return False
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    solver = manifest.get("solver", {})
    case = manifest.get("case", {})
    left = case.get("left_wall", {})
    right = case.get("right_wall", {})
    direction = manifest.get("direction_collision", {})
    velocity_grid = manifest.get("velocity_grid", {})
    final_refinement = manifest.get("final_refinement", {})
    final_spectral = final_refinement.get("exact_nonlinear_spectral")
    final_linear = final_refinement.get("local_maxwellian_linear_collision")
    expected_solver = {
        "nx": args.nx, "x_left": X_LEFT, "x_right": X_RIGHT, "kn": kn,
        "outer_tolerance": direction_outer_tolerance(args, kn, collision_backend), "final_refinement_outer_tolerance": args.outer_tolerance,
        "inner_relative_tolerance": args.inner_tolerance,
        "max_outer_iterations": args.max_outer_iterations if collision_backend == "neural" else args.bgk_max_outer_iterations,
        "max_inner_iterations": args.max_inner_iterations,
        "max_final_refinement_outer_iterations": args.final_max_outer_iterations, "max_final_refinement_inner_iterations": args.final_max_inner_iterations,
        "require_inner_convergence": args.require_inner_convergence, "spectral_switch_relative_residual": args.spectral_switch_relative,
        "spectral_switch_outer_iteration": args.spectral_switch_outer_iteration, "final_refinement_inner_relative_tolerance": args.final_inner_tolerance,
    }
    expected_refinement = {
        "enabled": final_spectral_refinement, "inner_relative_tolerance": args.final_inner_tolerance, "outer_tolerance": args.outer_tolerance,
        "max_outer_iterations": args.final_max_outer_iterations, "max_inner_iterations": args.final_max_inner_iterations,
        "direction_collision": "local_maxwellian_quasi_newton" if collision_backend == "neural" else None,
        "rhs_formula": "J_M(h-f)=-F_exact(f)" if collision_backend == "neural" else None,
        "post_refinement_residual_check": collision_backend == "neural",
    }
    common_matches = bool(
        summary.get("converged")
        and all(solver.get(key) == value for key, value in expected_solver.items())
        and tuple(left.get("velocity", ())) == LEFT_VELOCITY
        and tuple(right.get("velocity", ())) == RIGHT_VELOCITY
        and left.get("temperature") == WALL_TEMPERATURE
        and right.get("temperature") == WALL_TEMPERATURE
        and all(
            final_refinement.get(key) == value
            for key, value in expected_refinement.items()
        )
    )
    if collision_backend == "neural":
        direction_matches = bool(
            Path(str(direction.get("checkpoint"))).resolve() == args.checkpoint.resolve()
            and direction.get("batch_chunk") == args.gate_batch_chunk
            and direction.get("runtime_velocity_n") == args.nv
            and direction.get("negative_input_policy") == "clip_to_zero_at_gate_api"
        )
    else:
        direction_matches = bool(
            manifest.get("collision_backend") == "bgk"
            and manifest.get("solver_kind") == "newton_h_polarized_bgk"
            and summary.get("collision_backend") == "bgk"
            and direction.get("backend") == "bgk"
            and direction.get("formula") == "0.5 * rho * (M[f] - f)"
            and direction.get("rate") == 0.5
            and direction.get("n") == args.nv
            and direction.get("L") == args.velocity_L
            and direction.get("dtype") == "float64"
            and velocity_grid.get("n") == args.nv
            and velocity_grid.get("L") == args.velocity_L
        )
    matches = common_matches and direction_matches
    if not final_spectral_refinement:
        return matches
    expected_spectral = {"n": args.nv, "R": args.R, "angular_accuracy": args.final_spectral_angular_accuracy, "angular_chunk": args.spectral_angular_chunk, "batch_chunk": args.spectral_batch_chunk}
    expected_linear = {"n": args.nv, "R": args.R}
    return bool(matches and isinstance(final_spectral, dict) and isinstance(final_linear, dict) and all(final_spectral.get(key) == value for key, value in expected_spectral.items()) and all(final_linear.get(key) == value for key, value in expected_linear.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the configured Couette Kn sweep and plot five moments.")
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--nv", type=int, default=32)
    parser.add_argument("--velocity-L", type=float, default=10.0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-batch-chunk", type=int, default=32)
    parser.add_argument("--outer-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--kn01-neural-outer-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--inner-tolerance", type=float, default=DEFAULT_GATE_INNER_RELATIVE_TOLERANCE)
    parser.add_argument("--max-outer-iterations", type=int, default=50)
    parser.add_argument("--bgk-max-outer-iterations", type=int, default=100)
    parser.add_argument("--max-inner-iterations", type=int, default=MAX_INNER_ITERATIONS)
    parser.add_argument("--require-inner-convergence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spectral-switch-relative", type=float, default=1.0e-2)
    parser.add_argument("--spectral-switch-outer-iteration", type=int, default=10)
    parser.add_argument("--R", type=float, default=9.0)
    parser.add_argument("--spectral-angular-chunk", type=int, default=8)
    parser.add_argument("--spectral-batch-chunk", type=int, default=1)
    parser.add_argument("--final-spectral-angular-accuracy", type=int, default=23)
    parser.add_argument("--final-inner-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--final-max-outer-iterations", type=int, default=30)
    parser.add_argument("--final-max-inner-iterations", type=int, default=MAX_INNER_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="reuse only completed cases whose frozen parameters match")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_output = PACKAGE_DIR / "outputs" / f"couette_kn_sweep_nx{args.nx}_nv{args.nv}_{datetime.now():%Y%m%d_%H%M%S}"
    sweep_dir = (args.output_dir or default_output).resolve()
    if sweep_dir.exists() and any(sweep_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"nonempty output directory: {sweep_dir}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_manifest = {
        "format": "couette-kn-sweep-1d-v11",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kn_values": list(KN_VALUES),
        "domain": [X_LEFT, X_RIGHT],
        "nx": args.nx,
        "nv": args.nv,
        "left_wall": {
            "velocity": list(LEFT_VELOCITY),
            "temperature": WALL_TEMPERATURE,
        },
        "right_wall": {
            "velocity": list(RIGHT_VELOCITY),
            "temperature": WALL_TEMPERATURE,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "variants": [
            {
                "directory": name,
                "collision_backend": collision_backend,
                "final_spectral_refinement": enabled,
            }
            for name, collision_backend, enabled in SWEEP_VARIANTS
        ],
        "neural_outer_tolerances": {
            f"Kn={kn:g}": neural_outer_tolerance(args, kn) for kn in KN_VALUES
        },
        "bgk_outer_tolerance": args.outer_tolerance,
        "bgk_collision": {
            "formula": "0.5 * rho * (M[f] - f)",
            "maxwellian": "discrete_maximum_entropy_five_moment",
            "velocity_L": args.velocity_L,
        },
        "spectral_refinement_outer_tolerance": args.outer_tolerance,
        "solver_arguments": vars(args) | {"output_dir": str(sweep_dir)},
    }
    write_json(sweep_dir / "sweep_manifest.json", sweep_manifest)

    paths = []
    for variant_name, collision_backend, final_spectral_refinement in SWEEP_VARIANTS:
        variant_dir = sweep_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for kn in KN_VALUES:
            case_dir = variant_dir / kn_directory_name(kn)
            if args.resume and _completed_case_matches(
                case_dir,
                args,
                kn=kn,
                final_spectral_refinement=final_spectral_refinement,
                collision_backend=collision_backend,
            ):
                print(f"[{variant_name}, Kn={kn:g}] reusing verified converged result", flush=True)
                continue
            case_dir.mkdir(parents=True, exist_ok=True)
            command = build_case_command(
                args,
                kn=kn,
                case_dir=case_dir,
                final_spectral_refinement=final_spectral_refinement,
                collision_backend=collision_backend,
            )
            print(f"[{variant_name}, Kn={kn:g}] starting steady solve", flush=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        paths.extend(plot_from_sweep(variant_dir))

    print(f"sweep outputs: {sweep_dir}", flush=True)
    for path in paths:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
