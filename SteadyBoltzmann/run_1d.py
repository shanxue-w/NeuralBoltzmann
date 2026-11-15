"""Run a steady 1D Boltzmann experiment with neural/spectral or pure BGK collision."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import torch


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from SteadyBoltzmann.cases_1d import case_preset, with_wall_overrides
from SteadyBoltzmann.collision_1d import (
    BGKCollision,
    GateFNOCollision,
    LazyLocalMaxwellianLinearCollision,
    LazySpectralCollision,
    LinearCollisionConfig,
    SpectralConfig,
)
from SteadyBoltzmann.io_1d import json_ready, prepare_output, write_json, write_npz, write_profiles_csv
from SteadyBoltzmann.numerics_1d import (
    DEFAULT_GATE_INNER_RELATIVE_TOLERANCE,
    MAX_INNER_ITERATIONS,
    Solver1DConfig,
    SteadyBoltzmann1D,
    macroscopic_profiles,
    make_initial_maxwellian,
    make_velocity_grid,
)


DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / "boltzmann_tail_simplified_gate_fno_w36_l3_neuralop_10000_b8_s1234_bkw" / "best.pt"


def parse_vector(text: str) -> tuple[float, float, float]:
    return tuple(float(value) for value in text.split(","))  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve steady 1D-x/3D-v Boltzmann with neural/spectral or pure BGK collision.")
    parser.add_argument("--case", choices=("couette", "fourier", "custom"), default="couette")
    parser.add_argument("--left-u", type=parse_vector, default=None, metavar="UX,UY,UZ")
    parser.add_argument("--right-u", type=parse_vector, default=None, metavar="UX,UY,UZ")
    parser.add_argument("--left-temperature", type=float, default=None)
    parser.add_argument("--right-temperature", type=float, default=None)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--nv", type=int, default=32, help="velocity points per dimension")
    parser.add_argument("--velocity-L", type=float, default=10.0,
                        help="BGK velocity half-width; neural runs use the checkpoint value")
    parser.add_argument("--x-left", type=float, default=-0.5)
    parser.add_argument("--x-right", type=float, default=0.5)
    parser.add_argument("--kn", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--collision-backend", choices=("neural", "bgk"), default="neural")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-batch-chunk", type=int, default=32)
    parser.add_argument("--outer-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--final-outer-tolerance",
        type=float,
        default=None,
        help="exact-refinement outer tolerance; defaults to --outer-tolerance",
    )
    parser.add_argument("--inner-tolerance", type=float, default=DEFAULT_GATE_INNER_RELATIVE_TOLERANCE)
    parser.add_argument("--max-outer-iterations", type=int, default=50)
    parser.add_argument("--max-inner-iterations", type=int, default=MAX_INNER_ITERATIONS)
    parser.add_argument("--mu-scale", type=float, default=1.0)
    parser.add_argument("--fixed-mu", type=float, default=None)
    parser.add_argument("--require-inner-convergence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preserve-total-mass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--concurrent-direction-sweeps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spectral-switch-relative", type=float, default=1.0e-2)
    parser.add_argument("--spectral-switch-outer-iteration", type=int, default=10)
    parser.add_argument("--R", type=float, default=9.0)
    parser.add_argument("--spectral-angular-chunk", type=int, default=8)
    parser.add_argument("--spectral-batch-chunk", type=int, default=1)
    parser.add_argument("--final-spectral-refinement", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--final-spectral-angular-accuracy", type=int, default=13)
    parser.add_argument("--final-inner-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--final-max-outer-iterations", type=int, default=30)
    parser.add_argument("--final-max-inner-iterations", type=int, default=MAX_INNER_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.collision_backend == "bgk" and args.final_spectral_refinement:
        parser.error("pure BGK cannot be combined with --final-spectral-refinement")
    if args.velocity_L <= 0.0:
        parser.error("--velocity-L must be positive")
    case = with_wall_overrides(case_preset(args.case), left_velocity=args.left_u, right_velocity=args.right_u,
                               left_temperature=args.left_temperature, right_temperature=args.right_temperature)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = PACKAGE_DIR / "outputs" / f"{args.case}_nx{args.nx}_nv{args.nv}_kn{str(args.kn).replace('.', 'p')}_{stamp}"
    output_dir = (args.output_dir or default_output).resolve()
    prepare_output(output_dir, overwrite=args.overwrite)

    if args.collision_backend == "neural":
        direction_collision = GateFNOCollision(
            args.checkpoint,
            device=args.device,
            batch_chunk=args.gate_batch_chunk,
            n=args.nv,
        )
    else:
        direction_collision = BGKCollision(
            n=args.nv,
            L=args.velocity_L,
            device=args.device,
        )
    grid = make_velocity_grid(
        n=direction_collision.n,
        L=direction_collision.L,
        device=args.device,
    )
    initial = make_initial_maxwellian(grid, nx=args.nx, left_wall=case.left_wall,
                                     right_wall=case.right_wall, density=1.0)

    final_spectral_config = SpectralConfig(
        n=direction_collision.n,
        L=direction_collision.L,
        R=args.R,
        angular_accuracy=args.final_spectral_angular_accuracy,
        angular_chunk=args.spectral_angular_chunk,
        batch_chunk=args.spectral_batch_chunk,
    )
    final_nonlinear_collision = None
    final_linearized_collision = None
    linear_collision_config = None
    if args.final_spectral_refinement:
        final_nonlinear_collision = LazySpectralCollision(final_spectral_config)
        linear_collision_config = LinearCollisionConfig(
            n=direction_collision.n,
            L=direction_collision.L,
            R=args.R,
        )
        final_linearized_collision = LazyLocalMaxwellianLinearCollision(linear_collision_config)

    final_outer_tolerance = (
        args.outer_tolerance
        if args.final_outer_tolerance is None
        else args.final_outer_tolerance
    )

    solver_config = Solver1DConfig(
        nx=args.nx,
        x_left=args.x_left,
        x_right=args.x_right,
        kn=args.kn,
        max_outer_iterations=args.max_outer_iterations,
        max_inner_iterations=args.max_inner_iterations,
        max_final_refinement_outer_iterations=args.final_max_outer_iterations,
        max_final_refinement_inner_iterations=args.final_max_inner_iterations,
        outer_tolerance=args.outer_tolerance,
        final_refinement_outer_tolerance=final_outer_tolerance,
        inner_relative_tolerance=args.inner_tolerance,
        final_refinement_inner_relative_tolerance=args.final_inner_tolerance,
        mu_scale=args.mu_scale,
        fixed_mu=args.fixed_mu,
        preserve_total_mass=args.preserve_total_mass,
        require_inner_convergence=args.require_inner_convergence,
        concurrent_direction_sweeps=args.concurrent_direction_sweeps,
        spectral_switch_relative_residual=args.spectral_switch_relative,
        spectral_switch_outer_iteration=args.spectral_switch_outer_iteration,
        verbose=not args.quiet,
    )
    manifest = {
        "format": "steady-boltzmann-1d-v10",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": asdict(case),
        "solver": asdict(solver_config),
        "collision_backend": args.collision_backend,
        "solver_kind": (
            "newton_h" if args.collision_backend == "neural" else "newton_h_polarized_bgk"
        ),
        "direction_collision": asdict(direction_collision.metadata),
        "spectral": None,
        "final_refinement": {
            "enabled": args.final_spectral_refinement,
            "exact_nonlinear_spectral": asdict(final_spectral_config) if args.final_spectral_refinement else None,
            "direction_collision": (
                "local_maxwellian_quasi_newton"
                if args.collision_backend == "neural"
                else None
            ),
            "rhs_formula": (
                "J_M(h-f)=-F_exact(f)" if args.collision_backend == "neural" else None
            ),
            "local_maxwellian_linear_collision": asdict(linear_collision_config) if linear_collision_config else None,
            "inner_relative_tolerance": args.final_inner_tolerance,
            "outer_tolerance": final_outer_tolerance,
            "max_outer_iterations": args.final_max_outer_iterations,
            "max_inner_iterations": args.final_max_inner_iterations,
            "post_refinement_residual_check": args.collision_backend == "neural",
        },
        "state_dtype": "float64",
        "neural_dtype": "float32" if args.collision_backend == "neural" else None,
        "velocity_grid": {"n": direction_collision.n, "L": direction_collision.L},
        "command": sys.argv if argv is None else [sys.argv[0], *argv],
    }
    write_json(output_dir / "run_manifest.json", manifest)

    for warmup_iteration in range(args.warmup):
        direction_collision.evaluate(
            initial[:min(args.gate_batch_chunk, args.nx)],
            collect_diagnostics=warmup_iteration == 0,
        )
    if grid.v.is_cuda:
        torch.cuda.synchronize(grid.v.device)
    start = time.perf_counter()
    solver = SteadyBoltzmann1D(
        grid=grid,
        walls=(case.left_wall, case.right_wall),
        direction_collision=direction_collision,
        final_nonlinear_collision=final_nonlinear_collision,
        final_linearized_collision=final_linearized_collision,
        config=solver_config,
        direction_phase="gate" if args.collision_backend == "neural" else "bgk",
        direction_residual_backend="gate_fno" if args.collision_backend == "neural" else "bgk",
    )
    try:
        result = solver.solve(initial)
        if grid.v.is_cuda:
            torch.cuda.synchronize(grid.v.device)
        elapsed = time.perf_counter() - start
        profiles = macroscopic_profiles(result.state, grid)
        x = args.x_left + (torch.arange(args.nx, dtype=torch.float64) + 0.5) * solver_config.dx
        profile_arrays = {key: value.detach().cpu().numpy() for key, value in profiles.items()}
        arrays = {"x": x.numpy(), "v": grid.v.detach().cpu().numpy(),
                  "f": result.state.detach().cpu().numpy(), **profile_arrays}
        write_npz(output_dir / "solution.npz", arrays)
        write_profiles_csv(output_dir / "profiles.csv", arrays["x"], profile_arrays)
        write_json(output_dir / "history.json", result.history)
        gate_rows = [row for row in result.history if row.get("phase") == "gate"]
        bgk_rows = [row for row in result.history if row.get("phase") == "bgk"]
        final_row = result.history[-1] if result.history else None
        gate_trigger_row = gate_rows[-1] if result.final_refinement_applied else None
        summary = {
            "converged": result.converged,
            "collision_backend": args.collision_backend,
            "outer_iterations_evaluated": len(result.history),
            "direction_outer_iterations_evaluated": len(gate_rows) + len(bgk_rows),
            "gate_outer_iterations_evaluated": len(gate_rows),
            "bgk_outer_iterations_evaluated": len(bgk_rows),
            "used_spectral_residual": result.used_spectral_residual,
            "elapsed_seconds": elapsed,
            "target_total_mass": result.target_total_mass,
            "final_total_mass": float(solver_config.dx * grid.dv3 * torch.sum(result.state)),
            "minimum_f": final_row["minimum_f"] if final_row else None,
            "final_residual": final_row["weighted_residual"] if final_row else None,
            "gate_trigger_residual": gate_trigger_row["weighted_residual"] if gate_trigger_row else None,
            "final_residual_backend": final_row["residual_backend"] if final_row else None,
            "gate_trigger_residual_backend": gate_trigger_row["residual_backend"] if gate_trigger_row else None,
            "direction_collision_last_diagnostics": direction_collision.last_diagnostics,
            "gate_last_diagnostics": (
                direction_collision.last_diagnostics
                if args.collision_backend == "neural"
                else None
            ),
            "final_refinement_enabled": args.final_spectral_refinement,
            "final_refinement_applied": result.final_refinement_applied,
            "final_refinement_inner_converged": result.final_refinement_inner_converged,
            "final_refinement_converged": result.final_refinement_converged,
            "final_refinement_outer_iterations_evaluated": result.final_refinement_outer_iterations_evaluated,
            "final_refinement_updates": result.final_refinement_updates,
            "returned_state_residual_evaluated": bool(final_row and final_row.get("returned_state_residual_evaluated")),
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(json_ready(summary), indent=2), flush=True)
        print(f"outputs: {output_dir}", flush=True)
        return 0 if result.converged else 2
    finally:
        direction_collision.close()
        for collision in filter(None, (final_nonlinear_collision, final_linearized_collision)):
            collision.close()


if __name__ == "__main__":
    raise SystemExit(main())
