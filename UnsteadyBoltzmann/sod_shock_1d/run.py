"""One-off 1D-x/3D-v kinetic Sod/normal-shock computation.

The MUSCL finite-volume transport and collision terms form one unsplit
method-of-lines right-hand side. Half-range Maxwellian inflow is imposed only
on incoming velocities. Euler is the default integrator; SSPRK2 is optional.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import torch


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UnsteadyBoltzmann.collision import (
    collision_batch_chunk,
    collision_signature,
    make_unsteady_collision,
)
from UnsteadyBoltzmann.runtime import filename_float, guard_output_directory, run_fixed_time

DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / "boltzmann_tail_simplified_gate_fno_w36_l3_neuralop_10000_b8_s1234_bkw" / "best.pt"


def parameterized_output_directory(args: argparse.Namespace) -> Path:
    dx = 70.0 / float(args.nx)
    cfl_dt = 0.5 * dx / float(args.velocity_half_width)
    effective_dt = cfl_dt if args.dt is None else min(float(args.dt), cfl_dt)
    parts = [
        f"nx{args.nx}",
        f"nv{args.nv}",
        f"kn{filename_float(args.kn)}",
        f"tend{filename_float(args.t_end)}",
        f"dt{filename_float(effective_dt)}",
        f"L{filename_float(args.velocity_half_width)}",
        f"Ma{filename_float(args.mach)}",
        f"gam{filename_float(args.gamma)}",
        f"rhom{filename_float(args.rho_minus)}",
        f"Tm{filename_float(args.temperature_minus)}",
        "Tstd",
        args.collision_backend,
        args.time_integrator,
    ]
    if args.collision_backend == "spectral":
        parts.extend((f"R{filename_float(args.spectral_R)}", f"ra{args.nv}", f"aa{args.spectral_angular_accuracy}"))
    return args.output_dir / "_".join(parts)


def _load_checkpoint_payload(path: Path, *, mmap: bool = False) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    except TypeError:
        return torch.load(path, map_location="cpu")


def latest_compatible_resume_checkpoint(
    checkpoint_root: Path,
    *,
    target_checkpoint: Path,
    t_end: float,
    nominal_dt: float,
    signature: dict[str, Any],
) -> tuple[Path, float] | None:
    """Find the furthest compatible saved state not beyond ``t_end``."""

    candidates = set(checkpoint_root.glob("*/checkpoint_latest.pt"))
    if target_checkpoint.is_file():
        candidates.add(target_checkpoint)
    tolerance = 1.0e-12 * max(1.0, abs(t_end))
    best: tuple[tuple[float, bool, int], Path, float] | None = None
    for path in candidates:
        try:
            payload = _load_checkpoint_payload(path, mmap=True)
            if (
                not isinstance(payload, dict)
                or payload.get("format_version") != 1
                or payload.get("case") != "sod_shock_1d"
                or payload.get("signature") != signature
            ):
                continue
            saved_dt = float(payload.get("nominal_dt", math.nan))
            time_value = float(payload.get("time", math.nan))
            completed_steps = payload.get("completed_steps")
            state = payload.get("state")
            if not math.isclose(saved_dt, nominal_dt, rel_tol=1.0e-14, abs_tol=0.0):
                continue
            if (
                not math.isfinite(time_value)
                or time_value < 0.0
                or time_value > t_end + tolerance
                or not isinstance(completed_steps, int)
                or completed_steps < 0
                or not isinstance(state, torch.Tensor)
            ):
                continue
            modified_ns = path.stat().st_mtime_ns
        except Exception:
            # Discovery must not let an unrelated or incomplete checkpoint
            # prevent a valid older state from being resumed.
            continue
        key = (min(time_value, t_end), path == target_checkpoint, modified_ns)
        if best is None or key > best[0]:
            best = (key, path, min(time_value, t_end))
    if best is None:
        return None
    return best[1], best[2]


class SodShock1D:
    """Case-specific finite-volume solver for the prescribed shock states."""

    def __init__(
        self,
        collision: Any,
        *,
        nx: int = 100,
        nv: int = 64,
        velocity_half_width: float = 10.0,
        kn: float = 1.0,
        gamma: float = 5.0 / 3.0,
        mach: float = 2.0,
        rho_minus: float = 1.0,
        temperature_minus: float = 1.0,
        time_integrator: str = "euler",
        device: str | torch.device = "cuda",
    ) -> None:
        self.collision = collision
        self.nx = int(nx)
        self.nv = int(nv)
        self.velocity_half_width = float(velocity_half_width)
        self.kn = float(kn)
        self.gamma = float(gamma)
        self.mach = float(mach)
        self.rho_minus = float(rho_minus)
        self.temperature_minus = float(temperature_minus)
        self.time_integrator = time_integrator
        requested_device = torch.device(device)

        self.x_left = -35.0
        self.x_right = 35.0
        self.dx = (self.x_right - self.x_left) / self.nx
        self.x = self.x_left + (torch.arange(self.nx, dtype=torch.float64, device=requested_device) + 0.5) * self.dx
        self.device = self.x.device

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

        self.dv = 2.0 * self.velocity_half_width / self.nv
        self.dv3 = self.dv**3
        self.v = (torch.arange(self.nv, dtype=torch.float64, device=self.device) - self.nv // 2) * self.dv
        self.vx, self.vy, self.vz = torch.meshgrid(self.v, self.v, self.v, indexing="ij")
        self.v2 = self.vx.square() + self.vy.square() + self.vz.square()
        self.positive_vx = self.vx.clamp_min(0.0)
        self.negative_vx = self.vx.clamp_max(0.0)
        self.maxwellian_minus = self._maxwellian(self.rho_minus, self.u_minus, self.temperature_minus)
        self.maxwellian_plus = self._maxwellian(self.rho_plus, self.u_plus, self.temperature_plus)

        self.transport_cfl = 0.5
        self.maximum_transport_speed = self.velocity_half_width
        self.cfl_time_step = self.transport_cfl * self.dx / self.maximum_transport_speed
        self.last_nominal_dt: float | None = None
        self.last_number_of_steps = 0

    def _maxwellian(
        self, density: float, velocity_x: float, temperature: float
    ) -> torch.Tensor:
        relative_speed_squared = (self.vx - velocity_x).square() + self.vy.square() + self.vz.square()
        return density * torch.exp(-relative_speed_squared / (2.0 * temperature)) / (2.0 * math.pi * temperature) ** 1.5

    def initial_state(self) -> torch.Tensor:
        """Sample the two analytic Maxwellians without discrete normalization."""

        return torch.where((self.x < 0.0)[:, None, None, None], self.maxwellian_minus[None], self.maxwellian_plus[None])

    def moments(self, state: torch.Tensor) -> torch.Tensor:
        """Return [rho, ux, uy, uz, T] with p=rho*T."""

        rho = self.dv3 * state.sum(dim=(-3, -2, -1))
        momentum = torch.stack(tuple(
            self.dv3 * torch.sum(state * component[None], dim=(-3, -2, -1))
            for component in (self.vx, self.vy, self.vz)), dim=-1)
        velocity = momentum / rho[:, None]
        energy = self.dv3 * torch.sum(state * self.v2[None], dim=(-3, -2, -1))
        temperature = (energy / rho - velocity.square().sum(dim=-1)) / 3.0
        return torch.cat((rho[:, None], velocity, temperature[:, None]), dim=1)

    @staticmethod
    def _minmod(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        same_sign = left * right > 0.0
        limited = torch.minimum(left.abs(), right.abs())
        return torch.where(same_sign, torch.sign(left) * limited, 0.0)

    def transport_interface_fluxes(self, state: torch.Tensor) -> torch.Tensor:
        """Return all Nx+1 MUSCL fluxes with half-range inflow boundaries."""

        left_ghost = torch.where(self.vx > 0.0, self.maxwellian_minus, state[0])
        right_ghost = torch.where(self.vx < 0.0, self.maxwellian_plus, state[-1])
        backward_neighbor = torch.cat((left_ghost[None], state[:-1]), dim=0)
        forward_neighbor = torch.cat((state[1:], right_ghost[None]), dim=0)
        slope = self._minmod(state - backward_neighbor, forward_neighbor - state)

        left_trace = state[:-1] + 0.5 * slope[:-1]
        right_trace = state[1:] - 0.5 * slope[1:]
        interior_flux = self.positive_vx[None] * left_trace + self.negative_vx[None] * right_trace

        left_boundary_flux = self.positive_vx * self.maxwellian_minus + self.negative_vx * (state[0] - 0.5 * slope[0])
        right_boundary_flux = self.positive_vx * (state[-1] + 0.5 * slope[-1]) + self.negative_vx * self.maxwellian_plus
        return torch.cat((left_boundary_flux[None], interior_flux, right_boundary_flux[None]), dim=0)

    def transport_rhs(self, state: torch.Tensor) -> torch.Tensor:
        flux = self.transport_interface_fluxes(state)
        return -(flux[1:] - flux[:-1]) / self.dx

    def rhs(self, state: torch.Tensor) -> torch.Tensor:
        """Return the unsplit method-of-lines transport-plus-collision RHS."""
        collision = self.collision.evaluate(state)
        return self.transport_rhs(state) + collision / self.kn

    def advance(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        first = state + dt * self.rhs(state)
        if self.time_integrator == "euler":
            return first
        return 0.5 * state + 0.5 * (first + dt * self.rhs(first))

    def run(
        self,
        *,
        t_end: float = 1.0,
        dt: float | None = None,
        progress: bool = True,
        checkpoint_path: Path | None = None,
        checkpoint_interval_seconds: float = 1800.0,
        resume: bool = False,
        resume_from_path: Path | None = None,
        checkpoint_signature: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        nominal_dt = self.cfl_time_step if dt is None else min(dt, self.cfl_time_step)
        self.last_nominal_dt = nominal_dt
        resume_path = (resume_from_path or checkpoint_path) if resume else None
        state, self.last_number_of_steps = run_fixed_time(
            initial_state=self.initial_state,
            advance=self.advance,
            t_end=t_end,
            nominal_dt=nominal_dt,
            device=self.device,
            case="sod_shock_1d",
            progress=progress,
            checkpoint_path=checkpoint_path,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
            resume_path=resume_path,
            signature=checkpoint_signature or {},
        )
        return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="1D kinetic normal shock")
    parser.add_argument("--Nx", "--nx", dest="nx", type=int, default=100)
    parser.add_argument("--Nv", "--nv", dest="nv", type=int, default=64)
    parser.add_argument("--T", "--t-end", "--final-time", dest="t_end", type=float, default=1.0)
    parser.add_argument("--Kn", "--kn", dest="kn", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--Ma", "--ma", "--mach", dest="mach", type=float, default=2.0)
    parser.add_argument("--rho-minus", type=float, default=1.0)
    parser.add_argument("--temperature-minus", type=float, default=1.0)
    parser.add_argument("--velocity-half-width", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--time-integrator", choices=("euler", "ssprk2"), default="euler")
    parser.add_argument("--collision-backend", choices=("gate", "spectral"), default="gate")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gate-batch-chunk", type=int, default=32)
    parser.add_argument("--spectral-R", type=float, default=9.0)
    parser.add_argument("--spectral-angular-accuracy", type=int, default=13)
    parser.add_argument("--spectral-angular-chunk", type=int, default=37)
    parser.add_argument("--spectral-batch-chunk", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


from UnsteadyBoltzmann.sod_shock_1d.postprocess import save_results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = parameterized_output_directory(args)
    guard_output_directory(output_dir, overwrite=args.overwrite, resume=args.resume)

    batch_chunk = collision_batch_chunk(args)
    collision = make_unsteady_collision(args, batch_chunk)
    solver = SodShock1D(
        collision,
        nx=args.nx,
        nv=args.nv,
        velocity_half_width=args.velocity_half_width,
        kn=args.kn,
        gamma=args.gamma,
        mach=args.mach,
        rho_minus=args.rho_minus,
        temperature_minus=args.temperature_minus,
        time_integrator=args.time_integrator,
        device=args.device,
    )
    nominal_dt = solver.cfl_time_step if args.dt is None else min(float(args.dt), solver.cfl_time_step)
    resume_checkpoint = output_dir / "checkpoint_latest.pt"
    checkpoint_signature = {
        "case": "sod_shock_1d",
        "Nx": args.nx,
        "Nv": args.nv,
        "Kn": args.kn,
        "gamma": args.gamma,
        "Mach": args.mach,
        "rho_minus": args.rho_minus,
        "temperature_minus": args.temperature_minus,
        "temperature_convention": "standard",
        "velocity_half_width": args.velocity_half_width,
        "time_integrator": args.time_integrator,
        **collision_signature(args),
    }
    resume_source: Path | None = None
    resume_source_time: float | None = None
    if args.resume:
        selected = latest_compatible_resume_checkpoint(
            args.output_dir, target_checkpoint=resume_checkpoint, t_end=args.t_end,
            nominal_dt=nominal_dt, signature=checkpoint_signature)
        if selected is None:
            raise FileNotFoundError(
                "no compatible SOD checkpoint exists at or below "
                f"T={args.t_end:.10g} under {args.output_dir}"
            )
        resume_source, resume_source_time = selected
    if not args.quiet:
        print(f"backend={args.collision_backend}, time integrator={args.time_integrator}, unsplit=True", flush=True)
        print(f"dt={nominal_dt:.10g}, dt/Kn={nominal_dt / args.kn:.6g}, collision x-batch={batch_chunk}", flush=True)
        states = (f"minus=({solver.rho_minus:.8g},{solver.u_minus:.8g},{solver.temperature_minus:.8g}), "
                  f"plus=({solver.rho_plus:.8g},{solver.u_plus:.8g},{solver.temperature_plus:.8g})")
        print(f"states: {states}", flush=True)
        print(f"result directory: {output_dir}", flush=True)
        print(f"rolling checkpoint: {resume_checkpoint} (every {args.checkpoint_minutes:g} wall-clock minutes)", flush=True)
        if resume_source is not None:
            print(f"resume source: {resume_source} (saved t={resume_source_time:.10g})", flush=True)
    if nominal_dt / args.kn > 1.0:
        print(f"warning: explicit collision has dt/Kn={nominal_dt / args.kn:.6g}; this may be unstable",
              file=sys.stderr, flush=True)

    final_state = solver.run(
        t_end=args.t_end,
        dt=args.dt,
        progress=not args.quiet,
        checkpoint_path=resume_checkpoint,
        checkpoint_interval_seconds=60.0 * args.checkpoint_minutes,
        resume=args.resume,
        resume_from_path=resume_source,
        checkpoint_signature=checkpoint_signature,
    )
    save_results(
        output_dir,
        solver=solver,
        final_state=final_state,
        run_configuration={
            **vars(args),
            "Nx": args.nx,
            "Nv": args.nv,
            "T_end": args.t_end,
            "Kn": args.kn,
            "gamma": args.gamma,
            "Mach": args.mach,
            "dt": solver.last_nominal_dt,
            "number_of_steps": solver.last_number_of_steps,
            "collision_x_batch_chunk": batch_chunk,
            "spectral_radial_accuracy": args.nv,
            "resume_checkpoint": resume_checkpoint,
            "resume_source_checkpoint": resume_source,
            "resume_source_time": resume_source_time,
        },
    )
    print(f"saved final arrays and three profile PDFs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
