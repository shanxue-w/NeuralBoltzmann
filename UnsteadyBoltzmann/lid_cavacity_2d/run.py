"""Stretched-grid finite-volume cavity: transport/2, collision, transport/2.

Transport uses conservative semi-Lagrangian CWENO3 remapping, with a
mean-preserving Bernstein positivity limiter. Its directional splitting is
x/2, y, x/2. Diffuse-wall inflow is integrated from the evolving outgoing
flux, rather than frozen at the start of a sweep. A sweep longer than one
box crossing is subdivided; there is no restriction involving the smallest
spatial cell. Collision acts directly on cell averages (SSPRK2 by default).

CWENO3 describes the reconstruction order, not a proved third-order cavity
solver: nonlinear cell-average collision, splitting and wall/corner errors
must be included in convergence studies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UnsteadyBoltzmann.collision import collision_batch_chunk, collision_signature, make_unsteady_collision
from UnsteadyBoltzmann.lid_cavacity_2d._uniform_solver import (
    LidDrivenCavity2D as UniformCavity,
    build_parser as uniform_parser,
)
from UnsteadyBoltzmann.runtime import filename_float, guard_output_directory, run_fixed_time
from UnsteadyBoltzmann.lid_cavacity_2d.snapshots import SECOND_COMPONENTS, fields_from_raw_moments, save_moment_snapshot


def bernstein_integrals(edges, left, right):
    """Integrals of the three quadratic Bernstein bases over clipped intervals.

    Two-point Gauss quadrature is exact and has positive weights. Returning
    nonnegative weights avoids cancellation in cumulative-integral differences.
    Shape: [number of intervals, number of source cells, 3].
    """
    left, right = np.atleast_1d(left), np.atleast_1d(right)
    lo = np.maximum(left[:, None], edges[:-1])
    hi = np.minimum(right[:, None], edges[1:])
    length = np.maximum(hi - lo, 0.0)
    midpoint = (lo + hi) / 2.0
    result = np.zeros((*length.shape, 3))
    for sign in (-1.0, 1.0):
        z = ((midpoint + sign * length / math.sqrt(12.0) - edges[:-1]) / np.diff(edges)).clip(0.0, 1.0)
        result += 0.5 * length[..., None] * np.stack(((1.0 - z)**2, 2.0*z*(1.0-z), z**2), axis=-1)
    return result


class LidDrivenCavity2D(UniformCavity):
    """Reuse the original velocity grid, Maxwellians, moments and initial state."""

    def __init__(self, collision, *, nx=32, ny=48, nv=32, stretch=2.0,
                 velocity_half_width=10.0, kn=0.1, wall_temperature=1.0,
                 lid_speed=0.15, time_integrator="ssprk2", device="cuda"):
        if nx < 3 or ny < 3 or nv < 4 or nv % 2:
            raise ValueError("Nx, Ny must be >= 3; Nv must be even and >= 4")
        if not all(math.isfinite(value) and value > 0 for value in (kn, wall_temperature, velocity_half_width)):
            raise ValueError("Kn, wall temperature and velocity half width must be finite and positive")
        if not math.isfinite(stretch) or not 0 <= stretch <= 10 or not math.isfinite(lid_speed):
            raise ValueError("stretch must be in [0,10] and lid speed must be finite")
        if time_integrator not in ("euler", "ssprk2"):
            raise ValueError("collision integrator must be euler or ssprk2")
        super().__init__(collision, nx=nx, ny=ny, nv=nv, kn=kn,
                         velocity_half_width=velocity_half_width, wall_temperature=wall_temperature,
                         lid_speed=lid_speed, time_integrator=time_integrator, device=device)
        self.stretch = float(stretch)
        s = np.linspace(0.0, 1.0, ny + 1)
        y_edges = s if stretch == 0 else np.expm1(-stretch*s) / np.expm1(-stretch)
        self.edges = (np.linspace(0.0, 1.0, nx + 1), y_edges)
        self.x = torch.as_tensor((self.edges[0][:-1] + self.edges[0][1:])/2, device=self.device)
        self.y = torch.as_tensor((y_edges[:-1] + y_edges[1:])/2, device=self.device)
        self.dy = torch.as_tensor(np.diff(y_edges), device=self.device)
        self.cell_area = self.dx * self.dy[None, :]
        self.cfl_time_step = 0.5 / (velocity_half_width/self.dx + velocity_half_width/float(self.dy.min()))
        self._geometry = [self._reconstruction_geometry(edges) for edges in self.edges]
        self._plans = {}  # At most one current duration per axis.

    def _reconstruction_geometry(self, edges):
        width = np.diff(edges)
        center = (edges[1:] + edges[:-1])/2
        n = len(width)
        neighbors = np.column_stack((np.arange(n)-1, np.arange(n)+1))
        neighbors[0], neighbors[-1] = [1, 2], [n-2, n-3]
        offset = (center[neighbors] - center[:, None]) / width[:, None]
        second = offset**2 + (width[neighbors]/width[:, None])**2/12 - 1/12
        inverse = np.linalg.inv(np.stack((offset, second), axis=-1))
        return (
            torch.as_tensor(neighbors, device=self.device),
            torch.as_tensor(offset, device=self.device),
            torch.as_tensor(inverse, device=self.device),
            torch.as_tensor(width**2, device=self.device),
        )

    def reconstruct(self, values, axis):
        """CWENO3 on one spatial line, returned as nonnegative Bernstein coefficients.

        values: [cell, normal velocity, two tangential velocities]. Every
        candidate has exactly the supplied cell average. One-sided stencils
        are used at both walls. Scaling all Bernstein coefficients about the
        average preserves their mean and ensures positivity everywhere.
        """
        neighbors, offset, inverse, epsilon = self._geometry[axis]
        shape = (-1, 1, 1, 1)
        left, right = values[neighbors[:, 0]] - values, values[neighbors[:, 1]] - values
        a = inverse[:, 0, 0].reshape(shape)*left + inverse[:, 0, 1].reshape(shape)*right
        b = inverse[:, 1, 0].reshape(shape)*left + inverse[:, 1, 1].reshape(shape)*right
        a_left = left / offset[:, 0].reshape(shape)
        a_right = right / offset[:, 1].reshape(shape)
        a_center, b_center = 2*a - 0.5*(a_left+a_right), 2*b
        scale = torch.maximum(values.abs(), torch.maximum((values+left).abs(), (values+right).abs()))
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        eps = epsilon.reshape(shape)
        w0 = 0.5 / (eps + (a_center/scale)**2 + (13/3)*(b_center/scale)**2)**2
        wl = 0.25 / (eps + (a_left/scale)**2)**2
        wr = 0.25 / (eps + (a_right/scale)**2)**2
        total = w0 + wl + wr
        a = (w0*a_center + wl*a_left + wr*a_right)/total
        b = w0*b_center/total
        coefficients = torch.stack((values - a/2 + b/6, values - b/3, values + a/2 + b/6), dim=2)
        minimum = coefficients.amin(dim=2)
        theta = torch.where(minimum < 0, values/(values-minimum).clamp_min(torch.finfo(values.dtype).tiny), 1.0)
        theta = torch.where(minimum < 0, theta*(1-8*torch.finfo(values.dtype).eps), theta)
        return values[:, :, None] + theta[:, :, None]*(coefficients-values[:, :, None])

    def _plan(self, axis, duration):
        """Positive remap matrices, including exact time-integrated wall injection."""
        if axis in self._plans and self._plans[axis][0] == duration:
            return self._plans[axis][1]
        edges = self.edges[axis]
        width, n = np.diff(edges), len(edges)-1
        velocity = self.v.detach().cpu().numpy()
        lo = edges[:-1][None, :] - velocity[:, None]*duration
        hi = edges[1:][None, :] - velocity[:, None]*duration
        interior = bernstein_integrals(edges, lo.ravel(), hi.ravel()).reshape(self.nv, n, n, 3)
        interior /= width[None, :, None, None]
        interior = torch.as_tensor(interior.reshape(self.nv, n, 3*n), device=self.device)
        walls = []
        for side in (0, 1):
            maxwellian = self.top_wall_unit if axis == 1 and side == 1 else self.stationary_wall_unit
            if axis == 1:
                maxwellian = maxwellian.transpose(0, 1)
            incoming = velocity > 0 if side == 0 else velocity < 0
            denominator = self.dv3 * float((self.v.abs()[:, None, None]*maxwellian*torch.as_tensor(incoming, device=self.device)[:, None, None]).sum())
            if denominator <= 0 or not math.isfinite(denominator):
                raise ValueError("velocity grid does not resolve the diffuse wall Maxwellian")
            near, far = (edges[:-1], edges[1:]) if side == 0 else (1-edges[1:], 1-edges[:-1])
            queries = []
            for q in np.flatnonzero(incoming):
                # Emission times whose reflected particles end in each target cell.
                t_lo = np.maximum(0.0, duration-far/abs(velocity[q]))
                t_hi = np.minimum(duration, duration-near/abs(velocity[q]))
                queries.extend((q, j, t_lo[j], t_hi[j]) for j in np.flatnonzero(t_hi > t_lo))
            q_index, cells, t_lo, t_hi = np.array(queries).T
            q_index, cells = q_index.astype(np.int64), cells.astype(np.int64)
            kernel = np.zeros((len(queries), self.nv, n, 3))
            outgoing = velocity < 0 if side == 0 else velocity > 0
            for q in np.flatnonzero(outgoing):
                # Outgoing flux integrated over [t_lo,t_hi] is a swept spatial
                # integral. It depends only on initial data for <= one crossing.
                a, b = abs(velocity[q])*t_lo, abs(velocity[q])*t_hi
                if side == 1:
                    a, b = 1-b, 1-a
                kernel[:, q] = bernstein_integrals(edges, a, b)
            factor = abs(velocity[q_index]) / (width[cells]*denominator)
            walls.append((
                torch.as_tensor(q_index, device=self.device),
                torch.as_tensor(cells, device=self.device),
                torch.as_tensor(kernel.reshape(len(queries), -1), device=self.device),
                torch.as_tensor(factor, device=self.device),
                maxwellian.reshape(self.nv, -1),
            ))
        plan = interior, walls
        self._plans[axis] = duration, plan
        return plan

    def sweep(self, state, axis, duration):
        if duration == 0:
            return state
        # A whole-domain flight bound, independent of Nx, Ny or min(dy).
        count = max(1, math.ceil(duration*self.velocity_half_width))
        step = duration/count
        interior, walls = self._plan(axis, step)
        n = self.nx if axis == 0 else self.ny
        for _ in range(count):
            result = torch.empty_like(state)
            for line in range(self.ny if axis == 0 else self.nx):
                values = state[:, line] if axis == 0 else state[line].transpose(1, 2)
                coefficients = self.reconstruct(values, axis).permute(1, 0, 2, 3, 4).contiguous()
                transported = torch.bmm(interior, coefficients.reshape(self.nv, 3*n, -1))
                outgoing_moments = coefficients.sum(dim=(-2, -1)).reshape(-1)*self.dv3
                for q_index, cells, kernel, factor, maxwellian in walls:
                    emitted = (kernel @ outgoing_moments)*factor
                    transported[q_index, cells] += emitted[:, None]*maxwellian[q_index]
                values = transported.transpose(0, 1).reshape(n, self.nv, self.nv, self.nv)
                if axis == 0:
                    result[:, line] = values
                else:
                    result[line] = values.transpose(1, 2)
            state = result
        return state

    def transport(self, state, duration):
        state = self.sweep(state, 0, duration/2)
        state = self.sweep(state, 1, duration)
        return self.sweep(state, 0, duration/2)

    @staticmethod
    def _checked(state):
        if not bool(torch.isfinite(state).all()) or float(state.min()) < -1.0e-13:
            raise FloatingPointError("collision stage is nonfinite or negative; reduce --dt")
        return state.clamp_min(0.0)  # Only roundoff below zero is accepted.

    def advance(self, state, dt):
        state = self.transport(state, dt/2)
        first = self._checked(state + (dt/self.kn)*self.collision.evaluate(state.reshape(-1, self.nv, self.nv, self.nv)).reshape_as(state))
        if self.time_integrator == "ssprk2":
            q = self.collision.evaluate(first.reshape(-1, self.nv, self.nv, self.nv)).reshape_as(first)
            first = self._checked(0.5*state + 0.5*(first + (dt/self.kn)*q))
        return self.transport(first, dt/2)

    def snapshot_fields(self, state):
        if not hasattr(self, "_moment_basis"):
            velocities = (self.vx, self.vy, self.vz)
            basis = [torch.ones_like(self.vx), *velocities]
            basis.extend(velocities[i] * velocities[j] for i, j in SECOND_COMPONENTS)
            basis.extend(0.5 * self.v2 * velocity for velocity in velocities)
            self._moment_basis = torch.stack(basis, dim=-1).reshape(-1, 13)
        raw = ((state.reshape(-1, self.nv**3) @ self._moment_basis) * self.dv3
               ).reshape(self.nx, self.ny, 13).detach().cpu().numpy()
        return fields_from_raw_moments(raw)

    def run(self, *, t_end=1.0, dt=0.005, progress=True, checkpoint_path=None,
            checkpoint_interval_seconds=1800.0, checkpoint_time_interval=None,
            resume=False, checkpoint_signature=None, resume_from=None,
            snapshot_interval=0.5, snapshot_dir=None):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(t_end) or t_end < 0:
            raise ValueError("dt must be finite and positive; T must be finite and nonnegative")
        self.last_nominal_dt = dt
        if snapshot_dir is None and checkpoint_path is not None:
            snapshot_dir = Path(checkpoint_path).parent / "snapshots"
        signature = checkpoint_signature or {}

        def snapshot(state, time_value, completed_steps):
            fields = self.snapshot_fields(state)
            x, y = self.x.cpu().numpy(), self.y.cpu().numpy()
            metadata = dict(signature, case="lid_cavacity_2d_csl", T_end=time_value,
                            completed_steps=completed_steps, nominal_dt=dt,
                            snapshot_interval=snapshot_interval,
                            reference_title="Deterministic snapshot")
            save_moment_snapshot(snapshot_dir, time_value=time_value, x=x, y=y, **fields,
                                 x_faces=self.edges[0], y_faces=self.edges[1], metadata=metadata,
                                 v=self.v.cpu().numpy())

        state, self.last_number_of_steps = run_fixed_time(
            initial_state=self.initial_state, advance=self.advance, t_end=t_end,
            nominal_dt=dt, device=self.device, case="lid_cavacity_2d_csl",
            progress=progress, checkpoint_path=checkpoint_path,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
            checkpoint_time_interval=checkpoint_time_interval,
            resume_path=resume_from if resume_from is not None else checkpoint_path if resume else None,
            signature=signature,
            snapshot_interval=snapshot_interval if snapshot_dir is not None else None,
            snapshot_callback=snapshot if snapshot_dir is not None else None,
        )
        return state


def build_parser():
    parser = uniform_parser(output_dir=CASE_DIR / "output")
    parser.description = "Stretched finite-volume cavity: CSL transport/2, collision, CSL transport/2"
    parser.set_defaults(ny=48, dt=0.005, time_integrator="ssprk2")
    parser.add_argument("--stretch", type=float, default=2.0, help="y-grid exponential stretching; 0 is uniform")
    parser.add_argument("--snapshot-interval", type=float, default=0.5,
                        help="exact simulation-time spacing of snapshots; also save t=0 and T")
    parser.add_argument("--resume-from", type=Path, help="resume an explicit full checkpoint, including at a smaller T")
    return parser


def save_results(output, solver, state, configuration):
    output.mkdir(parents=True, exist_ok=True)
    x, y = solver.x.cpu().numpy(), solver.y.cpu().numpy()
    fields = solver.snapshot_fields(state)
    moments = fields["moments"]
    for name, array in dict(x=x, y=y, x_faces=solver.edges[0], y_faces=solver.edges[1],
                            final_moments=moments,
                            final_pressure_tensor=fields["pressure_tensor"], final_stress_tensor=fields["stress_tensor"],
                            final_heat_flux=fields["heat_flux"],
                            final_raw_moments=fields["raw_moments"],
                            final_density=moments[..., 0], final_velocity=moments[..., 1:4],
                            final_temperature=moments[..., 4]).items():
        np.save(output/f"{name}.npy", array)
    configuration.update(
        mass=float(np.sum(moments[..., 0]*np.diff(solver.edges[0])[:, None]*np.diff(solver.edges[1])[None, :])),
        initial_mass=float(solver.initial_maxwellian.sum())*solver.dv3,
        minimum_f=float(state.min()), moment_columns=["rho", "ux", "uy", "uz", "temperature"],
    )
    (output/"run_config.json").write_text(json.dumps(configuration, indent=2, default=str)+"\n")
    from UnsteadyBoltzmann.lid_cavacity_2d.postprocess import save_moment_plots
    streamfunction = save_moment_plots(output, x=x, y=y, moments=moments,
                                     time_value=configuration["T_end"], source_title="Deterministic terminal state")
    np.save(output/"final_streamfunction.npy", streamfunction)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dt <= 0 or not math.isfinite(args.dt) or args.t_end < 0 or not math.isfinite(args.t_end):
        raise ValueError("dt must be finite and positive; T must be finite and nonnegative")
    name = "_".join((f"nx{args.nx}", f"ny{args.ny}", f"nv{args.nv}",
                     f"kn{filename_float(args.kn)}", f"tend{filename_float(args.t_end)}",
                     f"dt{filename_float(args.dt)}", f"stretch{filename_float(args.stretch)}",
                     f"L{filename_float(args.velocity_half_width)}", f"Ulid{filename_float(args.lid_speed)}",
                     f"Tw{filename_float(args.wall_temperature)}", "csl3_tct", args.collision_backend, args.time_integrator))
    output = args.output_dir/name
    guard_output_directory(output, overwrite=args.overwrite, resume=args.resume or args.resume_from is not None)
    collision = make_unsteady_collision(args, collision_batch_chunk(args))
    try:
        solver = LidDrivenCavity2D(collision, nx=args.nx, ny=args.ny, nv=args.nv, stretch=args.stretch,
                                  kn=args.kn, velocity_half_width=args.velocity_half_width,
                                  wall_temperature=args.wall_temperature, lid_speed=args.lid_speed,
                                  time_integrator=args.time_integrator, device=args.device)
        signature = dict(case="lid_cavacity_2d_csl", Nx=args.nx, Ny=args.ny, Nv=args.nv, Kn=args.kn,
                         stretch=args.stretch, velocity_half_width=args.velocity_half_width,
                         wall_temperature=args.wall_temperature, lid_speed=args.lid_speed,
                         temperature_convention="standard", time_integrator=args.time_integrator,
                         splitting="transport/2-collision-transport/2; transport=x/2-y-x/2",
                         reconstruction="nonuniform CWENO3; positive Bernstein coefficients",
                         boundary="time-integrated directional Maxwell diffuse reflection v1",
                         **collision_signature(args))
        checkpoint = output/"checkpoint_latest.pt"
        if not args.quiet:
            print(f"grid={args.nx}x{args.ny}x{args.nv}^3, dy=[{float(solver.dy.min()):.6g}, {float(solver.dy.max()):.6g}]")
            print(f"dt={args.dt:g}; original explicit bound on this mesh={solver.cfl_time_step:.6g}; ratio={args.dt/solver.cfl_time_step:.3g}")
            print(f"Strang transport-collision-transport, collision={args.time_integrator}; output={output}")
            print(f"13-moment snapshots every {args.snapshot_interval:g}; directory={output/'snapshots'}")
        state = solver.run(t_end=args.t_end, dt=args.dt, progress=not args.quiet, checkpoint_path=checkpoint,
                           checkpoint_interval_seconds=60*args.checkpoint_minutes,
                           checkpoint_time_interval=args.checkpoint_time_interval, resume=args.resume,
                           checkpoint_signature=signature, resume_from=args.resume_from,
                           snapshot_interval=args.snapshot_interval, snapshot_dir=output/"snapshots")
        save_results(output, solver, state, dict(vars(args), **signature, T_end=args.t_end,
                                                number_of_steps=solver.last_number_of_steps))
        print(f"saved 13-moment snapshots, final fields and PDFs to {output}; full state in {checkpoint}")
    finally:
        collision.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
