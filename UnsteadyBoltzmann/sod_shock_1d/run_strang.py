"""Conservative semi-Lagrangian normal shock on a statically refined grid.

T(dt/2) C(dt) T(dt/2), with positive nonuniform CWENO3 cell reconstructions
and SSPRK2 collision substeps. No transport CFL restriction is imposed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from UnsteadyBoltzmann.sod_shock_1d.run import (
    SodShock1D, build_parser as base_parser, collision_batch_chunk,
    collision_signature, make_unsteady_collision, guard_output_directory,
    latest_compatible_resume_checkpoint, save_results, filename_float,
)
from UnsteadyBoltzmann.sod_shock_1d.spatial_grid import grid_edges, add_mesh_arguments


class StrangShock1D(SodShock1D):
    def __init__(self, collision, *, refine_factor=1, refine_center=0.0,
                 refine_width=10.0, collision_dt=None, reconstruction="cweno3",
                 mesh="sinh", stretch=5.0, **kwargs):
        kwargs["time_integrator"] = "ssprk2"
        super().__init__(collision, **kwargs)
        if self.nx < 2 or self.nv < 2 or self.kn <= 0:
            raise ValueError("nx and nv must be >= 2 and Kn must be positive")
        edges = grid_edges(self.nx, mesh=mesh, stretch=stretch, refine_factor=refine_factor,
                           refine_center=refine_center, refine_width=refine_width)
        self.edges = torch.tensor(edges, dtype=torch.float64, device=self.device)
        self.widths = self.edges.diff()
        self.x = (self.edges[1:] + self.edges[:-1]) / 2
        self.nx = len(self.x)
        if reconstruction not in ("cweno3", "minmod"):
            raise ValueError("reconstruction must be cweno3 or minmod")
        self.reconstruction = reconstruction
        if reconstruction == "cweno3" and self.nx >= 3:
            self._geometry = self._reconstruction_geometry()
        # The inherited run() uses this bound; infinity disables CFL clipping.
        self.cfl_time_step = math.inf
        self.collision_dt = collision_dt
        if collision_dt is not None and (not math.isfinite(collision_dt) or collision_dt <= 0):
            raise ValueError("collision-dt must be finite and positive")

    def initial_state(self):
        fraction = ((-self.edges[:-1]).clamp_min(0) / self.widths).clamp_max(1)
        return (fraction[:, None, None, None] * self.maxwellian_minus
                + (1 - fraction[:, None, None, None]) * self.maxwellian_plus)

    def slopes(self, state):
        """Physical-space slopes, limited on nonuniform cell centers."""
        slope = torch.zeros_like(state)
        gradient = (state[1:] - state[:-1]) / self.x.diff()[:, None, None, None]
        slope[1:-1] = self._minmod(gradient[:-1], gradient[1:])
        # Zero boundary slopes, and a positivity bound on both cell traces.
        limit = 2 * state.clamp_min(0) / self.widths[:, None, None, None]
        return slope.sign() * torch.minimum(slope.abs(), limit)

    def _reconstruction_geometry(self):
        """Cell-average quadratic fits as in lid_cavacity_2d/run.py."""
        edges = self.edges.cpu().numpy()
        width = np.diff(edges)
        center = (edges[1:] + edges[:-1]) / 2
        n = len(width)
        neighbors = np.column_stack((np.arange(n)-1, np.arange(n)+1))
        neighbors[0], neighbors[-1] = [1, 2], [n-2, n-3]
        offset = (center[neighbors] - center[:, None]) / width[:, None]
        second = offset**2 + (width[neighbors]/width[:, None])**2/12 - 1/12
        inverse = np.linalg.inv(np.stack((offset, second), axis=-1))
        return tuple(torch.as_tensor(v, device=self.device) for v in
                     (neighbors, offset, inverse, width**2))

    def reconstruct(self, values):
        """CWENO3 Bernstein coefficients [cell, vx, coefficient, vy, vz].

        Each candidate has the input cell average. One-sided stencils are
        used at the open boundaries. For nonnegative input averages, scaling
        about the mean makes the polynomial nonnegative without changing mass.
        """
        if self.nx < 3:
            return values[:, :, None].expand(-1, -1, 3, -1, -1)
        neighbors, offset, inverse, epsilon = self._geometry
        shape = (-1, 1, 1, 1)
        left = values[neighbors[:, 0]] - values
        right = values[neighbors[:, 1]] - values
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
        coefficients = torch.stack((values-a/2+b/6, values-b/3, values+a/2+b/6), dim=2)
        minimum = coefficients.amin(dim=2)
        theta = torch.where(minimum < 0, values/(values-minimum).clamp_min(torch.finfo(values.dtype).tiny), 1.0)
        theta = torch.where(minimum < 0, theta*(1-8*torch.finfo(values.dtype).eps), theta)
        return values[:, :, None] + theta[:, :, None]*(coefficients-values[:, :, None])

    def primitive(self, state, reconstruction, points):
        """Integral from the left boundary, evaluated at (edge, vx) points.

        Exterior extensions are fixed reservoirs. Departures outside the left
        domain occur only for incoming vx>0, and similarly vx<0 on the right.
        """
        mass = state * self.widths[:, None, None, None]
        prefix = torch.cat((torch.zeros_like(state[:1]), mass.cumsum(0)), dim=0)
        index = torch.searchsorted(self.edges, points.contiguous(), right=True) - 1
        index = index.clamp(0, self.nx - 1)
        velocity_index = torch.arange(self.nv, device=self.device)[None, :]
        bounded = points.clamp(self.x_left, self.x_right)
        offset = (bounded - self.edges[index])[:, :, None, None]
        width = self.widths[index][:, :, None, None]
        if self.reconstruction == "minmod":
            partial = (state[index, velocity_index] * offset
                       + 0.5 * reconstruction[index, velocity_index] * offset * (offset - width))
        else:
            t = offset / width
            c = reconstruction[index, velocity_index]
            # Exact antiderivatives of (1-t)^2, 2t(1-t), and t^2.
            partial = width * (c[:, :, 0]*(t-t*t+t**3/3)
                               + c[:, :, 1]*(t*t-2*t**3/3) + c[:, :, 2]*t**3/3)
        value = prefix[index, velocity_index] + partial
        value += (points - self.x_left).clamp_max(0)[:, :, None, None] * self.maxwellian_minus
        value += (points - self.x_right).clamp_min(0)[:, :, None, None] * self.maxwellian_plus
        return value

    def transport(self, state, dt):
        departure = self.edges[:, None] - dt * self.v[None, :]
        reconstruction = self.slopes(state) if self.reconstruction == "minmod" else self.reconstruct(state)
        integral = self.primitive(state, reconstruction, departure)
        return (integral[1:] - integral[:-1]) / self.widths[:, None, None, None]

    def advance(self, state, dt):
        state = self.transport(state, dt / 2)
        count = 1 if self.collision_dt is None else max(1, math.ceil(dt / self.collision_dt))
        h = dt / count
        for _ in range(count):
            first = state + h / self.kn * self.collision.evaluate(state)
            state = 0.5 * state + 0.5 * (first + h / self.kn * self.collision.evaluate(first))
        return self.transport(state, dt / 2)

    def run(self, *, dt=0.035, **kwargs):
        if dt is None or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        return super().run(dt=dt, **kwargs)


def build_parser():
    parser = base_parser()
    parser.description = __doc__
    parser.set_defaults(nx=100, nv=32, kn=0.1, mach=2.0, t_end=50.0,
                        dt=0.035, time_integrator="ssprk2",
                        output_dir=Path(__file__).resolve().parent / "output_strang")
    for action in parser._actions:
        if action.dest == "time_integrator":
            action.choices = ("ssprk2",)
            action.help = "collision substep integrator (Strang composition is fixed)"
    parser.add_argument("--refine-factor", type=int, default=1,
                        help="subdivide selected base cells by this factor")
    parser.add_argument("--refine-center", type=float, default=0.0)
    parser.add_argument("--refine-width", type=float, default=10.0,
                        help="full width of window selecting base-cell centers")
    parser.add_argument("--collision-dt", type=float, default=None,
                        help="maximum collision substep; default one SSPRK2 step per outer step")
    parser.add_argument("--reconstruction", choices=("cweno3", "minmod"), default="cweno3")
    return add_mesh_arguments(parser)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.dt) or args.dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(args.t_end) or args.t_end < 0:
        raise ValueError("terminal time must be finite and nonnegative")
    chunk = collision_batch_chunk(args)
    collision = make_unsteady_collision(args, chunk)
    solver = StrangShock1D(collision, **{k: getattr(args, k) for k in (
        "nx", "nv", "velocity_half_width", "kn", "gamma", "mach", "rho_minus",
        "temperature_minus", "device", "refine_factor", "refine_center", "refine_width", "collision_dt", "reconstruction", "mesh", "stretch")})
    signature = {
        "case": "sod_shock_1d", "scheme": ("conservative_sl_linear_strang_v1" if args.reconstruction == "minmod"
                                           else "conservative_sl_cweno3_strang_v1"),
        **{k: getattr(args, k) for k in ("nx", "nv", "velocity_half_width", "kn", "gamma",
            "mach", "rho_minus", "temperature_minus", "refine_factor", "refine_center", "refine_width")},
        "collision_dt": solver.collision_dt, **collision_signature(args),
    }
    if args.mesh == "sinh":
        for key in ("refine_factor", "refine_center", "refine_width"):
            signature.pop(key)
        signature.update(mesh="sinh_v1", stretch=args.stretch)
    import hashlib
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()[:12]
    output = args.output_dir / (f"nx{solver.nx}_nv{args.nv}_Ma{filename_float(args.mach)}"
        f"_kn{filename_float(args.kn)}_tend{filename_float(args.t_end)}"
        f"_dt{filename_float(args.dt)}_{args.collision_backend}_strang_{digest}")
    guard_output_directory(output, overwrite=args.overwrite, resume=args.resume)
    checkpoint = output / "checkpoint_latest.pt"
    source = None
    if args.resume:
        selected = latest_compatible_resume_checkpoint(args.output_dir, target_checkpoint=checkpoint,
            t_end=args.t_end, nominal_dt=args.dt, signature=signature)
        if selected is None:
            raise FileNotFoundError("no compatible Strang checkpoint found")
        source = selected[0]
    if not args.quiet:
        substeps = 1 if solver.collision_dt is None else math.ceil(args.dt / solver.collision_dt)
        print(f"reconstruction={args.reconstruction}, Nx={solver.nx}, dx_min={solver.widths.min().item():.6g}, dt={args.dt}, "
              f"transport CFL={args.dt * args.velocity_half_width / solver.widths.min().item():.6g}, "
              f"collision substeps={substeps}, collision dt/Kn={args.dt / substeps / args.kn:.6g}", flush=True)
        print(f"output: {output}", flush=True)
    state = solver.run(t_end=args.t_end, dt=args.dt, progress=not args.quiet,
        checkpoint_path=checkpoint, checkpoint_interval_seconds=60 * args.checkpoint_minutes,
        resume=args.resume, resume_from_path=source, checkpoint_signature=signature)
    save_results(output, solver=solver, final_state=state, run_configuration={
        **vars(args), "Nx": solver.nx, "Nv": args.nv, "Kn": args.kn, "Mach": args.mach,
        "T_end": args.t_end, "dt": args.dt, "collision_dt": solver.collision_dt,
        "number_of_steps": solver.last_number_of_steps, "resume_source_checkpoint": source,
    })
    # The shared writer describes the original solver; specialize its metadata.
    path = output / "run_config.json"
    config = json.loads(path.read_text())
    config.update(time_discretization="Strang T(dt/2) C(dt) T(dt/2); SSPRK2 collision",
                  transport=f"conservative semi-Lagrangian; nonuniform {args.reconstruction} reconstruction",
                  reconstruction=args.reconstruction,
                  spatial_grid=f"{args.mesh} nonuniform finite-volume cell centers",
                  collision_dt=solver.collision_dt, collision_x_batch_chunk=chunk)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    np.save(output / "x_edges.npy", solver.edges.cpu().numpy())
    rho = solver.moments(state)[:, 0].cpu().numpy()
    x = solver.x.cpu().numpy()
    thickness = abs(solver.rho_plus - solver.rho_minus) / np.max(np.abs(np.gradient(rho, x)))
    (output / "shock_thickness.json").write_text(json.dumps({
        "density_thickness": float(thickness), "definition": "abs(rho_plus-rho_minus)/max(abs(d rho/d x))",
        "gradient": "numpy.gradient on cell centers", "grid_convergence_verified": False,
    }, indent=2) + "\n")
    print(f"saved {output}; density thickness estimate={thickness:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
