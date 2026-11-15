"""Float64 1D-x/3D-v numerics for the steady Boltzmann solver.

The Newton unknown in this module is the complete next state ``h=f^(k+1)``:

    v_x d_x h = Kn^{-1} [Q(f+h)-Q(h)-2Q(f)].

The initial Gate phase solves directly for the complete next state ``h``.
The final exact phase uses the signed correction ``s=h-f`` and the same local
Maxwellian LinearCollision geometry as the existing C++ ``Newton1D`` solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal, Protocol

import torch


Side = Literal["left", "right"]
MAX_INNER_ITERATIONS = 20
DEFAULT_GATE_INNER_RELATIVE_TOLERANCE = 1.0e-3


class CollisionOperator(Protocol):
    """Minimal batch interface shared by Gate-FNO and spectral evaluators."""

    def evaluate(self, state: torch.Tensor) -> torch.Tensor:
        """Return ``Q(state,state)`` with the same shape as ``state``."""


class LocalMaxwellianLinearOperator(Protocol):
    def evaluate(self, perturbation: torch.Tensor, *, density: torch.Tensor,
                 velocity: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """Return ``Q[M,perturbation]+Q[perturbation,M]`` cellwise."""


@dataclass(frozen=True)
class VelocityGrid:
    n: int
    L: float
    dv: float
    dv3: float
    v: torch.Tensor
    vx: torch.Tensor
    vy: torch.Tensor
    vz: torch.Tensor
    v2: torch.Tensor


@dataclass(frozen=True)
class WallCondition:
    """Maxwell-wall velocity and temperature; density is flux determined."""

    velocity: tuple[float, float, float]
    temperature: float


@dataclass(frozen=True)
class WallInflow:
    values: torch.Tensor
    density: torch.Tensor
    incoming_flux: torch.Tensor
    outgoing_flux: torch.Tensor
    total_flux: torch.Tensor


@dataclass(frozen=True)
class DiffuseWallKernel:
    """Precomputed unit Maxwell inflow and half-space masks for one wall."""

    side: Side
    unit_inflow: torch.Tensor
    outgoing_mask: torch.Tensor
    unit_incoming_flux: torch.Tensor


@dataclass(frozen=True)
class Solver1DConfig:
    nx: int = 16
    x_left: float = -0.5
    x_right: float = 0.5
    kn: float = 1.0
    max_outer_iterations: int = 30
    max_inner_iterations: int = MAX_INNER_ITERATIONS
    max_final_refinement_outer_iterations: int = 30
    max_final_refinement_inner_iterations: int = MAX_INNER_ITERATIONS
    outer_tolerance: float = 1.0e-5
    final_refinement_outer_tolerance: float | None = None
    inner_relative_tolerance: float = DEFAULT_GATE_INNER_RELATIVE_TOLERANCE
    final_refinement_inner_relative_tolerance: float = 1.0e-6
    mu_scale: float = 1.0
    mu_floor: float = 1.0e-8
    fixed_mu: float | None = None
    preserve_total_mass: bool = True
    require_inner_convergence: bool = False
    concurrent_direction_sweeps: bool = True
    spectral_switch_relative_residual: float = 1.0e-3
    spectral_switch_outer_iteration: int | None = None
    verbose: bool = False

    @property
    def dx(self) -> float:
        return (self.x_right - self.x_left) / self.nx


@dataclass(frozen=True)
class SolveResult:
    state: torch.Tensor
    history: list[dict[str, Any]]
    converged: bool
    used_spectral_residual: bool
    target_total_mass: float
    final_refinement_applied: bool
    final_refinement_inner_converged: bool | None
    final_refinement_converged: bool | None
    final_refinement_outer_iterations_evaluated: int
    final_refinement_updates: int


def make_velocity_grid(n: int, L: float, *, device: str | torch.device) -> VelocityGrid:
    """Create the centered ``[-L,L)^3`` grid used by the checkpoint."""

    n = int(n)
    L = float(L)
    dv = 2.0 * L / n
    v = (torch.arange(n, dtype=torch.float64, device=device) - n // 2) * dv
    vx, vy, vz = torch.meshgrid(v, v, v, indexing="ij")
    return VelocityGrid(n=n, L=L, dv=dv, dv3=dv**3, v=v, vx=vx, vy=vy, vz=vz,
                        v2=vx.square() + vy.square() + vz.square())


def discrete_maxwellian(grid: VelocityGrid, *, density: float,
                        velocity: tuple[float, float, float] | torch.Tensor,
                        temperature: float | torch.Tensor) -> torch.Tensor:
    """Return a Maxwellian normalized to the requested discrete density."""

    velocity_tensor = torch.as_tensor(velocity, dtype=torch.float64, device=grid.v.device)
    temperature_tensor = torch.as_tensor(temperature, dtype=torch.float64, device=grid.v.device)
    exponent = -((grid.vx - velocity_tensor[0]).square() + (grid.vy - velocity_tensor[1]).square()
                 + (grid.vz - velocity_tensor[2]).square()) / (2.0 * temperature_tensor)
    value = torch.exp(exponent) / (2.0 * math.pi * temperature_tensor) ** 1.5
    return value * (float(density) / (grid.dv3 * value.sum()))


def make_initial_maxwellian(grid: VelocityGrid, *, nx: int, left_wall: WallCondition,
                            right_wall: WallCondition, density: float = 1.0) -> torch.Tensor:
    """Cell-centred local Maxwellians interpolating wall velocity and T."""

    fractions = (torch.arange(nx, dtype=torch.float64, device=grid.v.device) + 0.5) / nx
    left_velocity = torch.tensor(left_wall.velocity, dtype=torch.float64, device=grid.v.device)
    right_velocity = torch.tensor(right_wall.velocity, dtype=torch.float64, device=grid.v.device)
    velocity = (1.0 - fractions[:, None]) * left_velocity[None] + fractions[:, None] * right_velocity[None]
    temperature = (1.0 - fractions) * left_wall.temperature + fractions * right_wall.temperature
    exponent = -(
        (grid.vx[None] - velocity[:, 0, None, None, None]).square()
        + (grid.vy[None] - velocity[:, 1, None, None, None]).square()
        + (grid.vz[None] - velocity[:, 2, None, None, None]).square()
    ) / (2.0 * temperature[:, None, None, None])
    value = torch.exp(exponent) / (2.0 * math.pi * temperature[:, None, None, None]) ** 1.5
    mass = grid.dv3 * value.sum(dim=(-3, -2, -1))
    return value * (float(density) / mass)[:, None, None, None]


def diffuse_wall_inflow(boundary_state: torch.Tensor, grid: VelocityGrid, wall: WallCondition, *,
                        side: Side, kernel: DiffuseWallKernel | None = None) -> WallInflow:
    """Build flux-normalized diffuse inflow from the current outgoing state."""

    kernel = kernel or make_diffuse_wall_kernel(grid, wall, side=side)
    outgoing_flux_tensor = grid.dv3 * torch.sum(grid.vx[kernel.outgoing_mask] * boundary_state[kernel.outgoing_mask])
    density_tensor = -outgoing_flux_tensor / kernel.unit_incoming_flux

    values = density_tensor * kernel.unit_inflow
    incoming_flux_tensor = density_tensor * kernel.unit_incoming_flux
    total_flux_tensor = incoming_flux_tensor + outgoing_flux_tensor
    return WallInflow(values=values, density=density_tensor, incoming_flux=incoming_flux_tensor,
                      outgoing_flux=outgoing_flux_tensor, total_flux=total_flux_tensor)


def make_diffuse_wall_kernel(grid: VelocityGrid, wall: WallCondition, *, side: Side) -> DiffuseWallKernel:
    """Precompute the fixed Maxwellian part of a diffuse reflection wall."""

    incoming_mask = grid.vx > 0.0 if side == "left" else grid.vx < 0.0
    outgoing_mask = grid.vx < 0.0 if side == "left" else grid.vx > 0.0
    unit_wall = discrete_maxwellian(grid, density=1.0, velocity=wall.velocity, temperature=wall.temperature)
    unit_inflow = torch.zeros_like(unit_wall)
    unit_inflow[incoming_mask] = unit_wall[incoming_mask]
    unit_incoming_flux = grid.dv3 * torch.sum(grid.vx * unit_inflow)
    return DiffuseWallKernel(side=side, unit_inflow=unit_inflow,
                             outgoing_mask=outgoing_mask, unit_incoming_flux=unit_incoming_flux)


def transport_residual(
    state: torch.Tensor,
    collision: torch.Tensor,
    grid: VelocityGrid,
    *,
    walls: tuple[WallCondition, WallCondition],
    dx: float,
    kn: float,
    wall_kernels: tuple[DiffuseWallKernel, DiffuseWallKernel] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return ``v_x D_x f-Q(f)/Kn`` with first-order upwind fluxes."""

    wall_kernels = wall_kernels or (make_diffuse_wall_kernel(grid, walls[0], side="left"), make_diffuse_wall_kernel(grid, walls[1], side="right"))
    left = diffuse_wall_inflow(state[0], grid, walls[0], side="left", kernel=wall_kernels[0])
    right = diffuse_wall_inflow(state[-1], grid, walls[1], side="right", kernel=wall_kernels[1])
    left_neighbor = torch.cat((left.values[None], state[:-1]), dim=0)
    right_neighbor = torch.cat((state[1:], right.values[None]), dim=0)
    advection = torch.where(grid.vx[None] >= 0.0, grid.vx[None] * (state - left_neighbor) / dx,
                            grid.vx[None] * (right_neighbor - state) / dx)
    residual = advection - collision / kn
    statistics = torch.stack((dx * grid.dv3 * torch.sum(residual.square()), left.density, right.density, left.total_flux, right.total_flux, state.min(), dx * grid.dv3 * torch.sum(state))).tolist()
    return residual, {
        "weighted_l2": math.sqrt(statistics[0]),
        "left_wall_density": statistics[1],
        "right_wall_density": statistics[2],
        "left_mass_flux": statistics[3],
        "right_mass_flux": statistics[4],
        "minimum_state": statistics[5],
        "total_mass": statistics[6],
    }


def regularized_sweep(
    collision_rhs: torch.Tensor,
    h_old: torch.Tensor,
    mu: torch.Tensor,
    grid: VelocityGrid,
    *,
    dx: float,
    kn: float,
    left_inflow: torch.Tensor,
    right_inflow: torch.Tensor,
    concurrent_directions: bool = True,
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream] | None = None,
) -> torch.Tensor:
    """Solve one regularized source iteration for the complete state ``h``."""

    nx = h_old.shape[0]

    flat_rhs = collision_rhs.reshape(nx, -1)
    flat_old = h_old.reshape(nx, -1)
    flat_output = torch.empty_like(flat_old)
    vx = grid.vx.reshape(-1)
    positive = vx > 0.0
    negative = vx < 0.0
    zero = ~(positive | negative)
    vx_positive = vx[positive]
    vx_negative = vx[negative]
    source = (flat_rhs + mu[:, None] * flat_old) / kn

    def sweep_positive() -> None:
        previous = left_inflow.reshape(-1)[positive]
        coefficient = vx_positive / dx
        for index in range(nx):
            denominator = mu[index] / kn + coefficient
            following = (source[index, positive] + coefficient * previous) / denominator
            flat_output[index, positive] = following
            previous = following

    def sweep_negative() -> None:
        following = right_inflow.reshape(-1)[negative]
        coefficient = -vx_negative / dx
        for index in range(nx - 1, -1, -1):
            denominator = mu[index] / kn + coefficient
            previous = (source[index, negative] + coefficient * following) / denominator
            flat_output[index, negative] = previous
            following = previous

    use_streams = concurrent_directions and h_old.is_cuda
    if use_streams:
        streams = streams or (torch.cuda.Stream(device=h_old.device), torch.cuda.Stream(device=h_old.device))
        current = torch.cuda.current_stream(h_old.device)
        streams[0].wait_stream(current)
        streams[1].wait_stream(current)
        with torch.cuda.stream(streams[0]):
            sweep_positive()
        with torch.cuda.stream(streams[1]):
            sweep_negative()
        current.wait_stream(streams[0])
        current.wait_stream(streams[1])
        flat_output.record_stream(streams[0])
        flat_output.record_stream(streams[1])
    else:
        sweep_positive()
        sweep_negative()
    flat_output[:, zero] = source[:, zero] / (mu[:, None] / kn)
    return flat_output.reshape_as(h_old)


def batched_newton_rhs(collision: CollisionOperator, *, f: torch.Tensor,
                       h: torch.Tensor, q_f: torch.Tensor) -> torch.Tensor:
    """Evaluate ``Q(f+h)-Q(h)-2Q(f)`` with one pair batch."""

    pair = torch.cat((f + h, h), dim=0)
    pair_collision = collision.evaluate(pair)
    q_f_plus_h, q_h = pair_collision.split(f.shape[0], dim=0)
    return q_f_plus_h - q_h - 2.0 * q_f


def macroscopic_profiles(state: torch.Tensor, grid: VelocityGrid) -> dict[str, torch.Tensor]:
    """Return cellwise density, velocity, temperature, stress, and heat flux."""

    rho = grid.dv3 * state.sum(dim=(-3, -2, -1))
    momentum = torch.stack(tuple(grid.dv3 * torch.sum(state * component[None], dim=(-3, -2, -1))
                                 for component in (grid.vx, grid.vy, grid.vz)), dim=-1)
    velocity = momentum / rho[:, None]
    energy = grid.dv3 * torch.sum(state * grid.v2[None], dim=(-3, -2, -1))
    temperature = (energy / rho - velocity.square().sum(dim=-1)) / 3.0
    cx = grid.vx[None] - velocity[:, 0, None, None, None]
    cy = grid.vy[None] - velocity[:, 1, None, None, None]
    cz = grid.vz[None] - velocity[:, 2, None, None, None]
    c2 = cx.square() + cy.square() + cz.square()
    pxy = grid.dv3 * torch.sum(state * cx * cy, dim=(-3, -2, -1))
    heat_flux_x = 0.5 * grid.dv3 * torch.sum(state * cx * c2, dim=(-3, -2, -1))
    return {"rho": rho, "ux": velocity[:, 0], "uy": velocity[:, 1], "uz": velocity[:, 2],
            "temperature": temperature, "pxy": pxy, "heat_flux_x": heat_flux_x}


class SteadyBoltzmann1D:
    """Newton-h outer iteration with a batched direction collision."""

    def __init__(
        self,
        *,
        grid: VelocityGrid,
        walls: tuple[WallCondition, WallCondition],
        direction_collision: CollisionOperator,
        config: Solver1DConfig,
        final_nonlinear_collision: CollisionOperator | None = None,
        final_linearized_collision: LocalMaxwellianLinearOperator | None = None,
        direction_phase: str = "gate",
        direction_residual_backend: str = "gate_fno",
    ) -> None:
        self.grid = grid
        self.walls = walls
        self.direction_collision = direction_collision
        self.final_nonlinear_collision = final_nonlinear_collision
        self.final_linearized_collision = final_linearized_collision
        self.direction_phase = direction_phase
        self.direction_residual_backend = direction_residual_backend
        self.config = config
        self._streams = (torch.cuda.Stream(device=grid.v.device), torch.cuda.Stream(device=grid.v.device)) if config.concurrent_direction_sweeps and grid.v.is_cuda else None
        self._wall_kernels = (make_diffuse_wall_kernel(grid, walls[0], side="left"),
                              make_diffuse_wall_kernel(grid, walls[1], side="right"))

    def _mu(self, f: torch.Tensor) -> torch.Tensor:
        if self.config.fixed_mu is not None:
            return torch.full((self.config.nx,), self.config.fixed_mu, dtype=torch.float64, device=f.device)
        density = self.grid.dv3 * f.sum(dim=(-3, -2, -1))
        return (self.config.mu_scale * density).clamp_min(self.config.mu_floor)

    def _source_iteration_solve(
        self,
        f: torch.Tensor,
        mu: torch.Tensor,
        collision_rhs_builder: Callable[[torch.Tensor], torch.Tensor],
        *,
        tolerance: float,
        require_convergence: bool,
        log_prefix: str,
        max_iterations: int,
    ) -> tuple[torch.Tensor, list[dict[str, float]], bool]:
        h = f.clone()
        rows: list[dict[str, float]] = []
        converged = False
        for inner_iteration in range(max_iterations):
            collision_rhs = collision_rhs_builder(h)
            left = diffuse_wall_inflow(h[0], self.grid, self.walls[0], side="left", kernel=self._wall_kernels[0])
            right = diffuse_wall_inflow(h[-1], self.grid, self.walls[1], side="right", kernel=self._wall_kernels[1])
            swept = regularized_sweep(
                collision_rhs,
                h,
                mu,
                self.grid,
                dx=self.config.dx,
                kn=self.config.kn,
                left_inflow=left.values,
                right_inflow=right.values,
                concurrent_directions=self.config.concurrent_direction_sweeps,
                streams=self._streams,
            )
            h_new = swept
            measure = self.config.dx * self.grid.dv3
            statistics = torch.stack((measure * torch.sum((h_new - h).square()), measure * torch.sum(h_new.square()), h_new.min(), left.density, right.density, measure * torch.sum(h_new))).tolist()
            difference = math.sqrt(statistics[0])
            scale = max(math.sqrt(statistics[1]), torch.finfo(torch.float64).tiny)
            relative_error = difference / scale
            row = {
                "inner_iteration": float(inner_iteration),
                "relative_change": relative_error,
                "relative_error": relative_error,
                "weighted_change": difference,
                "minimum_h": statistics[2],
                "left_wall_density": statistics[3],
                "right_wall_density": statistics[4],
                "total_mass": statistics[5],
            }
            rows.append(row)
            h = h_new
            if self.config.verbose:
                print(f"    {log_prefix} {inner_iteration:03d}: rel_error={relative_error:.6e}", flush=True)
            if relative_error <= tolerance:
                converged = True
                break
        if not converged and require_convergence:
            raise RuntimeError(f"{log_prefix} source iteration did not converge")
        return h, rows, converged

    def _inner_solve(self, f: torch.Tensor, q_f: torch.Tensor,
                     mu: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, float]], bool]:
        return self._source_iteration_solve(
            f,
            mu,
            lambda h: batched_newton_rhs(
                self.direction_collision,
                f=f,
                h=h,
                q_f=q_f,
            ),
            tolerance=self.config.inner_relative_tolerance,
            require_convergence=self.config.require_inner_convergence,
            log_prefix="inner",
            max_iterations=self.config.max_inner_iterations,
        )

    def _final_refinement_inner_solve(self, f: torch.Tensor, exact_residual: torch.Tensor,
                                      mu: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, float]], bool]:
        """Solve the local-Maxwellian quasi-Newton correction equation.

        For ``s=h-f`` the fixed-point equation is

        ``(T-L_M/Kn)s = -F_exact(f)``.

        This follows the C++ ``Newton1D`` source iteration.  The signed
        correction uses the linear diffuse-wall map, and Gate-FNO is not
        evaluated anywhere in this refinement solve.
        """

        profiles = macroscopic_profiles(f, self.grid)
        density = profiles["rho"]
        velocity = torch.stack((profiles["ux"], profiles["uy"], profiles["uz"]), dim=-1)
        temperature = profiles["temperature"]
        s = torch.zeros_like(f)
        rows: list[dict[str, float]] = []
        converged = False
        measure = self.config.dx * self.grid.dv3
        f_scale = max(math.sqrt(float(measure * torch.sum(f.square()))), torch.finfo(torch.float64).tiny)
        for inner_iteration in range(self.config.max_final_refinement_inner_iterations):
            linear_term = torch.zeros_like(s) if inner_iteration == 0 else self.final_linearized_collision.evaluate(s, density=density, velocity=velocity, temperature=temperature)

            collision_rhs = linear_term - self.config.kn * exact_residual
            left = diffuse_wall_inflow(s[0], self.grid, self.walls[0], side="left", kernel=self._wall_kernels[0])
            right = diffuse_wall_inflow(s[-1], self.grid, self.walls[1], side="right", kernel=self._wall_kernels[1])
            s_new = regularized_sweep(
                collision_rhs,
                s,
                mu,
                self.grid,
                dx=self.config.dx,
                kn=self.config.kn,
                left_inflow=left.values,
                right_inflow=right.values,
                concurrent_directions=self.config.concurrent_direction_sweeps,
                streams=self._streams,
            )
            h_new = f + s_new
            statistics = torch.stack((measure * torch.sum((s_new - s).square()), measure * torch.sum(s_new.square()), h_new.min(), left.density, right.density, measure * torch.sum(h_new))).tolist()
            difference = math.sqrt(statistics[0])
            scale = max(math.sqrt(statistics[1]), torch.finfo(torch.float64).tiny)
            relative_error = difference / scale
            row = {
                "inner_iteration": float(inner_iteration),
                "relative_change": relative_error,
                "relative_error": relative_error,
                "weighted_change": difference,
                "relative_step": math.sqrt(statistics[1]) / f_scale,
                "minimum_h": statistics[2],
                "left_wall_density_correction": statistics[3],
                "right_wall_density_correction": statistics[4],
                "total_mass": statistics[5],
            }
            rows.append(row)
            s = s_new
            if self.config.verbose:
                print(f"    final inner {inner_iteration:03d}: rel_error={relative_error:.6e}, "
                      f"step={row['relative_step']:.6e}", flush=True)
            if relative_error <= self.config.final_refinement_inner_relative_tolerance:
                converged = True
                break
        return f + s, rows, converged

    def solve(self, initial: torch.Tensor) -> SolveResult:
        f = initial.clone()
        final_outer_tolerance = (
            self.config.outer_tolerance
            if self.config.final_refinement_outer_tolerance is None
            else self.config.final_refinement_outer_tolerance
        )
        target_total_mass = float(self.config.dx * self.grid.dv3 * torch.sum(initial))
        history: list[dict[str, Any]] = []
        initial_gate_residual: float | None = None
        converged = False
        gate_phase_converged = False
        final_refinement_applied = False
        final_refinement_inner_converged: bool | None = None
        final_refinement_converged: bool | None = None
        final_refinement_outer_iterations_evaluated = 0
        final_refinement_updates = 0

        # Phase 1: direction-collision residual and coarse inner solves.
        for outer_iteration in range(self.config.max_outer_iterations + 1):
            q_f_direction = self.direction_collision.evaluate(f)
            _, gate_diagnostics = transport_residual(f, q_f_direction, self.grid, walls=self.walls, dx=self.config.dx, kn=self.config.kn, wall_kernels=self._wall_kernels)
            gate_norm = gate_diagnostics["weighted_l2"]
            initial_gate_residual = initial_gate_residual or max(gate_norm, torch.finfo(torch.float64).tiny)
            gate_relative = gate_norm / initial_gate_residual

            outer_norm = gate_norm
            row: dict[str, Any] = {
                "phase": self.direction_phase,
                "outer_iteration": outer_iteration,
                "phase_outer_iteration": outer_iteration,
                "residual_backend": self.direction_residual_backend,
                "weighted_residual": outer_norm,
                f"{self.direction_phase}_weighted_residual": gate_norm,
                f"{self.direction_phase}_relative_residual": gate_relative,
                "left_wall_density": gate_diagnostics["left_wall_density"],
                "right_wall_density": gate_diagnostics["right_wall_density"],
                "left_mass_flux": gate_diagnostics["left_mass_flux"],
                "right_mass_flux": gate_diagnostics["right_mass_flux"],
                "minimum_f": gate_diagnostics["minimum_state"],
                "outer_update": "none",
                "inner_iterations": [],
            }
            if self.config.verbose:
                print(
                    f"outer {outer_iteration:03d}: residual={outer_norm:.6e} "
                    f"({self.direction_residual_backend}), "
                    f"{self.direction_phase}_relative={gate_relative:.6e}",
                    flush=True,
                )
            if outer_norm <= self.config.outer_tolerance:
                row["returned_state_residual_evaluated"] = True
                gate_phase_converged = True
                if self.final_nonlinear_collision is None:
                    converged = True
                else:
                    row["outer_update"] = "switch_to_exact_refinement"
                history.append(row)
                break
            if outer_iteration == self.config.max_outer_iterations:
                row["returned_state_residual_evaluated"] = True
                history.append(row)
                break

            mu = self._mu(f)
            h, inner_rows, inner_converged = self._inner_solve(f, q_f_direction, mu)
            if self.config.preserve_total_mass:
                new_mass = self.config.dx * self.grid.dv3 * torch.sum(h)
                new_mass_value = float(new_mass)
                h = h * (target_total_mass / new_mass_value)

            # h already is f^(k+1); no correction subtraction occurs here.
            f = h
            row["outer_update"] = "f_equals_h"
            row["inner_iterations"] = inner_rows
            row["inner_converged"] = inner_converged
            row["returned_state_residual_evaluated"] = False
            row["accepted_total_mass"] = target_total_mass if self.config.preserve_total_mass else inner_rows[-1]["total_mass"]
            history.append(row)

        # Phase 2: every outer error is the true spectral steady residual.
        if gate_phase_converged and self.final_nonlinear_collision is not None:
            final_refinement_applied = True
            initial_exact_residual: float | None = None
            for refinement_outer_iteration in range(self.config.max_final_refinement_outer_iterations + 1):
                q_f_exact = self.final_nonlinear_collision.evaluate(f)

                exact_residual, exact_diagnostics = transport_residual(
                    f, q_f_exact, self.grid, walls=self.walls, dx=self.config.dx,
                    kn=self.config.kn, wall_kernels=self._wall_kernels)
                exact_norm = exact_diagnostics["weighted_l2"]
                initial_exact_residual = initial_exact_residual or max(exact_norm, torch.finfo(torch.float64).tiny)
                exact_relative = exact_norm / initial_exact_residual
                row = {
                    "phase": "exact_refinement",
                    "outer_iteration": len(history),
                    "phase_outer_iteration": refinement_outer_iteration,
                    "residual_backend": "spectral_exact",
                    "weighted_residual": exact_norm,
                    "exact_relative_residual": exact_relative,
                    "left_wall_density": exact_diagnostics["left_wall_density"],
                    "right_wall_density": exact_diagnostics["right_wall_density"],
                    "left_mass_flux": exact_diagnostics["left_mass_flux"],
                    "right_mass_flux": exact_diagnostics["right_mass_flux"],
                    "minimum_f": exact_diagnostics["minimum_state"],
                    "outer_update": "none",
                    "inner_iterations": [],
                    "returned_state_residual_evaluated": True,
                }
                final_refinement_outer_iterations_evaluated += 1
                if self.config.verbose:
                    print(f"refine outer {refinement_outer_iteration:03d}: residual={exact_norm:.6e} "
                          f"(spectral_exact), exact_relative={exact_relative:.6e}", flush=True)

                if exact_norm <= final_outer_tolerance:
                    converged = True
                    final_refinement_converged = True
                    history.append(row)
                    break
                if refinement_outer_iteration == self.config.max_final_refinement_outer_iterations:
                    final_refinement_converged = False
                    history.append(row)
                    break

                mu = self._mu(f)
                h, final_inner_rows, inner_converged = self._final_refinement_inner_solve(f, exact_residual, mu)
                if self.config.preserve_total_mass:
                    new_mass = self.config.dx * self.grid.dv3 * torch.sum(h)
                    new_mass_value = float(new_mass)
                    h = h * (target_total_mass / new_mass_value)

                f = h
                final_refinement_updates += 1
                final_refinement_inner_converged = inner_converged
                row["outer_update"] = "final_maxwellian_quasi_newton"
                row["inner_iterations"] = final_inner_rows
                row["inner_converged"] = inner_converged
                row["returned_state_residual_evaluated"] = False
                row["accepted_total_mass"] = target_total_mass if self.config.preserve_total_mass else final_inner_rows[-1]["total_mass"]
                history.append(row)

        return SolveResult(
            state=f,
            history=history,
            converged=converged,
            used_spectral_residual=final_refinement_applied,
            target_total_mass=target_total_mass,
            final_refinement_applied=final_refinement_applied,
            final_refinement_inner_converged=final_refinement_inner_converged,
            final_refinement_converged=final_refinement_converged,
            final_refinement_outer_iterations_evaluated=final_refinement_outer_iterations_evaluated,
            final_refinement_updates=final_refinement_updates,
        )
