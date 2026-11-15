"""Discrete maximum-entropy Maxwellian on a Cartesian velocity grid.

The solver matches the five discrete moments of ``f`` with an exponential
family

    M(v) = exp(lambda_0 + lambda_1 v_x/L + lambda_2 v_y/L
               + lambda_3 v_z/L + lambda_4 |v|^2/L^2).

Only the distribution values are differentiated by the surrounding neural
model, so the small Newton solve intentionally runs under ``no_grad``.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class DiscreteMaxwellian(nn.Module):
    def __init__(
        self, *, n: int = 64, L: float = 10.0, max_iter: int = 12,
        tolerance_float32: float = 2e-6, tolerance_float64: float = 2e-11,
        moment_floor: float = 1e-30,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.L = float(L)
        self.dv = 2.0 * self.L / self.n
        self.dv3 = self.dv**3
        self.max_iter = int(max_iter)
        self.tolerance_float32 = float(tolerance_float32)
        self.tolerance_float64 = float(tolerance_float64)
        self.moment_floor = float(moment_floor)

        v = (torch.arange(self.n, dtype=torch.float64) - self.n // 2) * self.dv
        vx, vy, vz = torch.meshgrid(v, v, v, indexing="ij")
        v2 = vx.square() + vy.square() + vz.square()
        psi = torch.stack(
            (torch.ones_like(vx), vx / self.L, vy / self.L, vz / self.L, v2 / self.L**2), dim=-1
        ).reshape(-1, 5)
        self.register_buffer("v", v, persistent=False)
        self.register_buffer("vx", vx, persistent=False)
        self.register_buffer("vy", vy, persistent=False)
        self.register_buffer("vz", vz, persistent=False)
        self.register_buffer("v2", v2, persistent=False)
        self.register_buffer("psi", psi, persistent=False)
        self.register_buffer(
            "newton_eye", torch.eye(5, dtype=torch.float64), persistent=False
        )

    @staticmethod
    def _as_grid(f: torch.Tensor) -> tuple[torch.Tensor, bool]:
        had_channel = f.ndim == 5
        return (f[:, 0] if had_channel else f), had_channel

    def moments(self, f: torch.Tensor) -> torch.Tensor:
        grid, _ = self._as_grid(f)
        psi = self.psi.to(grid)
        flat = grid.reshape(grid.shape[0], -1)
        return self.dv3 * (flat[:, :, None] * psi[None, :, :]).sum(dim=1)

    def _initial_lambda(self, target: torch.Tensor) -> torch.Tensor:
        rho = target[:, 0].clamp_min(self.moment_floor)
        u = self.L * target[:, 1:4] / rho[:, None]
        mean_v2 = self.L * self.L * target[:, 4] / rho
        temperature = ((mean_v2 - u.square().sum(dim=-1)) / 3.0).clamp_min(1e-6)
        lam = target.new_empty(target.shape)
        thermal_term = 1.5 * torch.log(2.0 * math.pi * temperature)
        kinetic_term = u.square().sum(dim=-1) / (2.0 * temperature)
        lam[:, 0] = torch.log(rho) - thermal_term - kinetic_term
        lam[:, 1:4] = self.L * u / temperature[:, None]
        lam[:, 4] = -(self.L * self.L) / (2.0 * temperature)
        return lam

    def _distribution_and_moments(
        self, lam: torch.Tensor, *, dtype: torch.dtype, compute_jacobian: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        psi = self.psi.to(device=lam.device, dtype=dtype)
        exponent = lam.to(dtype) @ psi.T
        maximum = 70.0 if dtype == torch.float64 else 40.0
        distribution = torch.exp(exponent.clamp(min=-90.0, max=maximum))
        moments = self.dv3 * (distribution[:, :, None] * psi[None, :, :]).sum(dim=1)
        jacobian = self.dv3 * torch.einsum("bp,pi,pj->bij", distribution, psi, psi) if compute_jacobian else None
        return distribution, moments, jacobian

    def forward(self, f: torch.Tensor, *, return_diagnostics: bool = False):
        grid, had_channel = self._as_grid(f)
        solve_dtype = torch.float64 if grid.dtype == torch.float64 else torch.float32
        with torch.no_grad():
            physical_target = self.moments(grid.to(solve_dtype))
            density = physical_target[:, :1]
            valid_density = torch.isfinite(density[:, 0]) & (density[:, 0] > 0.0)
            density_safe = torch.where(valid_density[:, None], density, torch.ones_like(density))
            # Solve only for the unit-mass shape and restore density
            # afterwards.  This removes density from the Newton problem and
            # from exponent clipping, so M(c f) = c M(f) throughout the
            # representable floating-point range.
            target = physical_target / density_safe
            lam = self._initial_lambda(target)
            # All basis functions are already nondimensionalized by L, so the
            # mass is the correct common scale for all five residuals.  Using
            # each target component as its own scale makes a physically zero
            # momentum demand impossible float32 absolute accuracy and can
            # drive an otherwise converged centered Maxwellian away from its
            # solution.
            scale = target[:, :1].abs().clamp_min(self.moment_floor)
            tolerance = self.tolerance_float64 if solve_dtype == torch.float64 else self.tolerance_float32
            iterations = 0
            relative_error = torch.full((grid.shape[0],), math.inf, device=grid.device, dtype=solve_dtype)
            needs_final_evaluation = True

            for iterations in range(1, self.max_iter + 1):
                distribution, moments, _ = self._distribution_and_moments(
                    lam, dtype=solve_dtype, compute_jacobian=False
                )
                needs_final_evaluation = False
                residual = moments - target
                relative_error = (residual.abs() / scale).amax(dim=-1)
                if bool((relative_error <= tolerance).all()):
                    break

                psi = self.psi.to(device=grid.device, dtype=solve_dtype)
                jacobian = self.dv3 * torch.einsum("bp,pi,pj->bij", distribution, psi, psi)
                eye = self.newton_eye.to(device=grid.device, dtype=solve_dtype)[None]
                delta = torch.linalg.solve(jacobian + 1e-10 * eye, residual[..., None]).squeeze(-1)
                current_score = (residual / scale).square().sum(dim=-1)
                accepted = torch.zeros_like(current_score, dtype=torch.bool)
                candidate_lam = lam

                for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    trial_lam = lam - damping * delta
                    _, trial_moments, _ = self._distribution_and_moments(
                        trial_lam, dtype=solve_dtype, compute_jacobian=False
                    )
                    trial_score = ((trial_moments - target) / scale).square().sum(dim=-1)
                    better = (~accepted) & (trial_score < current_score)
                    candidate_lam = torch.where(better[:, None], trial_lam, candidate_lam)
                    accepted = accepted | better

                lam = torch.where(accepted[:, None], candidate_lam, lam - 0.03125 * delta)
                needs_final_evaluation = True

            if needs_final_evaluation:
                distribution, moments, _ = self._distribution_and_moments(
                    lam, dtype=solve_dtype, compute_jacobian=False
                )
            relative_error = ((moments - target).abs() / scale).amax(dim=-1)
            candidate = (density_safe * distribution).reshape_as(grid).to(grid)
            converged = valid_density & torch.isfinite(relative_error) & (relative_error <= tolerance)
            maxwellian = candidate

        output = maxwellian[:, None] if had_channel else maxwellian
        if not return_diagnostics:
            return output
        return output, {
            "iterations": iterations,
            "max_relative_moment_error": relative_error.max(),
            "relative_moment_error": relative_error,
            "converged": converged,
            "density": density[:, 0],
            "physical_moments": physical_target,
            "lambda": lam,
        }
