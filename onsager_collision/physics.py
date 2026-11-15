"""Minimal velocity-space geometry for the constant-kernel Onsager model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


MAXWELL_KERNEL_CONSTANT = 1.0 / (4.0 * math.pi)


@dataclass(frozen=True)
class ConstantMaxwellKernel:
    """Normalized Maxwell-molecule kernel used by the current labels."""

    value: float = MAXWELL_KERNEL_CONSTANT
    homogeneity: float = 0.0


@dataclass(frozen=True)
class VelocityGrid:
    """Flattened uniform Cartesian quadrature on ``[-L,L)^3``."""

    points: torch.Tensor
    weights: torch.Tensor
    n: int
    L: float

    @classmethod
    def cartesian(
        cls,
        n: int,
        L: float,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "VelocityGrid":
        dv = 2.0 * float(L) / int(n)
        one_d = (torch.arange(n, device=device, dtype=dtype) - n // 2) * dv
        points = torch.stack(
            torch.meshgrid(one_d, one_d, one_d, indexing="ij"), dim=-1
        ).reshape(-1, 3)
        weights = torch.full((n**3,), dv**3, device=device, dtype=dtype)
        return cls(points=points, weights=weights, n=int(n), L=float(L))

    @property
    def size(self) -> int:
        return self.n**3

    @property
    def dv(self) -> float:
        return 2.0 * self.L / self.n

    def physical_invariants(self) -> torch.Tensor:
        radius2 = torch.sum(self.points.square(), dim=-1, keepdim=True)
        return torch.cat(
            (torch.ones_like(radius2), self.points, radius2), dim=-1
        )


def weighted_moments(
    f: torch.Tensor, grid: VelocityGrid
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return density, bulk velocity, and scalar temperature."""

    weights = grid.weights[None]
    rho = torch.sum(weights * f, dim=1).clamp_min(
        torch.finfo(f.dtype).tiny
    )
    momentum = torch.einsum("bn,nd->bd", weights * f, grid.points)
    velocity = momentum / rho[:, None]
    centered = grid.points[None] - velocity[:, None]
    energy = torch.sum(
        weights * f * torch.sum(centered.square(), dim=-1), dim=1
    )
    temperature = (energy / (3.0 * rho)).clamp_min(1e-8)
    return rho, velocity, temperature


def continuous_maxwellian(
    grid: VelocityGrid,
    rho: torch.Tensor,
    velocity: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    centered = grid.points[None] - velocity[:, None]
    radius2 = torch.sum(centered.square(), dim=-1)
    normalizer = rho / torch.pow(2.0 * math.pi * temperature, 1.5)
    return normalizer[:, None] * torch.exp(
        -radius2 / (2.0 * temperature[:, None])
    )


def logarithmic_mean(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Stable positive logarithmic mean of two positive tensors."""

    tiny = torch.finfo(left.dtype).tiny
    left = left.clamp_min(tiny)
    right = right.clamp_min(tiny)
    difference = left - right
    relative = torch.abs(difference) / torch.maximum(left, right)
    log_difference = torch.log(left) - torch.log(right)
    denominator = torch.where(
        relative < 1e-5, torch.ones_like(log_difference), log_difference
    )
    quotient = difference / denominator
    return torch.where(relative < 1e-5, 0.5 * (left + right), quotient)


@dataclass(frozen=True)
class GaussianMixtureBatch:
    """Small analytic fixture used by structure tests and examples."""

    component_masses: torch.Tensor
    means: torch.Tensor
    cholesky: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.component_masses.shape[0]

    @property
    def mass(self) -> torch.Tensor:
        return self.component_masses.sum(dim=1)

    def density(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim == 2:
            points = points[None].expand(self.batch_size, -1, -1)
        difference = points[:, :, None] - self.means[:, None]
        rhs = difference.permute(0, 2, 3, 1)
        solved = torch.linalg.solve_triangular(
            self.cholesky, rhs, upper=False
        ).permute(0, 3, 1, 2)
        exponent = -0.5 * torch.sum(solved.square(), dim=-1)
        determinant_sqrt = torch.prod(
            torch.diagonal(self.cholesky, dim1=-2, dim2=-1), dim=-1
        )
        normalizer = (2.0 * math.pi) ** (-1.5) / determinant_sqrt
        values = torch.exp(exponent) * normalizer[:, None]
        return torch.sum(values * self.component_masses[:, None], dim=-1)

    def analytic_moments(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rho = self.mass
        velocity = torch.sum(
            self.component_masses[:, :, None] * self.means, dim=1
        ) / rho[:, None]
        covariance = self.cholesky @ self.cholesky.transpose(-1, -2)
        centered = self.means - velocity[:, None]
        energy = (
            torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1)
            + torch.sum(centered.square(), dim=-1)
        )
        temperature = torch.sum(
            self.component_masses * energy, dim=1
        ) / (3.0 * rho)
        return rho, velocity, temperature

    def blend_with_matching_maxwellian(
        self, alpha: torch.Tensor
    ) -> "GaussianMixtureBatch":
        rho, velocity, temperature = self.analytic_moments()
        eye = torch.eye(
            3, device=self.means.device, dtype=self.means.dtype
        )[None, None]
        maxwell_cholesky = torch.sqrt(temperature)[:, None, None, None] * eye
        alpha = alpha.to(device=self.means.device, dtype=self.means.dtype)
        return GaussianMixtureBatch(
            component_masses=torch.cat(
                (
                    ((1.0 - alpha) * rho)[:, None],
                    alpha[:, None] * self.component_masses,
                ),
                dim=1,
            ),
            means=torch.cat((velocity[:, None], self.means), dim=1),
            cholesky=torch.cat((maxwell_cholesky, self.cholesky), dim=1),
        )


def random_gaussian_mixture(
    batch_size: int,
    components: int,
    *,
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype,
    mass: float = 10.0,
) -> GaussianMixtureBatch:
    """Generate a smooth positive non-equilibrium test state."""

    weights = torch.rand(
        batch_size,
        components,
        device=device,
        dtype=dtype,
        generator=generator,
    ).clamp_min(1e-4)
    component_masses = mass * weights / weights.sum(dim=1, keepdim=True)
    means = 2.5 * torch.randn(
        batch_size,
        components,
        3,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    raw = torch.randn(
        batch_size,
        components,
        3,
        3,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    covariance = 0.15 * (raw @ raw.transpose(-1, -2))
    covariance = covariance + 0.55 * torch.eye(
        3, device=device, dtype=dtype
    )
    state = GaussianMixtureBatch(
        component_masses, means, torch.linalg.cholesky(covariance)
    )
    alpha = 0.1 + 0.8 * torch.rand(
        batch_size, device=device, dtype=dtype, generator=generator
    )
    return state.blend_with_matching_maxwellian(alpha)
