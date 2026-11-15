"""FNO-conditioned full-rank Onsager collision surrogate.

The FNO reads only the normalized kinetic state.  The entropy force is passed
through a separate operator that is linear for every fixed state.  Writing

    K_f = D_f + S^* G_f S,

with positive pointwise multipliers ``D_f`` and ``G_f`` makes the learned
metric self-adjoint and positive semidefinite by construction.  The outer
decay-weighted adjoint projections preserve collision invariants and the
Maxwellian tail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from boltzmann_spectral import ComplexSpectralConv3d, real_parameter_count
from boltzmann_triton import fused_add_gelu

from .physics import (
    VelocityGrid,
    continuous_maxwellian,
    logarithmic_mean,
    weighted_moments,
)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


@dataclass(frozen=True)
class _OnsagerConfig:
    """Frozen pilot configuration for the FNO-conditioned metric."""

    state_channels: int = 7
    conditioner_modes: int = 8
    conditioner_width: int = 24
    conditioner_layers: int = 3
    analysis_channels: int = 4
    analysis_modes: int = 8
    initial_local_rate: float = 0.5
    initial_gate_rate: float = 0.25
    initial_filter_scale: float = 0.02
    log_floor: float = 1e-20

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _StateFNOConditioner(nn.Module):
    """One FNO trunk producing local positive-metric logits.

    The last spectral layer maps directly to the output channels.  This is the
    trained architecture, rather than an inference-only fusion, and avoids a
    width-by-width fourth spectral block followed by a separate readout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        modes: int,
        width: int,
        layers: int,
        local_biases: torch.Tensor,
    ) -> None:
        super().__init__()
        self.layers = int(layers)
        self.lift = nn.Conv3d(in_channels, width, kernel_size=1)
        self.hidden_spectral = nn.ModuleList(
            ComplexSpectralConv3d(width, width, modes, modes, modes)
            for _ in range(layers - 1)
        )
        self.hidden_local = nn.ModuleList(
            nn.Conv3d(width, width, kernel_size=1)
            for _ in range(layers - 1)
        )
        self.readout_spectral = ComplexSpectralConv3d(
            width, out_channels, modes, modes, modes
        )
        self.readout_local = nn.Conv3d(width, out_channels, kernel_size=1)
        self.reset_readout(local_biases)

    def reset_readout(self, local_biases: torch.Tensor) -> None:
        nn.init.zeros_(self.readout_spectral.weight)
        nn.init.zeros_(self.readout_local.weight)
        with torch.no_grad():
            self.readout_local.bias.copy_(local_biases)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.lift(features)
        for spectral, local in zip(
            self.hidden_spectral, self.hidden_local, strict=True
        ):
            value = fused_add_gelu(spectral(value), local(value))
        return self.readout_spectral(value) + self.readout_local(value)


class _OnsagerCore(nn.Module):
    r"""Apply an FNO-conditioned full-rank positive Onsager metric.

    For ``eta=log(f/M[f])`` and logarithmic-mean mobility ``b_f``, define

    .. math::
       z_f=\sqrt{b_f} C_f^\dagger\eta,
       \qquad
       K_f z=D_fz+S^*G_fSz.

    The returned collision field is

    .. math::
       Q_\theta(f)=-C_f\sqrt{b_f}K_fz_f.

    ``D_f`` and every channel of ``G_f`` are positive FNO outputs.  ``S`` is a
    learned complex low-frequency analysis filter and ``S^*`` is implemented
    explicitly with the conjugate filter.  Thus the force-to-flux map remains
    linear for fixed ``f`` even though its coefficients depend nonlinearly on
    the state.
    """

    architecture_name = "decay_fno_onsager"

    def __init__(self, config: _OnsagerConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else _OnsagerConfig()
        biases = torch.tensor(
            [
                _inverse_softplus(self.config.initial_local_rate),
                *(
                    _inverse_softplus(self.config.initial_gate_rate)
                    for _ in range(self.config.analysis_channels)
                ),
            ],
            dtype=torch.float32,
        )
        self.conditioner = _StateFNOConditioner(
            self.config.state_channels,
            1 + self.config.analysis_channels,
            modes=self.config.conditioner_modes,
            width=self.config.conditioner_width,
            layers=self.config.conditioner_layers,
            local_biases=biases,
        )
        filter_shape = (
            self.config.analysis_channels,
            2 * self.config.analysis_modes,
            2 * self.config.analysis_modes,
            2 * self.config.analysis_modes,
            2,
        )
        self.analysis_filter_ri = nn.Parameter(
            self.config.initial_filter_scale * torch.randn(filter_shape)
        )

    @property
    def parameter_count(self) -> int:
        return real_parameter_count(self)

    @staticmethod
    def _flatten_input(f: torch.Tensor, grid: VelocityGrid) -> torch.Tensor:
        if f.ndim == 3:
            f = f.unsqueeze(0)
        return f.reshape(f.shape[0], grid.size)

    @staticmethod
    def normalized_invariants(coordinate: torch.Tensor) -> torch.Tensor:
        radius2 = torch.sum(coordinate.square(), dim=-1, keepdim=True)
        return torch.cat(
            (
                torch.ones_like(radius2),
                coordinate,
                (radius2 - 3.0) / math.sqrt(6.0),
            ),
            dim=-1,
        )

    @staticmethod
    def _projection_geometry(
        mobility: torch.Tensor,
        invariants: torch.Tensor,
        grid: VelocityGrid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis = mobility[:, :, None] * invariants
        gram = torch.einsum(
            "bni,n,bnj->bij", invariants, grid.weights, basis
        )
        return basis, 0.5 * (gram + gram.transpose(-1, -2))

    @staticmethod
    def _project_force(
        values: torch.Tensor,
        invariants: torch.Tensor,
        basis: torch.Tensor,
        gram: torch.Tensor,
        grid: VelocityGrid,
    ) -> torch.Tensor:
        rhs = torch.einsum("bni,n,bn->bi", basis, grid.weights, values)
        coefficients = torch.linalg.solve(
            gram.transpose(-1, -2), rhs[..., None]
        )[..., 0]
        return values - torch.einsum("bni,bi->bn", invariants, coefficients)

    @staticmethod
    def _project_output(
        values: torch.Tensor,
        invariants: torch.Tensor,
        basis: torch.Tensor,
        gram: torch.Tensor,
        grid: VelocityGrid,
    ) -> torch.Tensor:
        rhs = torch.einsum("bni,n,bn->bi", invariants, grid.weights, values)
        coefficients = torch.linalg.solve(gram, rhs[..., None])[..., 0]
        return values - torch.einsum("bni,bi->bn", basis, coefficients)

    def _state_features(
        self,
        f: torch.Tensor,
        maxwellian: torch.Tensor,
        rho: torch.Tensor,
        coordinate: torch.Tensor,
        grid: VelocityGrid,
    ) -> torch.Tensor:
        normalized_f = f / rho[:, None]
        normalized_m = maxwellian / rho[:, None]
        denominator = normalized_f + normalized_m + self.config.log_floor
        residual = (normalized_f - normalized_m) / denominator
        log_ratio = torch.tanh(
            torch.log(normalized_f) - torch.log(normalized_m)
        )
        radius2 = torch.sum(coordinate.square(), dim=-1)
        coordinate_scale = torch.sqrt(1.0 + radius2)
        radial = radius2 / (1.0 + radius2)
        maxwellian_shape = normalized_m / normalized_m.amax(
            dim=1, keepdim=True
        ).clamp_min(self.config.log_floor)
        channels = torch.stack(
            (
                residual,
                log_ratio,
                coordinate[..., 0] / coordinate_scale,
                coordinate[..., 1] / coordinate_scale,
                coordinate[..., 2] / coordinate_scale,
                radial,
                maxwellian_shape,
            ),
            dim=1,
        )
        return channels.reshape(
            f.shape[0], self.config.state_channels, grid.n, grid.n, grid.n
        )

    def _frequency_slice(self, n: int) -> tuple[slice, slice, slice]:
        modes = self.config.analysis_modes
        center = n // 2
        block = slice(center - modes, center + modes)
        return block, block, block

    def _analysis(self, values: torch.Tensor, grid: VelocityGrid) -> torch.Tensor:
        cube = values.reshape(values.shape[0], grid.n, grid.n, grid.n)
        spectrum = torch.fft.fftshift(
            torch.fft.fftn(cube, dim=(-3, -2, -1), norm="ortho"),
            dim=(-3, -2, -1),
        )
        frequency_slice = self._frequency_slice(grid.n)
        low = spectrum[(slice(None), *frequency_slice)]
        filters = torch.view_as_complex(self.analysis_filter_ri.contiguous()).to(
            dtype=low.dtype
        )
        low_channels = low[:, None] * filters[None]
        full = low_channels.new_zeros(
            values.shape[0],
            self.config.analysis_channels,
            grid.n,
            grid.n,
            grid.n,
        )
        full[(slice(None), slice(None), *frequency_slice)] = low_channels
        return torch.fft.ifftn(
            torch.fft.ifftshift(full, dim=(-3, -2, -1)),
            dim=(-3, -2, -1),
            norm="ortho",
        )

    def _adjoint_analysis(
        self,
        values: torch.Tensor,
        grid: VelocityGrid,
    ) -> torch.Tensor:
        spectrum = torch.fft.fftshift(
            torch.fft.fftn(values, dim=(-3, -2, -1), norm="ortho"),
            dim=(-3, -2, -1),
        )
        frequency_slice = self._frequency_slice(grid.n)
        low = spectrum[(slice(None), slice(None), *frequency_slice)]
        filters = torch.view_as_complex(self.analysis_filter_ri.contiguous()).to(
            dtype=low.dtype
        )
        low_scalar = torch.sum(filters.conj()[None] * low, dim=1)
        full = low_scalar.new_zeros(
            values.shape[0], grid.n, grid.n, grid.n
        )
        full[(slice(None), *frequency_slice)] = low_scalar
        return torch.fft.ifftn(
            torch.fft.ifftshift(full, dim=(-3, -2, -1)),
            dim=(-3, -2, -1),
            norm="ortho",
        ).real.reshape(values.shape[0], grid.size)

    def forward_with_aux(
        self, f: torch.Tensor, grid: VelocityGrid
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat = self._flatten_input(f, grid)
        raw_positive = flat.clamp_min(torch.finfo(flat.dtype).tiny)
        rho, velocity, temperature = weighted_moments(raw_positive, grid)
        density_floor = rho[:, None] * self.config.log_floor
        positive_f = torch.maximum(flat, density_floor)
        maxwellian = continuous_maxwellian(grid, rho, velocity, temperature)
        positive_maxwellian = torch.maximum(maxwellian, density_floor)
        eta = torch.log(positive_f) - torch.log(positive_maxwellian)
        mobility = logarithmic_mean(positive_f, positive_maxwellian)
        coordinate = (grid.points[None] - velocity[:, None]) / torch.sqrt(
            temperature
        )[:, None, None]
        invariants = self.normalized_invariants(coordinate)
        basis, gram = self._projection_geometry(mobility, invariants, grid)
        corrected_force = self._project_force(
            eta, invariants, basis, gram, grid
        )
        root_mobility = torch.sqrt(mobility)
        force = root_mobility * corrected_force

        features = self._state_features(
            positive_f, positive_maxwellian, rho, coordinate, grid
        )
        logits = self.conditioner(features).reshape(
            flat.shape[0], 1 + self.config.analysis_channels, grid.size
        )
        local_rate = rho[:, None] * F.softplus(logits[:, 0])
        channel_rates = rho[:, None, None] * F.softplus(logits[:, 1:])

        analysis = self._analysis(force, grid)
        channel_rate_cube = channel_rates.reshape(
            flat.shape[0], self.config.analysis_channels, grid.n, grid.n, grid.n
        )
        nonlocal_middle = self._adjoint_analysis(
            channel_rate_cube.to(analysis.dtype) * analysis,
            grid,
        )
        middle = local_rate * force + nonlocal_middle
        output = -self._project_output(
            root_mobility * middle, invariants, basis, gram, grid
        )

        cell_weight = grid.weights[0]
        dissipation = cell_weight * (
            torch.sum(local_rate * force.square(), dim=1)
            + torch.sum(
                channel_rate_cube * analysis.abs().square(),
                dim=(1, 2, 3, 4),
            )
        )
        return output, {
            "rho": rho,
            "velocity": velocity,
            "temperature": temperature,
            "maxwellian": maxwellian,
            "eta": eta,
            "mobility": mobility,
            "invariants": invariants,
            "correction_basis": basis,
            "projection_gram": gram,
            "corrected_force": corrected_force,
            "local_rate": local_rate,
            "channel_rates": channel_rates,
            "analysis": analysis,
            "spectral_dissipation": dissipation,
        }

    def forward(self, f: torch.Tensor, grid: VelocityGrid) -> torch.Tensor:
        return self.forward_with_aux(f, grid)[0]

    def weak_action(
        self,
        f: torch.Tensor,
        grid: VelocityGrid,
        probe_values: torch.Tensor,
    ) -> torch.Tensor:
        output = self(f, grid)
        return torch.einsum("bpn,n,bn->bp", probe_values, grid.weights, output)


class OnsagerCollision(_OnsagerCore):
    """FNO-conditioned Onsager collision model on one Cartesian grid."""

    def __init__(
        self,
        n: int,
        L: float,
        *,
        fno_modes: int = 8,
        fno_width: int = 24,
        fno_layers: int = 3,
        analysis_channels: int = 4,
        analysis_modes: int | None = None,
        initial_local_rate: float = 0.5,
        initial_gate_rate: float = 0.25,
        initial_filter_scale: float = 0.02,
        bgk_rate: float = 0.5,
        log_floor: float = 1e-20,
    ) -> None:
        resolved_analysis_modes = (
            int(fno_modes) if analysis_modes is None else int(analysis_modes)
        )
        super().__init__(
            _OnsagerConfig(
                conditioner_modes=int(fno_modes),
                conditioner_width=int(fno_width),
                conditioner_layers=int(fno_layers),
                analysis_channels=int(analysis_channels),
                analysis_modes=resolved_analysis_modes,
                initial_local_rate=float(initial_local_rate),
                initial_gate_rate=float(initial_gate_rate),
                initial_filter_scale=float(initial_filter_scale),
                log_floor=float(log_floor),
            )
        )
        self.n = int(n)
        self.L = float(L)
        self.bgk_rate = float(bgk_rate)
        grid = VelocityGrid.cartesian(self.n, self.L, dtype=torch.float32)
        self.register_buffer("_grid_points", grid.points, persistent=False)
        self.register_buffer("_grid_weights", grid.weights, persistent=False)

        self.model_config: dict[str, object] = {
            "n": self.n,
            "L": self.L,
            "fno_modes": int(fno_modes),
            "fno_width": int(fno_width),
            "fno_layers": int(fno_layers),
            "analysis_channels": int(analysis_channels),
            "analysis_modes": resolved_analysis_modes,
            "initial_local_rate": float(initial_local_rate),
            "initial_gate_rate": float(initial_gate_rate),
            "initial_filter_scale": float(initial_filter_scale),
            "bgk_rate": self.bgk_rate,
            "log_floor": float(log_floor),
        }

    def _stored_grid(self) -> VelocityGrid:
        return VelocityGrid(
            points=self._grid_points,
            weights=self._grid_weights,
            n=self.n,
            L=self.L,
        )

    @staticmethod
    def _remove_optional_channel(f: torch.Tensor) -> tuple[torch.Tensor, bool]:
        had_channel = f.ndim == 5
        return (f[:, 0] if had_channel else f), had_channel

    @staticmethod
    def _restore_shape(
        flat: torch.Tensor, grid: VelocityGrid, had_channel: bool
    ) -> torch.Tensor:
        cube = flat.reshape(flat.shape[0], grid.n, grid.n, grid.n)
        return cube[:, None] if had_channel else cube

    def collision_moments(self, q: torch.Tensor) -> torch.Tensor:
        q, _ = self._remove_optional_channel(q)
        grid = self._stored_grid()
        flat = self._flatten_input(q, grid)
        return torch.einsum(
            "ni,n,bn->bi", grid.physical_invariants(), grid.weights, flat
        )

    def entropy_pairing(
        self, f: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        f, _ = self._remove_optional_channel(f)
        q, _ = self._remove_optional_channel(q)
        grid = self._stored_grid()
        flat_f = self._flatten_input(f, grid)
        flat_q = self._flatten_input(q, grid)
        floor = torch.finfo(flat_f.dtype).tiny
        return torch.sum(
            grid.weights[None] * torch.log(flat_f.clamp_min(floor)) * flat_q,
            dim=1,
        )

    def bgk_collision(self, f: torch.Tensor) -> torch.Tensor:
        f_grid, had_channel = self._remove_optional_channel(f)
        grid = self._stored_grid()
        flat = self._flatten_input(f_grid, grid)
        positive = flat.clamp_min(torch.finfo(flat.dtype).tiny)
        rho, velocity, temperature = weighted_moments(positive, grid)
        maxwellian = continuous_maxwellian(
            grid, rho, velocity, temperature
        )
        q = (
            self.bgk_rate
            * rho[:, None]
            * (maxwellian - positive)
        )
        return self._restore_shape(q, grid, had_channel)

    def forward(
        self,
        f: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ):
        f_grid, had_channel = self._remove_optional_channel(f)
        resolved_grid = self._stored_grid()
        q, auxiliary = self.forward_with_aux(f_grid, resolved_grid)
        shaped_q = self._restore_shape(q, resolved_grid, had_channel)
        if not return_diagnostics:
            return shaped_q

        flat = self._flatten_input(f_grid, resolved_grid)
        density_floor = auxiliary["rho"][:, None] * self.config.log_floor
        positive_f = torch.maximum(flat, density_floor)
        positive_m = torch.maximum(auxiliary["maxwellian"], density_floor)
        bgk = (
            self.bgk_rate
            * auxiliary["rho"][:, None]
            * (positive_m - positive_f)
        )
        diagnostics: dict[str, torch.Tensor] = {
            **auxiliary,
            "bgk_collision": self._restore_shape(
                bgk, resolved_grid, had_channel
            ),
        }
        return shaped_q, diagnostics
