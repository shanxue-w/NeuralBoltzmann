"""Gate-FNO with one linear five-moment correction of ``G - nu*f``."""

from __future__ import annotations

import math

import torch
from torch import nn

from boltzmann_discrete_maxwellian import DiscreteMaxwellian
from boltzmann_fno_backbone import BoltzmannFNOBackbone
from boltzmann_triton import (
    fused_gain_gate_readout_cuda,
    is_triton_projection_available,
    linear_projection_statistics_cuda,
    triton_cuda_inference_available,
)


class LinearConservativeGainLossCollision(nn.Module):
    """Positive gain--loss proposal followed by one linear moment solve."""

    def __init__(
        self, *, n: int, L: float, rate_min: float, rate_max: float,
        entropic_bgk_rate: float, log_floor: float = 1.0e-30,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.L = float(L)
        self.dv = 2.0 * self.L / self.n
        self.dv3 = self.dv**3
        self.rate_min = float(rate_min)
        self.rate_max = float(rate_max)
        self.entropic_bgk_rate = float(entropic_bgk_rate)
        self.log_floor = float(log_floor)

        v = (torch.arange(self.n, dtype=torch.float64) - self.n // 2) * self.dv
        vx, vy, vz = torch.meshgrid(v, v, v, indexing="ij")
        v2 = vx.square() + vy.square() + vz.square()
        psi = torch.stack(
            (torch.ones_like(vx), vx / self.L, vy / self.L, vz / self.L, v2 / self.L**2), dim=-1
        ).reshape(-1, 5)
        self.register_buffer("psi", psi, persistent=False)

    @staticmethod
    def _as_grid(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
        had_channel = value.ndim == 5
        return (value[:, 0] if had_channel else value), had_channel

    @staticmethod
    def _broadcast_grid(value: torch.Tensor | float, *, reference: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if tensor.ndim == 5:
            tensor = tensor[:, 0]
        return torch.broadcast_to(tensor, reference.shape)

    def rate_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self.rate_min + (self.rate_max - self.rate_min) * torch.sigmoid(logits)

    def moments(self, value: torch.Tensor) -> torch.Tensor:
        grid, _ = self._as_grid(value)
        flat = grid.reshape(grid.shape[0], -1)
        psi = self.psi.to(grid)
        return self.dv3 * (flat[:, :, None] * psi[None]).sum(dim=1)

    def entropy_pairing(self, f: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        f_grid, _ = self._as_grid(f)
        q_grid, _ = self._as_grid(q)
        density = self.dv3 * f_grid.sum(dim=(-3, -2, -1))
        normalized = f_grid / density.clamp_min(torch.finfo(f_grid.dtype).tiny)[:, None, None, None]
        return self.dv3 * (normalized.clamp_min(self.log_floor).log() * q_grid).sum(dim=(-3, -2, -1))

    def _linear_shift(
        self, raw_q: torch.Tensor, f: torch.Tensor, density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``lambda.psi`` and rows for which the 5x5 solve failed."""

        batch = f.shape[0]
        psi = self.psi.to(f)
        same_dtype = f.dtype == raw_q.dtype == density.dtype
        same_device = f.device == raw_q.device == density.device
        contiguous = f.is_contiguous() and raw_q.is_contiguous()
        inference = not self.training and not torch.is_grad_enabled() and is_triton_projection_available(raw_q)
        use_cuda_statistics = inference and f.is_cuda and same_dtype and same_device and contiguous
        if use_cuda_statistics:
            raw_moments, gram_upper = linear_projection_statistics_cuda(
                f, raw_q, density.contiguous(), n=self.n, dv3=self.dv3
            )
            (g00, g01, g02, g03, g04, g11, g12, g13, g14,
             g22, g23, g24, g33, g34, g44) = gram_upper.unbind(dim=1)
            gram = torch.stack((
                torch.stack((g00, g01, g02, g03, g04), dim=-1),
                torch.stack((g01, g11, g12, g13, g14), dim=-1),
                torch.stack((g02, g12, g22, g23, g24), dim=-1),
                torch.stack((g03, g13, g23, g33, g34), dim=-1),
                torch.stack((g04, g14, g24, g34, g44), dim=-1),
            ), dim=-2)
        else:
            normalized_f = (f / density[:, None, None, None]).reshape(batch, -1)
            gram = self.dv3 * torch.einsum("bp,pi,pj->bij", normalized_f, psi, psi)
            raw_moments = self.moments(raw_q)
        right_hand_side = -raw_moments / density[:, None]
        multiplier, info = torch.linalg.solve_ex(gram, right_hand_side[..., None], check_errors=False)
        multiplier = multiplier.squeeze(-1)
        shift = multiplier @ psi.T
        failed = (info != 0) | (~torch.isfinite(shift).all(dim=-1))
        shift = torch.where(failed[:, None], torch.zeros_like(shift), shift)
        return shift.reshape_as(f), failed

    @staticmethod
    def _positivity_dt(loss_coefficient: torch.Tensor) -> torch.Tensor:
        maximum = loss_coefficient.clamp_min(0.0).reshape(loss_coefficient.shape[0], -1).amax(dim=-1)
        return torch.where(maximum > 0.0, maximum.reciprocal(), torch.full_like(maximum, torch.inf))

    def forward(
        self,
        f: torch.Tensor,
        maxwellian: torch.Tensor,
        bounded_rate: torch.Tensor,
        gain_deviation: torch.Tensor,
        tail_cap: torch.Tensor,
        *,
        density: torch.Tensor,
        frequency_scale: torch.Tensor | float | None = None,
        enforce_entropy: bool = True,
        return_positivity_dt: bool = False,
    ):
        f_grid, had_channel = self._as_grid(f)
        m_grid, _ = self._as_grid(maxwellian)
        rate_input = self._broadcast_grid(bounded_rate, reference=f_grid)
        gain_input = self._broadcast_grid(gain_deviation, reference=f_grid)
        cap = self._broadcast_grid(tail_cap, reference=f_grid)
        rate = rate_input.clamp(min=self.rate_min, max=self.rate_max)

        if frequency_scale is None:
            physical_scale = torch.ones_like(density)
        else:
            physical_scale = torch.as_tensor(frequency_scale, device=f_grid.device, dtype=f_grid.dtype)
            physical_scale = physical_scale.reshape(-1).expand_as(density)
        scaled_density = density * physical_scale
        nu = scaled_density[:, None, None, None] * rate
        loss = nu * f_grid

        baseline_gain = nu * m_grid
        positive_cap = cap > 0.0
        baseline_probability = baseline_gain / cap.clamp_min(torch.finfo(f_grid.dtype).tiny)
        fallback_probability = torch.full_like(baseline_probability, 0.5)
        baseline_probability = torch.where(positive_cap, baseline_probability, fallback_probability)
        baseline_probability = baseline_probability.clamp(
            min=torch.finfo(f_grid.dtype).tiny, max=1.0 - 8.0 * torch.finfo(f_grid.dtype).eps)
        raw_gain = cap * torch.sigmoid(torch.logit(baseline_probability) + gain_input)
        raw_q = raw_gain - loss
        shift, failed = self._linear_shift(raw_q, f_grid, density)
        q0 = raw_q + f_grid * shift
        q0 = torch.where(failed[:, None, None, None], torch.zeros_like(q0), q0)

        if enforce_entropy:
            bgk_frequency = scaled_density[:, None, None, None] * self.entropic_bgk_rate
            q_bgk = bgk_frequency * (m_grid - f_grid)
            entropy_before = self.entropy_pairing(f_grid, q0)
            positive_pairing = entropy_before > 0.0
            entropy_bgk = self.entropy_pairing(f_grid, q_bgk)
            denominator = torch.where(
                positive_pairing, entropy_before - entropy_bgk, torch.ones_like(entropy_before)
            )
            gamma = torch.where(
                positive_pairing, entropy_before / denominator, torch.zeros_like(entropy_before)
            ).clamp(min=0.0, max=1.0)
            learned_fraction = 1.0 - gamma
            q = learned_fraction[:, None, None, None] * q0 + gamma[:, None, None, None] * q_bgk
        else:
            gamma = torch.zeros_like(density)
            q = q0
            learned_fraction = torch.ones_like(density)
            bgk_frequency = None
        output = q[:, None] if had_channel else q
        if not return_positivity_dt:
            return output

        if bgk_frequency is None:
            effective_loss_coefficient = nu - shift
        else:
            learned_loss = learned_fraction[:, None, None, None] * (nu - shift)
            bgk_loss = gamma[:, None, None, None] * bgk_frequency
            effective_loss_coefficient = learned_loss + bgk_loss
        positivity_dt = self._positivity_dt(effective_loss_coefficient)
        positivity_dt = torch.where(failed, torch.full_like(positivity_dt, torch.inf), positivity_dt)

        return output, positivity_dt


class SimplifiedGateBoltzmannQ(nn.Module):
    """Dual-head Gate-FNO with conservation moved after raw gain--loss."""

    def __init__(
        self,
        *,
        n: int = 64,
        L: float = 10.0,
        fno_modes: int = 8,
        fno_width: int = 36,
        fno_layers: int = 3,
        rate_min: float = 0.02,
        rate_max: float = 8.0,
        initial_rate: float = 0.5,
        entropic_bgk_rate: float = 0.5,
        tail_exponent: float = 0.1,
        gain_deviation_limit: float = 10.0,
        rate_deviation_limit: float = 10.0,
        equilibrium_gate_scale: float = 0.0,
        enforce_entropy: bool = True,
        feature_floor: float = 1.0e-12,
        log_floor: float = 1.0e-30,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.L = float(L)
        self.dv = 2.0 * self.L / self.n
        self.dv3 = self.dv**3
        self.rate_min = float(rate_min)
        self.rate_max = float(rate_max)
        self.initial_rate = float(initial_rate)
        self.entropic_bgk_rate = float(entropic_bgk_rate)
        self.tail_exponent = float(tail_exponent)
        self.gain_deviation_limit = float(gain_deviation_limit)
        self.rate_deviation_limit = float(rate_deviation_limit)
        self.equilibrium_gate_scale = float(equilibrium_gate_scale)
        self.enforce_entropy = bool(enforce_entropy)
        self.feature_floor = float(feature_floor)
        self.log_floor = float(log_floor)

        self.maxwellian_solver = DiscreteMaxwellian(n=self.n, L=self.L)
        self.backbone = BoltzmannFNOBackbone(
            in_channels=7, modes=fno_modes, width=fno_width, layers=fno_layers
        )
        self.structure = LinearConservativeGainLossCollision(
            n=self.n, L=self.L, rate_min=self.rate_min, rate_max=self.rate_max,
            entropic_bgk_rate=self.entropic_bgk_rate, log_floor=self.log_floor)

        rate_fraction = (self.initial_rate - self.rate_min) / (self.rate_max - self.rate_min)
        initial_logit = math.log(rate_fraction / (1.0 - rate_fraction))
        self.initial_rate_logit_value = initial_logit
        self.register_buffer("initial_rate_logit", torch.tensor(initial_logit, dtype=torch.float64), persistent=False)
        proxy = torch.exp(-0.5 * self.maxwellian_solver.v2)
        proxy = proxy / (self.dv3 * proxy.sum())
        self.register_buffer("vacuum_proxy", proxy, persistent=False)
        proxy_moments = self.maxwellian_solver.moments(proxy[None])[0]
        self.register_buffer("vacuum_proxy_moments", proxy_moments, persistent=False)

        self.model_config = {
            "n": self.n,
            "L": self.L,
            "fno_modes": int(fno_modes),
            "fno_width": int(fno_width),
            "fno_layers": int(fno_layers),
            "rate_min": self.rate_min,
            "rate_max": self.rate_max,
            "initial_rate": self.initial_rate,
            "entropic_bgk_rate": self.entropic_bgk_rate,
            "tail_exponent": self.tail_exponent,
            "gain_deviation_limit": self.gain_deviation_limit,
            "rate_deviation_limit": self.rate_deviation_limit,
            "equilibrium_gate_scale": self.equilibrium_gate_scale,
            "enforce_entropy": self.enforce_entropy,
            "feature_floor": self.feature_floor,
            "log_floor": self.log_floor,
        }

    @staticmethod
    def _as_grid(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
        had_channel = value.ndim == 5
        return (value[:, 0] if had_channel else value), had_channel

    def distribution_moments(self, value: torch.Tensor) -> torch.Tensor:
        grid, _ = self._as_grid(value)
        return self.structure.moments(grid)

    def collision_moments(self, q: torch.Tensor) -> torch.Tensor:
        return self.distribution_moments(q)

    def entropy_pairing(self, f: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self.structure.entropy_pairing(f, q)

    def _physical_state(
        self, grid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        physical = grid.clamp_min(0.0)
        density = self.dv3 * physical.sum(dim=(-3, -2, -1))
        vacuum = density <= 0.0
        proxy = self.vacuum_proxy.to(physical).expand_as(physical)
        solver_input = torch.where(vacuum[:, None, None, None], proxy, physical)
        maxwellian, solver_status = self.maxwellian_solver(solver_input, return_diagnostics=True)
        finite_maxwellian = torch.isfinite(maxwellian).reshape(
            maxwellian.shape[0], -1
        ).all(dim=-1)
        active = (~vacuum) & solver_status["converged"] & finite_maxwellian
        physical = torch.where(active[:, None, None, None], physical, proxy)
        maxwellian = torch.where(active[:, None, None, None], maxwellian, proxy)
        density = torch.where(active, density, torch.ones_like(density))
        moments = torch.where(active[:, None], solver_status["physical_moments"].to(grid),
                              self.vacuum_proxy_moments.to(grid)[None])
        return physical, maxwellian, density, active, moments

    def _state_features(
        self,
        f: torch.Tensor,
        maxwellian: torch.Tensor,
        density: torch.Tensor,
        moments: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        velocity = self.L * moments[:, 1:4] / density[:, None]
        mean_v2 = self.L * self.L * moments[:, 4] / density
        temperature = ((mean_v2 - velocity.square().sum(dim=-1)) / 3.0).clamp_min(1.0e-8)

        amplitude = torch.maximum(f, maxwellian).amax(dim=(-3, -2, -1), keepdim=True)
        amplitude = amplitude.clamp_min(torch.finfo(f.dtype).tiny)
        floor = self.feature_floor * amplitude
        f_shape = f / amplitude
        maxwellian_shape = maxwellian / amplitude
        residual = (f - maxwellian) / (f + maxwellian + floor)

        sqrt_temperature = torch.sqrt(temperature)[:, None, None, None]
        vx = self.maxwellian_solver.vx.to(f)[None]
        vy = self.maxwellian_solver.vy.to(f)[None]
        vz = self.maxwellian_solver.vz.to(f)[None]
        cx = (vx - velocity[:, 0, None, None, None]) / sqrt_temperature
        cy = (vy - velocity[:, 1, None, None, None]) / sqrt_temperature
        cz = (vz - velocity[:, 2, None, None, None]) / sqrt_temperature
        c2 = cx.square() + cy.square() + cz.square()
        one_plus_c2 = 1.0 + c2
        coordinate_scale = torch.sqrt(one_plus_c2)
        radial = c2 / one_plus_c2
        features = torch.stack((f_shape, maxwellian_shape, residual, cx / coordinate_scale,
                                cy / coordinate_scale, cz / coordinate_scale, radial), dim=1)

        flat_f = f.reshape(f.shape[0], -1)
        flat_m = maxwellian.reshape(maxwellian.shape[0], -1)
        distance = torch.linalg.vector_norm(flat_f - flat_m, dim=-1)
        scale = torch.linalg.vector_norm(flat_f, dim=-1) + torch.linalg.vector_norm(flat_m, dim=-1)
        nonequilibrium = distance / scale.clamp_min(torch.finfo(f.dtype).tiny)
        inverse_density = density.reciprocal()[:, None, None, None]
        base_normalized = torch.maximum(f * inverse_density, maxwellian * inverse_density)
        return features, nonequilibrium, c2, base_normalized

    def _tail_cap(
        self, base_normalized: torch.Tensor, c2: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        log_weighted = torch.where(base_normalized > 0.0,
                                   base_normalized.log() + self.tail_exponent * c2,
                                   torch.full_like(base_normalized, -torch.inf))
        log_amplitude = log_weighted.amax(dim=(-3, -2, -1), keepdim=True)
        gaussian_envelope = torch.exp(log_amplitude - self.tail_exponent * c2)
        required_rate = max(self.rate_max, self.entropic_bgk_rate)
        density_scale = density[:, None, None, None].square()
        return required_rate * density_scale * (base_normalized + gaussian_envelope)

    def _forward_core(
        self, f: torch.Tensor, *, return_positivity_dt: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid, had_channel = self._as_grid(f)
        physical, maxwellian, density, active, moments = self._physical_state(grid)
        features, nonequilibrium, c2, base_normalized = self._state_features(
            physical, maxwellian, density, moments
        )

        gate = (nonequilibrium / (nonequilibrium + self.equilibrium_gate_scale)
                if self.equilibrium_gate_scale > 0.0 else nonequilibrium)
        gate = gate * active.to(gate.dtype)
        gate_grid = gate[:, None, None, None]
        use_cuda_inference = (not self.training and not torch.is_grad_enabled()
                              and triton_cuda_inference_available(features))
        if use_cuda_inference:
            readout_terms = self.backbone.fused_readout_terms_cuda(features)
            gain_deviation, bounded_rate = fused_gain_gate_readout_cuda(
                readout_terms.spectral,
                readout_terms.local,
                gate_grid,
                gain_limit=self.gain_deviation_limit,
                rate_limit=self.rate_deviation_limit,
                initial_rate_logit=self.initial_rate_logit_value,
                rate_min=self.rate_min,
                rate_max=self.rate_max,
            )
        else:
            raw = self.backbone(features)
            gain_argument = gate_grid * raw.gain_logits / self.gain_deviation_limit
            rate_argument = gate_grid * raw.rate_logits / self.rate_deviation_limit
            gain_deviation = self.gain_deviation_limit * torch.tanh(gain_argument)
            rate_deviation = self.rate_deviation_limit * torch.tanh(rate_argument)
            rate_logits = self.initial_rate_logit.to(grid) + rate_deviation
            bounded_rate = self.structure.rate_from_logits(rate_logits)
        tail_cap = self._tail_cap(base_normalized, c2, density)

        result = self.structure(
            physical[:, None] if had_channel else physical,
            maxwellian[:, None] if had_channel else maxwellian,
            bounded_rate[:, None] if had_channel else bounded_rate,
            gain_deviation[:, None] if had_channel else gain_deviation,
            tail_cap[:, None] if had_channel else tail_cap,
            density=density,
            enforce_entropy=self.enforce_entropy,
            return_positivity_dt=return_positivity_dt,
        )
        if return_positivity_dt:
            q, positivity_dt = result
        else:
            q = result
            positivity_dt = None
        q_grid, _ = self._as_grid(q)
        q_grid = torch.where(active[:, None, None, None], q_grid, torch.zeros_like(q_grid))
        if not return_positivity_dt:
            return q_grid[:, None] if had_channel else q_grid, None
        positivity_dt = torch.where(active, positivity_dt, torch.full_like(positivity_dt, torch.inf))
        return (q_grid[:, None] if had_channel else q_grid), positivity_dt

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        q, _ = self._forward_core(f, return_positivity_dt=False)
        return q

    def forward_with_positivity_dt(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, positivity_dt = self._forward_core(f, return_positivity_dt=True)
        return q, positivity_dt


def load_simplified_gate_model(
    checkpoint: dict[str, object], *, n: int | None = None,
) -> tuple[SimplifiedGateBoltzmannQ, dict[str, object]]:
    config = dict(checkpoint["model_config"])
    config["n"] = int(config["n"] if n is None else n)
    config.pop("structure_variant", None)
    config.pop("uses_log_input_features", None)
    model = SimplifiedGateBoltzmannQ(**config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, config


__all__ = [
    "LinearConservativeGainLossCollision",
    "SimplifiedGateBoltzmannQ",
    "load_simplified_gate_model",
]
