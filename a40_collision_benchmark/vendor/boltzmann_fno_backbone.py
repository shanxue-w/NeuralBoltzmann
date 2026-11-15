"""Global dual-head FNO backbone for structured Boltzmann surrogates."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

from boltzmann_spectral import ComplexSpectralConv3d
from boltzmann_triton import fused_add_gelu, fused_add_gelu_cuda


class BoltzmannFNOOutput(NamedTuple):
    """Dense latent fields consumed by the structured collision layer."""

    gain_logits: torch.Tensor
    rate_logits: torch.Tensor


class BoltzmannFNOReadoutTerms(NamedTuple):
    """Separate spectral and local terms of the fused dual-head readout."""

    spectral: torch.Tensor
    local: torch.Tensor


class BoltzmannFNOBackbone(nn.Module):
    """A shared global Fourier trunk with two zero-initialized scalar heads.

    Parameters
    ----------
    in_channels:
        Number of channel-first physical input fields.
    modes:
        Retained Fourier modes along each velocity dimension.
    width:
        Hidden channel width shared by all spectral layers.
    layers:
        Number of spectral-plus-pointwise mixing layers.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        modes: int = 8,
        width: int = 36,
        layers: int = 4,
        use_fused_forward: bool = True,
        use_fused_add_gelu: bool = True,
        use_triton_scatter: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.modes = int(modes)
        self.width = int(width)
        self.layers = int(layers)
        self.use_fused_forward = bool(use_fused_forward)
        self.use_fused_add_gelu = bool(use_fused_add_gelu)
        self.use_triton_scatter = bool(use_triton_scatter)

        self.lift = nn.Conv3d(self.in_channels, self.width, kernel_size=1)
        spectral_layers = [
            ComplexSpectralConv3d(self.width, self.width, self.modes, self.modes, self.modes)
            for _ in range(self.layers)
        ]
        self.spectral_layers = nn.ModuleList(spectral_layers)
        self.pointwise_layers = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(self.layers)])
        self.gain_head = nn.Conv3d(self.width, 1, kernel_size=1)
        self.rate_head = nn.Conv3d(self.width, 1, kernel_size=1)
        self._fused_cache_key = None
        self._fused_spectral_weight = None
        self._fused_local_weight = None
        self._fused_local_bias = None
        self.reset_output_heads()

    def reset_output_heads(self) -> None:
        """Make the surrounding collision model start from its BGK baseline."""

        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)
        nn.init.zeros_(self.rate_head.weight)
        nn.init.zeros_(self.rate_head.bias)

    def _clear_fused_cache(self) -> None:
        self._fused_cache_key = None
        self._fused_spectral_weight = None
        self._fused_local_weight = None
        self._fused_local_bias = None

    def _apply(self, fn):
        result = super()._apply(fn)
        result._clear_fused_cache()
        return result

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self._clear_fused_cache()

    def _readout_cache_key(self) -> tuple[tuple[object, ...], ...]:
        last_spectral = self.spectral_layers[-1]
        last_pointwise = self.pointwise_layers[-1]
        parameters = (
            last_spectral.weight, last_pointwise.weight, last_pointwise.bias,
            self.gain_head.weight, self.gain_head.bias, self.rate_head.weight, self.rate_head.bias,
        )
        return tuple((p._version, p.data_ptr(), p.device, p.dtype) for p in parameters)

    def _compose_readout(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_weight = torch.cat((self.gain_head.weight, self.rate_head.weight), dim=0)[:, :, 0, 0, 0]
        head_bias = torch.cat((self.gain_head.bias, self.rate_head.bias), dim=0)
        last_spectral = self.spectral_layers[-1]
        fused_spectral_weight = torch.einsum(
            "qioxyz,ao->qiaxyz", last_spectral.weight, head_weight.to(dtype=last_spectral.weight.dtype)
        )
        last_pointwise = self.pointwise_layers[-1]
        pointwise_weight = last_pointwise.weight[:, :, 0, 0, 0]
        fused_local_weight = torch.einsum("ao,oi->ai", head_weight, pointwise_weight)[:, :, None, None, None]
        fused_local_bias = torch.einsum("ao,o->a", head_weight, last_pointwise.bias) + head_bias
        return fused_spectral_weight, fused_local_weight, fused_local_bias

    def _fused_readout(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = self._readout_cache_key()
        if cache_key != self._fused_cache_key:
            with torch.no_grad():
                self._fused_spectral_weight, self._fused_local_weight, self._fused_local_bias = self._compose_readout()
            self._fused_cache_key = cache_key
        return self._fused_spectral_weight, self._fused_local_weight, self._fused_local_bias

    def _forward_reference(self, x: torch.Tensor) -> BoltzmannFNOOutput:
        x = self.lift(x)
        for index, (spectral, pointwise) in enumerate(zip(self.spectral_layers, self.pointwise_layers, strict=True)):
            x = spectral(x) + pointwise(x)
            if index + 1 != self.layers:
                x = F.gelu(x)
        gain_logits = self.gain_head(x).squeeze(1)
        rate_logits = self.rate_head(x).squeeze(1)
        return BoltzmannFNOOutput(gain_logits, rate_logits)

    def fused_readout_terms(self, x: torch.Tensor) -> BoltzmannFNOReadoutTerms:
        """Return the two last-readout terms for a caller-specific fusion."""

        x = self.lift(x)
        for spectral, pointwise in zip(self.spectral_layers[:-1], self.pointwise_layers[:-1], strict=True):
            spectral_output = spectral(x)
            pointwise_output = pointwise(x)
            x = (fused_add_gelu(spectral_output, pointwise_output) if self.use_fused_add_gelu
                 else F.gelu(spectral_output + pointwise_output))

        fused_spectral_weight, fused_local_weight, fused_local_bias = self._fused_readout()
        spectral_logits = self.spectral_layers[-1].forward_with_weight(
            x, fused_spectral_weight, use_triton_scatter=self.use_triton_scatter
        )
        local_logits = F.conv3d(x, fused_local_weight, fused_local_bias)
        return BoltzmannFNOReadoutTerms(spectral_logits, local_logits)

    def fused_readout_terms_cuda(self, x: torch.Tensor) -> BoltzmannFNOReadoutTerms:
        """Fixed production CUDA path without per-layer dispatch checks."""

        x = self.lift(x)
        for spectral, pointwise in zip(self.spectral_layers[:-1], self.pointwise_layers[:-1], strict=True):
            x = fused_add_gelu_cuda(spectral(x), pointwise(x))

        fused_spectral_weight, fused_local_weight, fused_local_bias = self._fused_readout()
        spectral_logits = self.spectral_layers[-1].forward_with_weight(
            x, fused_spectral_weight, use_triton_scatter=False
        )
        local_logits = F.conv3d(x, fused_local_weight, fused_local_bias)
        return BoltzmannFNOReadoutTerms(spectral_logits, local_logits)

    def _forward_fused(self, x: torch.Tensor) -> BoltzmannFNOOutput:
        terms = self.fused_readout_terms(x)
        logits = terms.spectral + terms.local
        return BoltzmannFNOOutput(logits[:, 0], logits[:, 1])

    def forward(self, x: torch.Tensor) -> BoltzmannFNOOutput:
        if self.use_fused_forward and not self.training and not torch.is_grad_enabled():
            return self._forward_fused(x)
        return self._forward_reference(x)


__all__ = ["BoltzmannFNOBackbone", "BoltzmannFNOOutput", "BoltzmannFNOReadoutTerms"]
