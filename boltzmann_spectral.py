"""Shared complex spectral layer used by the active gain-gated FNO."""

from __future__ import annotations

import torch
import torch._dynamo as dynamo
from torch import nn

from boltzmann_triton import scatter_corner_blocks

def real_parameter_count(module: nn.Module) -> int:
    """Count trainable real scalar degrees of freedom in a mixed-dtype model."""
    return sum(parameter.numel() * (2 if parameter.is_complex() else 1)
               for parameter in module.parameters() if parameter.requires_grad)


class ComplexSpectralConv3d(nn.Module):
    """3D FNO spectral layer for real inputs on a periodic grid.

    Input shape is ``(B, C_in, N1, N2, N3)`` and output shape is
    ``(B, C_out, N1, N2, N3)``. The layer learns the low modes of the real FFT
    spectrum, including the four signed-mode corner blocks in the first two
    Fourier directions.
    """

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int,
        *, use_triton_scatter: bool = False,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.use_triton_scatter = bool(use_triton_scatter)
        scale = 1.0 / (in_channels * out_channels)
        weight_shape = (4, in_channels, out_channels, modes1, modes2, modes3)
        self.weight = nn.Parameter(torch.complex(scale * torch.randn(weight_shape), scale * torch.randn(weight_shape)))

    def _apply(self, fn):
        """Apply real-model dtype conversions without discarding Im(weight)."""

        weight = self._parameters.pop("weight")
        super()._apply(fn)
        converted = torch.complex(fn(weight.real), fn(weight.imag))
        self._parameters["weight"] = nn.Parameter(converted, requires_grad=weight.requires_grad)
        return self

    @staticmethod
    def _mul_with_weight(
        x_ft: torch.Tensor, weight: torch.Tensor, block: int, m1: int, m2: int, m3: int
    ) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", x_ft, weight[block, :, :, :m1, :m2, :m3])

    @dynamo.disable
    def _forward_with_weight(
        self, x: torch.Tensor, weight: torch.Tensor, use_triton_scatter: bool
    ) -> torch.Tensor:
        B, _, N1, N2, N3 = x.shape
        out_channels = int(weight.shape[2])
        K3 = N3 // 2 + 1
        m1 = min(self.modes1, int(weight.shape[3]), N1 // 2)
        m2 = min(self.modes2, int(weight.shape[4]), N2 // 2)
        m3 = min(self.modes3, int(weight.shape[5]), K3)

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        corner_blocks = (
            self._mul_with_weight(x_ft[:, :, :m1, :m2, :m3], weight, 0, m1, m2, m3),
            self._mul_with_weight(x_ft[:, :, -m1:, :m2, :m3], weight, 1, m1, m2, m3),
            self._mul_with_weight(x_ft[:, :, :m1, -m2:, :m3], weight, 2, m1, m2, m3),
            self._mul_with_weight(x_ft[:, :, -m1:, -m2:, :m3], weight, 3, m1, m2, m3),
        )
        if use_triton_scatter:
            out_ft = scatter_corner_blocks(torch.stack(corner_blocks, dim=1), full_shape=(N1, N2, K3))
        else:
            out_ft = x_ft.new_zeros(B, out_channels, N1, N2, K3)
            out_ft[:, :, :m1, :m2, :m3] = corner_blocks[0]
            out_ft[:, :, -m1:, :m2, :m3] = corner_blocks[1]
            out_ft[:, :, :m1, -m2:, :m3] = corner_blocks[2]
            out_ft[:, :, -m1:, -m2:, :m3] = corner_blocks[3]

        return torch.fft.irfftn(out_ft, s=(N1, N2, N3), dim=(-3, -2, -1))

    def forward_with_weight(
        self, x: torch.Tensor, weight: torch.Tensor, *, use_triton_scatter: bool | None = None
    ) -> torch.Tensor:
        """Apply the layer with a compatible inference-time weight tensor."""

        scatter = self.use_triton_scatter if use_triton_scatter is None else bool(use_triton_scatter)
        return self._forward_with_weight(x, weight, scatter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_with_weight(x, self.weight, self.use_triton_scatter)
