"""Triton kernels used by the structured Boltzmann FNO."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - exercised only without Triton.
    triton = None
    tl = None
    libdevice = None


def is_triton_projection_available(value: torch.Tensor | None = None) -> bool:
    """Return whether the fused projection kernel can run for ``value``."""
    if triton is None:
        return False
    if value is None:
        return torch.cuda.is_available()
    return value.is_cuda and value.dtype in (torch.float32, torch.float64) and value.is_contiguous()


if triton is not None:

    @triton.jit
    def _projection_statistics_reduce_kernel(
        partial_ptr, output_ptr, chunks: tl.constexpr, reduction_size: tl.constexpr, dv3: tl.constexpr
    ):
        program = tl.program_id(0)
        batch_index = program // 20
        statistic_index = program - batch_index * 20
        offsets = tl.arange(0, reduction_size)
        mask = offsets < chunks
        partial_offset = (batch_index * chunks + offsets) * 20 + statistic_index
        values = tl.load(partial_ptr + partial_offset, mask=mask, other=0.0)
        output_offset = batch_index * 20 + statistic_index
        tl.store(output_ptr + output_offset, tl.sum(values) * dv3)

    @triton.jit
    def _linear_projection_statistics_partial_kernel(
        f_ptr,
        raw_q_ptr,
        density_ptr,
        partial_ptr,
        n: tl.constexpr,
        n2: tl.constexpr,
        points: tl.constexpr,
        chunks: tl.constexpr,
        is_float64: tl.constexpr,
        block_size: tl.constexpr,
    ):
        program = tl.program_id(0)
        batch_index = program // chunks
        chunk_index = program - batch_index * chunks
        offsets = chunk_index * block_size + tl.arange(0, block_size)
        mask = offsets < points
        flat_offsets = batch_index * points + offsets

        f_value = tl.load(f_ptr + flat_offsets, mask=mask, other=0.0)
        raw_q = tl.load(raw_q_ptr + flat_offsets, mask=mask, other=0.0)
        density = tl.load(density_ptr + batch_index)
        normalized_f = f_value / density

        ix = offsets // n2
        remainder = offsets - ix * n2
        iy = remainder // n
        iz = remainder - iy * n
        half = n // 2
        if is_float64:
            x = (2.0 * (ix - half)).to(tl.float64) / float(n)
            y = (2.0 * (iy - half)).to(tl.float64) / float(n)
            z = (2.0 * (iz - half)).to(tl.float64) / float(n)
        else:
            x = (2.0 * (ix - half)).to(tl.float32) / float(n)
            y = (2.0 * (iy - half)).to(tl.float32) / float(n)
            z = (2.0 * (iz - half)).to(tl.float32) / float(n)
        energy = x * x + y * y + z * z

        output_base = (batch_index * chunks + chunk_index) * 20
        tl.store(partial_ptr + output_base, tl.sum(raw_q, axis=0))
        tl.store(partial_ptr + output_base + 1, tl.sum(raw_q * x, axis=0))
        tl.store(partial_ptr + output_base + 2, tl.sum(raw_q * y, axis=0))
        tl.store(partial_ptr + output_base + 3, tl.sum(raw_q * z, axis=0))
        tl.store(partial_ptr + output_base + 4, tl.sum(raw_q * energy, axis=0))

        tl.store(partial_ptr + output_base + 5, tl.sum(normalized_f, axis=0))
        tl.store(partial_ptr + output_base + 6, tl.sum(normalized_f * x, axis=0))
        tl.store(partial_ptr + output_base + 7, tl.sum(normalized_f * y, axis=0))
        tl.store(partial_ptr + output_base + 8, tl.sum(normalized_f * z, axis=0))
        tl.store(partial_ptr + output_base + 9, tl.sum(normalized_f * energy, axis=0))
        tl.store(partial_ptr + output_base + 10, tl.sum(normalized_f * x * x, axis=0))
        tl.store(partial_ptr + output_base + 11, tl.sum(normalized_f * x * y, axis=0))
        tl.store(partial_ptr + output_base + 12, tl.sum(normalized_f * x * z, axis=0))
        tl.store(partial_ptr + output_base + 13, tl.sum(normalized_f * x * energy, axis=0))
        tl.store(partial_ptr + output_base + 14, tl.sum(normalized_f * y * y, axis=0))
        tl.store(partial_ptr + output_base + 15, tl.sum(normalized_f * y * z, axis=0))
        tl.store(partial_ptr + output_base + 16, tl.sum(normalized_f * y * energy, axis=0))
        tl.store(partial_ptr + output_base + 17, tl.sum(normalized_f * z * z, axis=0))
        tl.store(partial_ptr + output_base + 18, tl.sum(normalized_f * z * energy, axis=0))
        tl.store(partial_ptr + output_base + 19, tl.sum(normalized_f * energy * energy, axis=0))



def linear_projection_statistics_cuda(
    f: torch.Tensor, raw_q: torch.Tensor, density: torch.Tensor,
    *, n: int, dv3: float, block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Production reductions for the simplified linear moment projection."""

    points = n**3
    batch = f.shape[0]
    chunks = triton.cdiv(points, block_size)
    partial = torch.empty((batch, chunks, 20), device=f.device, dtype=f.dtype)
    output = torch.empty((batch, 20), device=f.device, dtype=f.dtype)
    _linear_projection_statistics_partial_kernel[(batch * chunks,)](
        f, raw_q, density, partial, n=n, n2=n * n, points=points, chunks=chunks,
        is_float64=f.dtype == torch.float64, block_size=block_size,
        num_warps=8 if block_size >= 512 else 4)
    reduction_size = triton.next_power_of_2(chunks)
    _projection_statistics_reduce_kernel[(batch * 20,)](
        partial, output, chunks=chunks, reduction_size=reduction_size, dv3=dv3,
        num_warps=8 if reduction_size >= 512 else 4)
    return output[:, :5], output[:, 5:]




if triton is not None:

    @triton.jit
    def _add_gelu_kernel(
        left_ptr, right_ptr, output_ptr, points: tl.constexpr, block_size: tl.constexpr
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < points
        value = tl.load(left_ptr + offsets, mask=mask, other=0.0)
        value += tl.load(right_ptr + offsets, mask=mask, other=0.0)
        result = 0.5 * value * (1.0 + tl.erf(value * 0.7071067811865476))
        tl.store(output_ptr + offsets, result, mask=mask)

    @triton.jit
    def _scatter_corner_blocks_kernel(
        blocks_ptr,
        output_ptr,
        complex_points: tl.constexpr,
        out_channels: tl.constexpr,
        n1: tl.constexpr,
        n2: tl.constexpr,
        k3_size: tl.constexpr,
        m1: tl.constexpr,
        m2: tl.constexpr,
        m3: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        output_mask = offsets < complex_points

        remainder = offsets
        k3 = remainder % k3_size
        remainder = remainder // k3_size
        k2 = remainder % n2
        remainder = remainder // n2
        k1 = remainder % n1
        remainder = remainder // n1
        channel = remainder % out_channels
        batch = remainder // out_channels

        negative_1 = k1 >= n1 - m1
        negative_2 = k2 >= n2 - m2
        valid = output_mask & ((k1 < m1) | negative_1) & ((k2 < m2) | negative_2) & (k3 < m3)
        local_1 = tl.where(negative_1, k1 - (n1 - m1), k1)
        local_2 = tl.where(negative_2, k2 - (n2 - m2), k2)
        corner = negative_1.to(tl.int32) + 2 * negative_2.to(tl.int32)
        source = (batch * 4 + corner) * out_channels + channel
        source = ((source * m1 + local_1) * m2 + local_2) * m3 + k3

        real = tl.load(blocks_ptr + 2 * source, mask=valid, other=0.0)
        imaginary = tl.load(
            blocks_ptr + 2 * source + 1, mask=valid, other=0.0
        )
        tl.store(output_ptr + 2 * offsets, real, mask=output_mask)
        tl.store(output_ptr + 2 * offsets + 1, imaginary, mask=output_mask)

    @triton.jit
    def _fused_gain_gate_readout_kernel(
        spectral_ptr,
        local_ptr,
        gate_ptr,
        gain_ptr,
        rate_ptr,
        points_per_batch: tl.constexpr,
        total_points: tl.constexpr,
        gain_limit: tl.constexpr,
        rate_limit: tl.constexpr,
        initial_rate_logit: tl.constexpr,
        rate_min: tl.constexpr,
        rate_span: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total_points
        batch = offsets // points_per_batch
        local_offset = offsets - batch * points_per_batch
        gain_offset = batch * (2 * points_per_batch) + local_offset
        rate_offset = gain_offset + points_per_batch
        gate_value = tl.load(gate_ptr + batch, mask=mask, other=0.0)

        raw_gain = tl.load(spectral_ptr + gain_offset, mask=mask, other=0.0)
        raw_gain += tl.load(local_ptr + gain_offset, mask=mask, other=0.0)
        raw_rate = tl.load(spectral_ptr + rate_offset, mask=mask, other=0.0)
        raw_rate += tl.load(local_ptr + rate_offset, mask=mask, other=0.0)
        gain = gain_limit * libdevice.tanh(gate_value * raw_gain / gain_limit)
        rate_deviation = rate_limit * libdevice.tanh(gate_value * raw_rate / rate_limit)
        rate = rate_min + rate_span * tl.sigmoid(initial_rate_logit + rate_deviation)
        tl.store(gain_ptr + offsets, gain, mask=mask)
        tl.store(rate_ptr + offsets, rate, mask=mask)


def triton_cuda_inference_available(value: torch.Tensor | None = None) -> bool:
    """Return whether the production float32 CUDA inference kernels are usable."""
    if triton is None:
        return False
    if value is None:
        return torch.cuda.is_available()
    return value.is_cuda and value.dtype == torch.float32 and value.is_contiguous()


def fused_add_gelu_cuda(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Production CUDA path for ``gelu(left + right)`` without dispatch checks."""

    output = torch.empty_like(left)
    points = left.numel()
    block_size = 256
    _add_gelu_kernel[(triton.cdiv(points, block_size),)](
        left, right, output, points=points, block_size=block_size, num_warps=4
    )
    return output


def fused_add_gelu(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Validated public helper; the production backbone calls the CUDA path."""

    use_triton = (triton_cuda_inference_available(left) and right.is_cuda and right.is_contiguous()
                  and right.dtype == torch.float32 and not torch.is_grad_enabled())
    if use_triton:
        return fused_add_gelu_cuda(left, right)
    return F.gelu(left + right)


def scatter_corner_blocks_reference(
    blocks: torch.Tensor, *, full_shape: tuple[int, int, int]
) -> torch.Tensor:
    batch, corners, out_channels, m1, m2, m3 = blocks.shape
    n1, n2, k3_size = full_shape
    output = blocks.new_zeros(batch, out_channels, n1, n2, k3_size)
    output[:, :, :m1, :m2, :m3] = blocks[:, 0]
    output[:, :, -m1:, :m2, :m3] = blocks[:, 1]
    output[:, :, :m1, -m2:, :m3] = blocks[:, 2]
    output[:, :, -m1:, -m2:, :m3] = blocks[:, 3]
    return output


def scatter_corner_blocks_cuda(
    blocks: torch.Tensor, *, full_shape: tuple[int, int, int]
) -> torch.Tensor:
    """Production complex64 CUDA scatter without backend checks."""

    batch, _, out_channels, m1, m2, m3 = blocks.shape
    n1, n2, k3_size = full_shape
    compact = torch.view_as_real(blocks.contiguous())
    output_real = torch.empty((batch, out_channels, n1, n2, k3_size, 2), device=blocks.device, dtype=torch.float32)
    complex_points = batch * out_channels * n1 * n2 * k3_size
    block_size = 256
    _scatter_corner_blocks_kernel[(triton.cdiv(complex_points, block_size),)](
        compact,
        output_real,
        complex_points=complex_points,
        out_channels=out_channels,
        n1=n1,
        n2=n2,
        k3_size=k3_size,
        m1=m1,
        m2=m2,
        m3=m3,
        block_size=block_size,
        num_warps=4,
    )
    return torch.view_as_complex(output_real)


def scatter_corner_blocks(blocks: torch.Tensor, *, full_shape: tuple[int, int, int]) -> torch.Tensor:
    """Validated scatter helper with a PyTorch reference fallback."""

    use_triton = (triton is not None and blocks.is_cuda and blocks.is_contiguous()
                  and blocks.dtype == torch.complex64 and not torch.is_grad_enabled())
    if use_triton:
        return scatter_corner_blocks_cuda(blocks, full_shape=full_shape)
    return scatter_corner_blocks_reference(blocks, full_shape=full_shape)


def fused_gain_gate_readout_cuda(
    spectral: torch.Tensor,
    local: torch.Tensor,
    gate: torch.Tensor,
    *,
    gain_limit: float,
    rate_limit: float,
    initial_rate_logit: float,
    rate_min: float,
    rate_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Production float32 CUDA gain-gate readout without dispatch checks."""

    batch = spectral.shape[0]
    points_per_batch = spectral[0, 0].numel()
    total_points = batch * points_per_batch
    gain = torch.empty_like(spectral[:, 0])
    rate = torch.empty_like(gain)
    block_size = 256
    _fused_gain_gate_readout_kernel[(triton.cdiv(total_points, block_size),)](
        spectral,
        local,
        gate.reshape(batch),
        gain,
        rate,
        points_per_batch=points_per_batch,
        total_points=total_points,
        gain_limit=float(gain_limit),
        rate_limit=float(rate_limit),
        initial_rate_logit=float(initial_rate_logit),
        rate_min=float(rate_min),
        rate_span=float(rate_max - rate_min),
        block_size=block_size,
        num_warps=4,
    )
    return gain, rate


def fused_gain_gate_readout(
    spectral: torch.Tensor,
    local: torch.Tensor,
    gate: torch.Tensor,
    *,
    gain_limit: float,
    rate_limit: float,
    initial_rate_logit: float,
    rate_min: float,
    rate_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validated helper; production gate inference calls the CUDA path."""

    use_triton = (triton_cuda_inference_available(spectral) and local.is_contiguous()
                  and local.dtype == torch.float32 and not torch.is_grad_enabled())
    if use_triton:
        return fused_gain_gate_readout_cuda(
            spectral, local, gate, gain_limit=gain_limit, rate_limit=rate_limit,
            initial_rate_logit=initial_rate_logit, rate_min=rate_min, rate_max=rate_max)
    logits = spectral + local
    gain = gain_limit * torch.tanh(gate * logits[:, 0] / gain_limit)
    rate_deviation = rate_limit * torch.tanh(gate * logits[:, 1] / rate_limit)
    rate = rate_min + (rate_max - rate_min) * torch.sigmoid(initial_rate_logit + rate_deviation)
    return gain, rate


__all__ = [
    "fused_add_gelu", "fused_add_gelu_cuda", "fused_gain_gate_readout", "fused_gain_gate_readout_cuda",
    "is_triton_projection_available", "linear_projection_statistics_cuda",
    "scatter_corner_blocks", "scatter_corner_blocks_cuda", "scatter_corner_blocks_reference",
    "triton_cuda_inference_available",
]
