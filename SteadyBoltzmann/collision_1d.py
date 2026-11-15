"""GPU collision backends for the steady 1D solver."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class GateFNOMetadata:
    checkpoint: str
    checkpoint_epoch_one_based: int
    checkpoint_training_dtype: str
    inference_dtype: str
    batch_chunk: int
    checkpoint_velocity_n: int
    runtime_velocity_n: int
    resolution_override: bool
    checkpoint_model_config: dict[str, Any]
    model_config: dict[str, Any]
    negative_input_policy: str


def _clip_gate_input(state: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Clip only the tensor passed to Gate-FNO, without mutating PDE state."""

    clipped = state.clamp_min(0.0)
    removed = clipped - state
    statistics = torch.stack((state.min(), removed.sum(), state.abs().sum(),
                              (state < 0.0).sum(dtype=torch.float64))).tolist()
    reference = max(statistics[2], torch.finfo(torch.float64).tiny)
    diagnostics = {"input_minimum_before_clip": statistics[0], "input_negative_l1_clipped": statistics[1],
                   "input_negative_l1_fraction": statistics[1] / reference,
                   "input_negative_nodes": int(statistics[3])}
    return clipped, diagnostics


class GateFNOCollision:
    """Evaluate the simplified Gate-FNO in fp32 and return a float64 GPU field."""

    def __init__(self, checkpoint_path: str | Path, *, device: str | torch.device = "cuda",
                 batch_chunk: int = 32, n: int | None = None) -> None:
        from boltzmann_simplified_gate_fno import load_simplified_gate_model

        checkpoint_path = Path(checkpoint_path).resolve()
        self.device = torch.device(device)
        self.device = self.device if self.device.index is not None else torch.device("cuda", torch.cuda.current_device())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = dict(checkpoint["model_config"])
        model, config = load_simplified_gate_model(checkpoint, n=n)
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        # Parameters are immutable during inference.
        self.model.requires_grad_(False)
        self.n = int(config["n"])
        self.L = float(config["L"])
        self.batch_chunk = int(batch_chunk)
        self.last_diagnostics: dict[str, float | int | bool] = {}
        self.metadata = GateFNOMetadata(
            checkpoint=str(checkpoint_path),
            checkpoint_epoch_one_based=int(checkpoint["epoch"]) + 1,
            checkpoint_training_dtype=str(checkpoint.get("dtype")),
            inference_dtype="float32",
            batch_chunk=self.batch_chunk,
            checkpoint_velocity_n=int(checkpoint_config["n"]),
            runtime_velocity_n=self.n,
            resolution_override=self.n != int(checkpoint_config["n"]),
            checkpoint_model_config=checkpoint_config,
            model_config=config,
            negative_input_policy="clip_to_zero_at_gate_api",
        )

    def evaluate(self, state: torch.Tensor, *, collect_diagnostics: bool = False) -> torch.Tensor:
        """Evaluate a batch and optionally record moment diagnostics."""

        if collect_diagnostics:
            gate_input, input_diagnostics = _clip_gate_input(state)
        else:
            gate_input = state.clamp_min(0.0)

        pieces: list[torch.Tensor] = []
        moment_errors: list[float] = []
        for begin in range(0, state.shape[0], self.batch_chunk):
            stop = min(begin + self.batch_chunk, state.shape[0])
            # The only precision conversion in the PDE solve is at this API.
            tensor32 = gate_input[begin:stop].to(dtype=torch.float32)
            with torch.inference_mode():
                collision32 = self.model(tensor32)
            if collect_diagnostics:
                moment_errors.append(float(self.model.collision_moments(collision32).abs().max()))
            pieces.append(collision32.to(dtype=torch.float64))
        result = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        if collect_diagnostics:
            self.last_diagnostics = {**input_diagnostics, "diagnostics_requested": True,
                                     "full_diagnostics_collected": False,
                                     "effective_batch_chunk": min(self.batch_chunk, state.shape[0]),
                                     "collision_moment_abs_max": max(moment_errors)}
        return result

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


@dataclass(frozen=True)
class BGKCollisionMetadata:
    backend: str
    formula: str
    rate: float
    maxwellian: str
    n: int
    L: float
    dtype: str


class BGKCollision:
    """Float64 five-moment BGK operator ``rho/2 * (M[f] - f)``."""

    def __init__(self, *, n: int, L: float, device: str | torch.device = "cuda") -> None:
        from boltzmann_discrete_maxwellian import DiscreteMaxwellian

        self.n = int(n)
        self.L = float(L)
        self.device = torch.device(device)
        self.dv3 = (2.0 * self.L / self.n) ** 3
        self.maxwellian = DiscreteMaxwellian(n=self.n, L=self.L).to(
            device=self.device, dtype=torch.float64
        )
        self.last_diagnostics: dict[str, float | int | bool] = {}
        self.metadata = BGKCollisionMetadata(
            backend="bgk",
            formula="0.5 * rho * (M[f] - f)",
            rate=0.5,
            maxwellian="discrete_maximum_entropy_five_moment",
            n=self.n,
            L=self.L,
            dtype="float64",
        )

    def evaluate(self, state: torch.Tensor, *, collect_diagnostics: bool = False) -> torch.Tensor:
        """Evaluate BGK without passing through the neural or spectral backends."""

        if state.dtype != torch.float64:
            raise TypeError(f"BGK state must be float64, received {state.dtype}")
        equilibrium, diagnostics = self.maxwellian(state, return_diagnostics=True)
        converged = diagnostics["converged"]
        if not bool(converged.all()):
            failed = int((~converged).sum())
            error = float(diagnostics["max_relative_moment_error"])
            raise RuntimeError(
                f"discrete BGK Maxwellian failed in {failed} cells; "
                f"max relative moment error={error:.6e}"
            )
        density = self.dv3 * state.sum(dim=(-3, -2, -1))
        result = 0.5 * density[:, None, None, None] * (equilibrium - state)
        if collect_diagnostics:
            self.last_diagnostics = {
                "diagnostics_requested": True,
                "maxwellian_max_relative_moment_error": float(
                    diagnostics["max_relative_moment_error"]
                ),
                "minimum_density": float(density.min()),
                "maximum_density": float(density.max()),
            }
        return result

    def close(self) -> None:
        del self.maxwellian
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


@dataclass(frozen=True)
class LinearCollisionConfig:
    n: int
    L: float
    R: float = 9.0
    kernel_constant: float = 1.0 / (4.0 * math.pi)

class LazyLocalMaxwellianLinearCollision:
    """GPU ``L_M s=Q[M,s]+Q[s,M]`` for all spatial cells in one batch."""

    def __init__(self, config: LinearCollisionConfig) -> None:
        self.config = config
        self._solver: Any | None = None

    def _initialize(self, device_index: int) -> None:
        if self._solver is not None:
            return
        import cupy as cp

        from linear_collision_gpu import LinearCollisionGPU

        with cp.cuda.Device(device_index):
            self._solver = LinearCollisionGPU(n=self.config.n, L=self.config.L, R=self.config.R,
                                              radial_accuracy=self.config.n,
                                              collision_kernel=lambda _: self.config.kernel_constant,
                                              backend="cupy")

    def evaluate(self, perturbation: torch.Tensor, *, density: torch.Tensor,
                 velocity: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        import cupy as cp

        device_index = perturbation.device.index if perturbation.device.index is not None else torch.cuda.current_device()
        self._initialize(device_index)
        with cp.cuda.Device(device_index):
            centered = cp.from_dlpack(perturbation.detach().contiguous())
            unshifted = cp.fft.ifftshift(centered, axes=(-3, -2, -1))
            result_unshifted = self._solver.linear_collision_batch(
                unshifted,
                cp.from_dlpack(density.detach().contiguous()),
                cp.from_dlpack(velocity.detach().contiguous()),
                cp.from_dlpack(temperature.detach().contiguous()),
                conserve_mass=True,
            )
            result_centered = cp.fft.fftshift(result_unshifted, axes=(-3, -2, -1))
            result = torch.utils.dlpack.from_dlpack(result_centered).clone()
        return result

    def close(self) -> None:
        if self._solver is None:
            return
        import cupy as cp

        del self._solver
        self._solver = None
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        gc.collect()


@dataclass(frozen=True)
class SpectralConfig:
    n: int
    L: float
    R: float = 9.0
    angular_accuracy: int = 23
    angular_chunk: int = 8
    batch_chunk: int = 1
    kernel_constant: float = 1.0 / (4.0 * math.pi)


class LazySpectralCollision:
    """Float64 CuPy spectral Q, constructed only at the hybrid switch."""

    def __init__(self, config: SpectralConfig) -> None:
        self.config = config
        self._solver: Any | None = None

    def _initialize(self, device_index: int) -> None:
        if self._solver is not None:
            return
        import cupy as cp

        from boltzmann_fft_gpu import BoltzmannFFT

        with cp.cuda.Device(device_index):
            self._solver = BoltzmannFFT(
                n=self.config.n,
                L=self.config.L,
                R=self.config.R,
                radial_accuracy=self.config.n,
                angular_accuracy=self.config.angular_accuracy,
                omega=1.0,
                kernel_constant=self.config.kernel_constant,
                backend="cupy",
                dtype=np.float64,
                lebedev_dir=REPO_ROOT / "Lebedev",
                loss_mode="sinc",
                enforce_mass=True,
                angular_chunk=self.config.angular_chunk,
                use_rfft=True,
            )

    def evaluate(self, state: torch.Tensor) -> torch.Tensor:
        import cupy as cp

        device_index = state.device.index if state.device.index is not None else torch.cuda.current_device()
        self._initialize(device_index)
        torch_stream = torch.cuda.current_stream(state.device)
        with cp.cuda.Device(device_index):
            # Run CuPy on Torch's current stream. Both DLPack conversions then
            # remain device resident without cross-stream host coordination.
            with cp.cuda.ExternalStream(torch_stream.cuda_stream):
                cupy_state = cp.from_dlpack(state.detach().contiguous())
                cupy_collision = self._solver.collision(cupy_state, batch_chunk=self.config.batch_chunk)
                # The tensor shares the producer allocation and keeps it alive;
                # an additional device clone would only add a full-field copy.
                result = torch.utils.dlpack.from_dlpack(cupy_collision)
        return result

    def close(self) -> None:
        if self._solver is None:
            return
        import cupy as cp

        del self._solver
        self._solver = None
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        gc.collect()
