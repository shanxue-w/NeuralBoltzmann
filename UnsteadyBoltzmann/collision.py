"""Collision backends local to the unsteady Boltzmann cases."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


class GateFNOCollision:
    """Evaluate Gate-FNO in fp32 and map its Q back to the solver velocity grid."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        batch_chunk: int = 32,
        n: int | None = None,
        physical_velocity_half_width: float | None = None,
        collision_kernel_homogeneity: float = 0.0,
    ) -> None:
        from boltzmann_simplified_gate_fno import load_simplified_gate_model

        checkpoint_path = Path(checkpoint_path).resolve()
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = dict(checkpoint["model_config"])
        runtime_n = int(checkpoint_config["n"] if n is None else n)
        minimum_n = 2 * int(checkpoint_config["fno_modes"])
        if runtime_n < minimum_n or runtime_n % 2 != 0:
            raise ValueError(
                f"runtime Nv must be even and at least {minimum_n} for this checkpoint"
            )
        model, config = load_simplified_gate_model(checkpoint, n=runtime_n)
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        self.model.requires_grad_(False)
        self.n = int(config["n"])
        self.L = float(config["L"])
        self.physical_velocity_half_width = self.L if physical_velocity_half_width is None else float(physical_velocity_half_width)
        self.collision_kernel_homogeneity = float(collision_kernel_homogeneity)
        self.velocity_scale = self.L / self.physical_velocity_half_width
        self.collision_output_scale = self.velocity_scale ** (-(3.0 + self.collision_kernel_homogeneity))
        self.batch_chunk = int(batch_chunk)

    def evaluate(self, state: torch.Tensor) -> torch.Tensor:
        gate_input = state.clamp_min(0.0)
        pieces: list[torch.Tensor] = []
        for begin in range(0, state.shape[0], self.batch_chunk):
            stop = min(begin + self.batch_chunk, state.shape[0])
            tensor32 = gate_input[begin:stop].to(dtype=torch.float32)
            with torch.inference_mode():
                collision32 = self.model(tensor32)
            pieces.append(collision32.to(dtype=torch.float64).mul_(self.collision_output_scale))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


@dataclass(frozen=True)
class SpectralConfig:
    n: int
    L: float
    R: float = 9.0
    angular_accuracy: int = 13
    angular_chunk: int = 8
    batch_chunk: int = 1
    kernel_constant: float = 1.0 / (4.0 * math.pi)

class LazySpectralCollision:
    """Float64 CuPy spectral Q, constructed only when selected by a case."""

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

        device_index = state.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        self._initialize(device_index)
        torch_stream = torch.cuda.current_stream(state.device)
        with cp.cuda.Device(device_index):
            with cp.cuda.ExternalStream(torch_stream.cuda_stream):
                cupy_state = cp.from_dlpack(state.detach().contiguous())
                cupy_collision = self._solver.collision(cupy_state, batch_chunk=self.config.batch_chunk)
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


def collision_batch_chunk(args: Any) -> int | None:
    if args.collision_backend == "bgk":
        return None
    if args.collision_backend == "spectral" and args.spectral_batch_chunk is not None:
        return int(args.spectral_batch_chunk)
    return int(args.gate_batch_chunk)


def collision_signature(args: Any) -> dict[str, Any]:
    gate = args.collision_backend == "gate"
    spectral = args.collision_backend == "spectral"
    return {
        "collision_backend": args.collision_backend,
        "gate_model_checkpoint": str(args.checkpoint.resolve()) if gate else None,
        "spectral_R": args.spectral_R if spectral else None,
        "spectral_radial_accuracy": args.nv if spectral else None,
        "spectral_angular_accuracy": args.spectral_angular_accuracy if spectral else None,
        "spectral_angular_chunk": args.spectral_angular_chunk if spectral else None,
    }


def make_unsteady_collision(args: Any, batch_chunk: int | None) -> Any | None:
    if args.collision_backend == "bgk":
        return None
    batch_chunk = int(batch_chunk)
    if args.collision_backend == "gate":
        return GateFNOCollision(
            args.checkpoint, device=args.device, batch_chunk=batch_chunk, n=args.nv,
            physical_velocity_half_width=args.velocity_half_width)
    return LazySpectralCollision(
        SpectralConfig(
            n=args.nv, L=args.velocity_half_width, R=args.spectral_R,
            angular_accuracy=args.spectral_angular_accuracy,
            angular_chunk=args.spectral_angular_chunk, batch_chunk=batch_chunk))
