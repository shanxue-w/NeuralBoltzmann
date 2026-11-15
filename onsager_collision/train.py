"""Standalone full-field trainer for the strict Onsager collision model."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

if __package__ in (None, ""):
    # ``python onsager_collision/train.py`` puts only this directory on
    # sys.path. Add the project root so the package can be imported normally.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from onsager_collision.model import OnsagerCollision
    from onsager_collision.physics import MAXWELL_KERNEL_CONSTANT
else:
    from .model import OnsagerCollision
    from .physics import MAXWELL_KERNEL_CONSTANT


MINIMUM_ANGULAR_ACCURACY = 23
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "boltzmann_dataset_n64_L10_R9_r64_a23_10000_npy_float64_bkw"
)


def validate_dataset(data_dir: Path) -> dict[str, object]:
    """Validate the constant-kernel FP64 spectral-label manifest."""

    manifest_path = Path(data_dir) / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernel = float(manifest.get("kernel_constant", math.nan))
    if not math.isclose(
        kernel, MAXWELL_KERNEL_CONSTANT, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("the current model requires B=1/(4*pi)")
    if int(manifest.get("angular_accuracy", -1)) < MINIMUM_ANGULAR_ACCURACY:
        raise ValueError("spectral labels require angular_accuracy >= 23")
    if manifest.get("dtype") != "float64":
        raise ValueError("stored f and Q fields must use float64")
    if manifest.get("source_dtype") != "float64":
        raise ValueError("spectral label generation must use float64")
    return manifest


class NpyFieldDataset(Dataset):
    """Lazy read-only access to one ``f.npy``/``q.npy`` split."""

    def __init__(self, root: Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.f_path = self.root / split / "f.npy"
        self.q_path = self.root / split / "q.npy"
        if not self.f_path.is_file() or not self.q_path.is_file():
            raise FileNotFoundError(
                f"split {split!r} requires both {self.f_path} and {self.q_path}"
            )
        f = np.load(self.f_path, mmap_mode="r")
        q = np.load(self.q_path, mmap_mode="r")
        if f.dtype != np.float64 or q.dtype != np.float64:
            raise ValueError(f"split {split!r} must use float64 arrays")
        if f.shape != q.shape or f.ndim != 4:
            raise ValueError(f"split {split!r} has incompatible f/Q shapes")
        self.length = int(f.shape[0])
        self.shape = tuple(int(value) for value in f.shape[1:])
        self._f = None
        self._q = None

    def _open(self) -> None:
        if self._f is None:
            self._f = np.load(self.f_path, mmap_mode="r")
            self._q = np.load(self.q_path, mmap_mode="r")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()
        f = np.array(self._f[index], copy=True)
        q = np.array(self._q[index], copy=True)
        return torch.from_numpy(f), torch.from_numpy(q)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_f"] = None
        state["_q"] = None
        return state


class FieldLoss(nn.Module):
    """Mean per-sample normalized field MSE with a fixed denominator floor."""

    def __init__(self, floor: float = 1e-4) -> None:
        super().__init__()
        self.floor2 = float(floor) ** 2

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        axes = tuple(range(1, target.ndim))
        error = (prediction - target).square().mean(dim=axes)
        target_energy = target.square().mean(dim=axes)
        return (error / (target_energy + self.floor2)).mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _subset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None or limit == len(dataset):
        return dataset
    if limit <= 0 or limit > len(dataset):
        raise ValueError("split limit must be between one and its full size")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit]
    return Subset(dataset, indices.tolist())


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def _sample_relative_l2(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    axes = tuple(range(1, target.ndim))
    numerator = (prediction - target).square().sum(dim=axes)
    denominator = target.square().sum(dim=axes).clamp_min(
        torch.finfo(target.dtype).tiny
    )
    return torch.sqrt(numerator / denominator)


def _extend(values: list[float], tensor: torch.Tensor) -> None:
    values.extend(
        float(value)
        for value in tensor.detach().to(device="cpu", dtype=torch.float64)
    )


def _summary(values: Iterable[float], prefix: str) -> dict[str, float]:
    tensor = torch.as_tensor(tuple(values), dtype=torch.float64)
    return {
        f"{prefix}_mean": float(tensor.mean()),
        f"{prefix}_median": float(torch.quantile(tensor, 0.5)),
        f"{prefix}_p95": float(torch.quantile(tensor, 0.95)),
        f"{prefix}_max": float(tensor.max()),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(
    model: OnsagerCollision,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    gradient_clip: float,
    epoch: int,
    log_every: int,
) -> dict[str, float]:
    model.train()
    losses = 0.0
    count = 0
    relative_values: list[float] = []
    started = time.perf_counter()
    for step, (f, target) in enumerate(loader, start=1):
        f = f.to(device=device, dtype=dtype, non_blocking=True)
        target = target.to(device=device, dtype=dtype, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(f)
        loss = loss_fn(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss in epoch {epoch}, step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(
                f"non-finite gradient in epoch {epoch}, step {step}"
            )
        optimizer.step()
        batch = f.shape[0]
        losses += float(loss.detach()) * batch
        count += batch
        _extend(relative_values, _sample_relative_l2(prediction, target))
        if log_every > 0 and (step % log_every == 0 or step == len(loader)):
            print(
                f"train epoch={epoch:04d} step={step:04d}/{len(loader):04d} "
                f"loss={float(loss.detach()):.6e}",
                flush=True,
            )
    result = {
        "loss": losses / count,
        "seconds": time.perf_counter() - started,
    }
    result.update(_summary(relative_values, "relative_l2"))
    return result


@torch.no_grad()
def evaluate(
    model: OnsagerCollision,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    losses = 0.0
    count = 0
    relative_values: list[float] = []
    bgk_values: list[float] = []
    moment_max = 0.0
    entropy_max = -math.inf
    inference_seconds = 0.0
    started = time.perf_counter()
    for f, target in loader:
        f = f.to(device=device, dtype=dtype, non_blocking=True)
        target = target.to(device=device, dtype=dtype, non_blocking=True)
        _synchronize(device)
        inference_started = time.perf_counter()
        prediction, diagnostics = model(f, return_diagnostics=True)
        _synchronize(device)
        inference_seconds += time.perf_counter() - inference_started
        loss = loss_fn(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite evaluation loss")
        batch = f.shape[0]
        losses += float(loss) * batch
        count += batch
        _extend(relative_values, _sample_relative_l2(prediction, target))
        _extend(
            bgk_values,
            _sample_relative_l2(diagnostics["bgk_collision"], target),
        )
        moment_max = max(
            moment_max, float(model.collision_moments(prediction).abs().max())
        )
        entropy_max = max(
            entropy_max, float(model.entropy_pairing(f, prediction).max())
        )
    result = {
        "loss": losses / count,
        "samples": float(count),
        "seconds": time.perf_counter() - started,
        "inference_ms_per_sample": 1000.0 * inference_seconds / count,
        "moment_max": moment_max,
        "entropy_max": entropy_max,
    }
    result.update(_summary(relative_values, "relative_l2"))
    result.update(_summary(bgk_values, "bgk_relative_l2"))
    return result


def _cosine_warmup(epoch: int, *, epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    remaining = max(epochs - warmup_epochs, 1)
    progress = min(max(epoch - warmup_epochs, 0) / remaining, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _save_checkpoint(
    path: Path,
    *,
    model: OnsagerCollision,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_generator: torch.Generator,
    epoch: int,
    best_validation: float,
    dtype_name: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "onsager-collision-v1",
            "epoch": int(epoch),
            "best_validation": float(best_validation),
            "dtype": dtype_name,
            "model_config": model.model_config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "train_generator_state": train_generator.get_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _restore_checkpoint(
    path: Path,
    *,
    model: OnsagerCollision,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_generator: torch.Generator,
    device: torch.device,
    dtype_name: str,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != "onsager-collision-v1":
        raise ValueError("unsupported checkpoint format")
    if checkpoint.get("model_config") != model.model_config:
        raise ValueError("checkpoint model config does not match")
    if checkpoint.get("dtype") != dtype_name:
        raise ValueError("checkpoint dtype does not match")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    random.setstate(checkpoint["random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.random.set_rng_state(checkpoint["torch_random_state"].cpu())
    if device.type == "cuda" and checkpoint.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_random_state"]]
        )
    train_generator.set_state(checkpoint["train_generator_state"].cpu())
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_validation"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/boltzmann_onsager_fno")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fno-modes", type=int, default=8)
    parser.add_argument("--fno-width", type=int, default=24)
    parser.add_argument("--fno-layers", type=int, default=3)
    parser.add_argument("--analysis-channels", type=int, default=4)
    parser.add_argument("--analysis-modes", type=int, default=None)
    parser.add_argument("--initial-local-rate", type=float, default=0.5)
    parser.add_argument("--initial-gate-rate", type=float, default=0.25)
    parser.add_argument("--initial-filter-scale", type=float, default=0.02)
    parser.add_argument("--bgk-rate", type=float, default=0.5)
    parser.add_argument("--log-floor", type=float, default=1e-20)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def run_training(args: argparse.Namespace) -> dict[str, float]:
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    manifest = validate_dataset(args.data_dir)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    train_data = NpyFieldDataset(args.data_dir, "train")
    val_data = NpyFieldDataset(args.data_dir, "val")
    test_data = NpyFieldDataset(args.data_dir, "test")
    expected_shape = tuple(int(value) for value in manifest["shape"])
    if any(data.shape != expected_shape for data in (train_data, val_data, test_data)):
        raise ValueError("dataset arrays disagree with the manifest shape")
    train_set = _subset(train_data, args.train_limit, args.seed + 101)
    val_set = _subset(val_data, args.val_limit, args.seed + 202)
    test_set = _subset(test_data, args.test_limit, args.seed + 303)
    train_generator = torch.Generator().manual_seed(args.seed + 404)
    train_loader = _loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        device=device,
        generator=train_generator,
    )
    val_loader = _loader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.num_workers,
        device=device,
    )
    test_loader = _loader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.num_workers,
        device=device,
    )

    model = OnsagerCollision(
        n=int(manifest["n"]),
        L=float(manifest["L"]),
        fno_modes=args.fno_modes,
        fno_width=args.fno_width,
        fno_layers=args.fno_layers,
        analysis_channels=args.analysis_channels,
        analysis_modes=args.analysis_modes,
        initial_local_rate=args.initial_local_rate,
        initial_gate_rate=args.initial_gate_rate,
        initial_filter_scale=args.initial_filter_scale,
        bgk_rate=args.bgk_rate,
        log_floor=args.log_floor,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _cosine_warmup(
            epoch, epochs=args.epochs, warmup_epochs=args.warmup_epochs
        ),
    )
    loss_fn = FieldLoss().to(device=device, dtype=dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "arguments": {**vars(args), "data_dir": str(args.data_dir),
                      "output_dir": str(args.output_dir),
                      "resume": str(args.resume) if args.resume else None},
        "model_config": model.model_config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dataset": {"format": manifest.get("format"), "shape": expected_shape,
                    "train": len(train_set), "val": len(val_set), "test": len(test_set)},
    }
    atomic_json(args.output_dir / "config.json", config)

    history_path = args.output_dir / "history.json"
    history: list[dict[str, object]] = []
    start_epoch = 0
    best_validation = math.inf
    if args.resume is not None:
        start_epoch, best_validation = _restore_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_generator=train_generator,
            device=device,
            dtype_name=args.dtype,
        )
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
            dtype=dtype,
            gradient_clip=args.gradient_clip,
            epoch=epoch + 1,
            log_every=args.log_every,
        )
        val_metrics = evaluate(
            model, val_loader, loss_fn, device=device, dtype=dtype
        )
        scheduler.step()
        validation = val_metrics["relative_l2_mean"]
        if validation < best_validation:
            best_validation = validation
            _save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_generator=train_generator,
                epoch=epoch,
                best_validation=best_validation,
                dtype_name=args.dtype,
            )
        _save_checkpoint(
            args.output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_generator=train_generator,
            epoch=epoch,
            best_validation=best_validation,
            dtype_name=args.dtype,
        )
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "best_val_relative_l2": best_validation,
        }
        history.append(record)
        atomic_json(history_path, history)
        print(json.dumps(record), flush=True)

    best = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best["model"])
    test_metrics = evaluate(
        model, test_loader, loss_fn, device=device, dtype=dtype
    )
    test_metrics["selected_epoch"] = float(int(best["epoch"]) + 1)
    test_metrics["best_val_relative_l2"] = float(best["best_validation"])
    atomic_json(args.output_dir / "test_metrics.json", test_metrics)
    print(json.dumps({"test": test_metrics}), flush=True)
    return test_metrics


def main(argv: Sequence[str] | None = None) -> dict[str, float]:
    args = build_parser().parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    main()
