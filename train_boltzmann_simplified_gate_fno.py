"""Standalone trainer for the separate linear-postprojection Gate-FNO."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Sequence

import torch

from boltzmann_simplified_gate_fno import SimplifiedGateBoltzmannQ
from onsager_collision.train import (
    FieldLoss,
    NpyFieldDataset,
    _cosine_warmup,
    _loader,
    _subset,
    atomic_json,
    seed_everything,
    validate_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("boltzmann_dataset_n64_L10_R9_r64_a23_10000_npy_float64_bkw"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/boltzmann_simplified_gate_fno"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fno-modes", type=int, default=8)
    parser.add_argument("--fno-width", type=int, default=36)
    parser.add_argument("--fno-layers", type=int, default=3)
    parser.add_argument("--rate-min", type=float, default=0.02)
    parser.add_argument("--rate-max", type=float, default=8.0)
    parser.add_argument("--initial-rate", type=float, default=0.5)
    parser.add_argument("--entropic-bgk-rate", type=float, default=0.5)
    parser.add_argument("--tail-exponent", type=float, default=0.1)
    parser.add_argument("--gain-deviation-limit", type=float, default=10.0)
    parser.add_argument("--rate-deviation-limit", type=float, default=10.0)
    parser.add_argument("--equilibrium-gate-scale", type=float, default=0.0)
    parser.add_argument("--enforce-entropy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--loss-floor", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def _sample_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    axes = tuple(range(1, target.ndim))
    numerator = (prediction - target).square().sum(dim=axes)
    denominator = target.square().sum(dim=axes).clamp_min(torch.finfo(target.dtype).tiny)
    return torch.sqrt(numerator / denominator)


def _train_epoch(
    model: SimplifiedGateBoltzmannQ,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    gradient_clip: float,
    epoch: int,
    log_every: int,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    relative_sum = 0.0
    count = 0
    started = time.perf_counter()
    for step, (f, target) in enumerate(loader, start=1):
        f = f.to(device=device, dtype=dtype, non_blocking=True)
        target = target.to(device=device, dtype=dtype, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(f)
        loss = loss_fn(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        batch = f.shape[0]
        loss_sum += float(loss.detach()) * batch
        relative_sum += float(_sample_relative_l2(prediction.detach(), target).sum())
        count += batch
        if log_every > 0 and (step % log_every == 0 or step == len(loader)):
            print(f"train epoch={epoch:04d} step={step:04d}/{len(loader):04d} "
                  f"loss={float(loss.detach()):.6e}", flush=True)
    return {
        "loss": loss_sum / count,
        "relative_l2_mean": relative_sum / count,
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def _evaluate(
    model: SimplifiedGateBoltzmannQ, loader, loss_fn: torch.nn.Module,
    *, device: torch.device, dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    relative_sum = 0.0
    global_error = 0.0
    global_target = 0.0
    moment_max = 0.0
    count = 0
    started = time.perf_counter()
    for f, target in loader:
        f = f.to(device=device, dtype=dtype, non_blocking=True)
        target = target.to(device=device, dtype=dtype, non_blocking=True)
        prediction = model(f)
        loss = loss_fn(prediction, target)
        batch = f.shape[0]
        loss_sum += float(loss) * batch
        relative_sum += float(_sample_relative_l2(prediction, target).sum())
        global_error += float((prediction - target).square().sum())
        global_target += float(target.square().sum())
        moment_max = max(moment_max, float(model.collision_moments(prediction).abs().max()))
        count += batch
    return {
        "loss": loss_sum / count,
        "relative_l2_mean": relative_sum / count,
        "relative_l2_global": math.sqrt(global_error / max(global_target, 1.0e-300)),
        "moment_max": moment_max,
        "seconds": time.perf_counter() - started,
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_training(args: argparse.Namespace) -> dict[str, float]:
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    manifest = validate_dataset(args.data_dir)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    split_data = {split: NpyFieldDataset(args.data_dir, split) for split in ("train", "val", "test")}
    train_set = _subset(split_data["train"], args.train_limit, args.seed + 101)
    val_set = _subset(split_data["val"], args.val_limit, args.seed + 202)
    test_set = _subset(split_data["test"], args.test_limit, args.seed + 303)
    generator = torch.Generator().manual_seed(args.seed + 404)
    train_loader = _loader(train_set, batch_size=args.batch_size, shuffle=True, workers=args.num_workers,
                           device=device, generator=generator)
    val_loader = _loader(val_set, batch_size=args.batch_size, shuffle=False, workers=args.num_workers, device=device)
    test_loader = _loader(test_set, batch_size=args.batch_size, shuffle=False, workers=args.num_workers, device=device)

    model = SimplifiedGateBoltzmannQ(
        n=int(manifest["n"]), L=float(manifest["L"]), fno_modes=args.fno_modes,
        fno_width=args.fno_width, fno_layers=args.fno_layers, rate_min=args.rate_min,
        rate_max=args.rate_max, initial_rate=args.initial_rate, entropic_bgk_rate=args.entropic_bgk_rate,
        tail_exponent=args.tail_exponent, gain_deviation_limit=args.gain_deviation_limit,
        rate_deviation_limit=args.rate_deviation_limit, equilibrium_gate_scale=args.equilibrium_gate_scale,
        enforce_entropy=args.enforce_entropy).to(device=device, dtype=dtype)
    loss_fn = FieldLoss(args.loss_floor).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: _cosine_warmup(epoch, epochs=args.epochs, warmup_epochs=args.warmup_epochs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_dir / "config.json",
        {
            "arguments": {
                **vars(args),
                "data_dir": str(args.data_dir),
                "output_dir": str(args.output_dir),
                "resume": str(args.resume) if args.resume else None,
            },
            "model_config": model.model_config,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        },
    )

    start_epoch = 0
    best_validation = math.inf
    history: list[dict[str, object]] = []
    history_path = args.output_dir / "history.json"
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []

    for epoch in range(start_epoch, args.epochs):
        train_metrics = _train_epoch(
            model, train_loader, optimizer, loss_fn, device=device, dtype=dtype,
            gradient_clip=args.gradient_clip, epoch=epoch + 1, log_every=args.log_every)
        val_metrics = _evaluate(model, val_loader, loss_fn, device=device, dtype=dtype)
        scheduler.step()
        validation = val_metrics["relative_l2_mean"]
        improved = validation < best_validation
        if improved:
            best_validation = validation
        payload = {
            "epoch": epoch,
            "best_validation": best_validation,
            "dtype": args.dtype,
            "model_config": model.model_config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        _save_checkpoint(args.output_dir / "latest.pt", payload)
        if improved:
            _save_checkpoint(args.output_dir / "best.pt", payload)
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

    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = _evaluate(model, test_loader, loss_fn, device=device, dtype=dtype)
    test_metrics["selected_epoch"] = float(int(best["epoch"]) + 1)
    test_metrics["best_val_relative_l2"] = float(best["best_validation"])
    atomic_json(args.output_dir / "test_metrics.json", test_metrics)
    print(json.dumps({"test": test_metrics}), flush=True)
    return test_metrics


def main(argv: Sequence[str] | None = None) -> dict[str, float]:
    return run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
