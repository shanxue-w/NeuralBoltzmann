"""Test relative L2 mean/std: Gate-FNO and BGK = rho/2 * (M - f)."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from boltzmann_simplified_gate_fno import load_simplified_gate_model
from onsager_collision.train import NpyFieldDataset
from train_boltzmann_simplified_gate_fno import _sample_relative_l2


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs/boltzmann_tail_simplified_gate_fno_w36_l3_neuralop_10000_b8_s1234_bkw"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    settings = json.loads((args.run_dir / "config.json").read_text())["arguments"]
    data_dir = args.data_dir or Path(settings["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    model, _ = load_simplified_gate_model(checkpoint)
    dtype = getattr(torch, settings["dtype"])
    model = model.to(device=args.device, dtype=dtype).eval()
    torch.set_float32_matmul_precision("high")
    dataset = NpyFieldDataset(data_dir, "test")
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False)
    errors = {"Gate-FNO": [], "BGK": []}
    started = time.perf_counter()
    completed = 0

    for f, q in loader:
        f, q = f.to(device=args.device, dtype=dtype), q.to(device=args.device, dtype=dtype)
        prediction = model(f)
        # Use the same moment-matched discrete Maxwellian as Gate-FNO.
        M, status = model.maxwellian_solver(f, return_diagnostics=True)
        if not status["converged"].all():
            raise RuntimeError("Discrete Maxwellian did not converge")
        rho = model.dv3 * f.sum(dim=(-3, -2, -1), keepdim=True)
        bgk = 0.5 * rho * (M - f)
        for name, value in (("Gate-FNO", prediction), ("BGK", bgk)):
            error = _sample_relative_l2(value, q)
            if not torch.isfinite(error).all():
                raise RuntimeError(f"Non-finite {name} relative L2 error")
            errors[name].append(error.double().cpu())
        completed += len(f)
        elapsed = time.perf_counter() - started
        eta = elapsed * (len(dataset) - completed) / completed
        print(f"{completed}/{len(dataset)} samples | elapsed={elapsed:.1f}s | ETA={eta:.1f}s",
              flush=True)

    print(f"Test samples: {len(dataset)}; dtype: {dtype}; device: {args.device}")
    print("Relative L2 error: ||Q_pred - Q||_2 / ||Q||_2; std uses ddof=1")
    result = {"test_samples": len(dataset), "ddof": 1, "data_dir": str(data_dir),
              "dtype": settings["dtype"], "device": args.device,
              "bgk": "rho/2 * (M - f), moment-matched discrete Maxwellian",
              "relative_l2": {}}
    for name, values in errors.items():
        values = torch.cat(values)
        mean, std = values.mean().item(), values.std(unbiased=True).item()
        result["relative_l2"][name] = {"mean": mean, "std": std}
        print(f"{name:8s} mean={mean:.10e}  std={std:.10e}")
    output = args.run_dir / "test_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
