"""Angular-quadrature sweep for homogeneous stream relaxation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from run_homogeneous import DEFAULT_CHECKPOINT, REFERENCE_ACCURACY, NeuralCollision, SpectralCollision, checkpoint_velocity_half_width, error_metrics, initial_condition, load_or_compute_reference, make_grid

DEFAULT_ACCURACIES = (3, 5, 7, 9, 11)


def time_gpu_resident_q(call: Any, synchronize: Any, warmup: int, repeats: int) -> dict[str, float | int]:
    for _ in range(warmup):
        call()
    synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = call()
        synchronize()
        samples.append(time.perf_counter() - start)
        del output
    values = np.asarray(samples)
    return {"repeats": repeats, "mean_seconds_per_sample": float(values.mean()), "std_seconds_per_sample": float(values.std()), "minimum_seconds_per_sample": float(values.min())}


def time_spectral_solver(rhs: SpectralCollision, value: np.ndarray, warmup: int, repeats: int) -> dict[str, float | int]:
    value_gpu = rhs.cp.asarray(value, dtype=rhs.solver.dtype)
    return time_gpu_resident_q(lambda: rhs.solver.collision(value_gpu, batch_chunk=1), rhs.cp.cuda.get_current_stream().synchronize, warmup, repeats)


def time_neural_solver(rhs: NeuralCollision, value: np.ndarray, warmup: int, repeats: int) -> dict[str, float | int]:
    tensor = rhs.torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))[None].to(rhs.device)
    with rhs.torch.inference_mode():
        return time_gpu_resident_q(lambda: rhs.model(tensor), rhs.torch.cuda.synchronize, warmup, repeats)


def integrate_uncorrected_ssprk2(name: str, rhs: Any, initial: np.ndarray, dtau: float, steps: int) -> tuple[np.ndarray, dict[str, Any]]:
    states = np.empty((steps + 1, *initial.shape), dtype=np.float64)
    states[0] = initial
    minimum = float(initial.min())
    stride = max(1, steps // 10)
    for step in range(steps):
        current = states[step]
        with np.errstate(over="ignore", invalid="ignore"):
            first = current + dtau * rhs(current)
            following = 0.5 * current + 0.5 * (first + dtau * rhs(first))
        if not np.isfinite(following).all():
            states[step + 1:] = np.nan
            return states, {"positivity_correction": False, "status": "failed_nonfinite", "failed_step": step + 1, "last_finite_tau": step * dtau, "minimum_stage_value": minimum, "minimum_saved_state": float(np.nanmin(states))}
        minimum = min(minimum, float(first.min()), float(following.min()))
        states[step + 1] = following
        if (step + 1) % stride == 0:
            print(f"  {name}: {step + 1:02d}/{steps:02d}", flush=True)
    return states, {"positivity_correction": False, "status": "completed", "failed_step": None, "last_finite_tau": steps * dtau, "minimum_stage_value": minimum, "minimum_saved_state": float(states.min())}


def run_spectral(args: argparse.Namespace, case_name: str, accuracy: int, initial: np.ndarray, grid: Any) -> tuple[np.ndarray, dict[str, Any], dict[str, float | int]]:
    print(f"[{case_name}] spectral a={accuracy}", flush=True)
    rhs = SpectralCollision(grid, args.R, accuracy, args.angular_chunk)
    timing = time_spectral_solver(rhs, initial, args.warmup, args.timing_repeats)
    states, diagnostics = integrate_uncorrected_ssprk2(f"spectral a={accuracy}", rhs, initial, args.dtau, round(args.tau_end / args.dtau))
    rhs.close()
    return states, diagnostics, timing


def time_spectral_q(args: argparse.Namespace, accuracy: int, initial: np.ndarray, grid: Any) -> dict[str, float | int]:
    rhs = SpectralCollision(grid, args.R, accuracy, args.angular_chunk)
    timing = time_spectral_solver(rhs, initial, args.warmup, args.timing_repeats)
    rhs.close()
    return timing


def run_neural(args: argparse.Namespace, case_name: str, initial: np.ndarray, grid: Any) -> tuple[np.ndarray, dict[str, Any], dict[str, float | int]]:
    print(f"[{case_name}] neural operator", flush=True)
    rhs = NeuralCollision(args.checkpoint, grid, args.device)
    timing = time_neural_solver(rhs, initial, args.warmup, args.timing_repeats)
    states, diagnostics = integrate_uncorrected_ssprk2("neural operator", rhs, initial, args.dtau, round(args.tau_end / args.dtau))
    rhs.close()
    return states, diagnostics, timing


def plot_case(output_dir: Path, tau: np.ndarray, rows: list[dict[str, Any]], accuracies: list[int]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/q_learning_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive = tau > 0.0
    for scale in ("absolute", "relative"):
        figure, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), constrained_layout=True)
        for axis, accuracy in zip(axes.flat, accuracies):
            for norm, color, marker in (("l1", "#007C91", "o"), ("l2", "#E68613", "s")):
                spectral = np.asarray([row[f"spectral_a{accuracy}_{scale}_{norm}"] for row in rows])
                neural = np.asarray([row[f"neural_operator_{scale}_{norm}"] for row in rows])
                axis.semilogy(tau[positive], spectral[positive], color=color, linewidth=1.8, marker=marker, markevery=5, markersize=3.6, label=rf"Spectral $L^{norm[-1]}_v$")
                axis.semilogy(tau[positive], neural[positive], color=color, linewidth=1.5, linestyle="--", label=rf"Neural $L^{norm[-1]}_v$")
            axis.set(title=rf"$a={accuracy}$", xlabel=r"dimensionless time $\tau=\rho t$", ylabel=f"{scale} error")
            axis.grid(alpha=0.22)
            axis.legend(frameon=False, fontsize=8, ncol=2)
        for axis in axes.flat[len(accuracies):]:
            axis.set_visible(False)
        figure.savefig(output_dir / f"angular_accuracy_{scale}_error_facets.png", dpi=240)
        plt.close(figure)


def write_summary_tables(output_dir: Path, final_rows: list[dict[str, Any]], accuracies: list[int], tau_end: float) -> None:
    long_rows = []
    for row in final_rows:
        methods = [("Neural operator", "", "neural_operator_")] + [(f"Spectral a={accuracy}", accuracy, f"spectral_a{accuracy}_") for accuracy in accuracies]
        for method, accuracy, prefix in methods:
            long_rows.append({"case": row["case"], "tau": tau_end, "method": method, "angular_accuracy": accuracy, "status": row[prefix + "status"], "last_finite_tau": row[prefix + "last_finite_tau"], "mean_q_ms": 1.0e3 * row[prefix + "mean_q_seconds"], **{key: row[prefix + key] for key in ("absolute_l1", "absolute_l2", "relative_l1", "relative_l2")}})
    with (output_dir / "final_error_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)

    lines = [f"# Angular-accuracy sweep at $\\tau={tau_end:g}$", "", f"All errors are measured against the independent spectral trajectory with angular accuracy $a={REFERENCE_ACCURACY}$."]
    for case_name, title in (("two_stream", "Two stream"), ("three_stream", "Three stream")):
        lines += ["", f"## {title}", "", "| method | status | last finite $\\tau$ | mean $Q$ time (ms) | absolute $L^1$ | absolute $L^2$ | relative $L^1$ | relative $L^2$ |", "|:---|:---|---:|---:|---:|---:|---:|---:|"]
        for row in (item for item in long_rows if item["case"] == case_name):
            values = ["--" if not np.isfinite(row[key]) else f"{row[key]:.6e}" for key in ("absolute_l1", "absolute_l2", "relative_l1", "relative_l2")]
            lines.append(f"| {row['method']} | {row['status']} | {row['last_finite_tau']:.1f} | {row['mean_q_ms']:.3f} | " + " | ".join(values) + " |")
    (output_dir / "final_error_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(args: argparse.Namespace, case_name: str, grid: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = args.output_dir / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    initial, initial_metadata = initial_condition(case_name, grid)
    steps = round(args.tau_end / args.dtau)
    tau = np.arange(steps + 1, dtype=np.float64) * args.dtau
    reference, _, reference_diagnostics, cache_used = load_or_compute_reference(args, case_name, initial, grid, steps, [])
    rows = [{"tau": value} for value in tau]
    diagnostics = {f"spectral_a{REFERENCE_ACCURACY}": reference_diagnostics}
    timings = {f"spectral_a{REFERENCE_ACCURACY}": time_spectral_q(args, REFERENCE_ACCURACY, initial, grid)}
    for accuracy in args.angular_accuracy:
        states, method_diagnostics, timing = run_spectral(args, case_name, accuracy, initial, grid)
        method = f"spectral_a{accuracy}"
        diagnostics[method] = method_diagnostics
        timings[method] = timing
        for index, row in enumerate(rows):
            row.update({f"{method}_{key}": value for key, value in error_metrics(states[index], reference[index], grid).items()})
        del states

    states, method_diagnostics, timing = run_neural(args, case_name, initial, grid)
    diagnostics["neural_operator"] = method_diagnostics
    timings["neural_operator"] = timing
    for index, row in enumerate(rows):
        row.update({f"neural_operator_{key}": value for key, value in error_metrics(states[index], reference[index], grid).items()})
    del states

    with (output_dir / "angular_accuracy_time_series.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_case(output_dir, tau, rows, args.angular_accuracy)

    metrics = ("absolute_l1", "absolute_l2", "relative_l1", "relative_l2")
    final = rows[-1]
    summary = {
        "case": case_name, "initial_condition": initial_metadata,
        "grid": {"n": grid.n, "L": grid.L, "R": args.R, "radial_accuracy": grid.n, "reference_angular_accuracy": REFERENCE_ACCURACY, "compared_angular_accuracies": args.angular_accuracy, "kernel_constant": "1/(4*pi)", "loss_mode": "sinc"},
        "time_integration": {"integrator": "uncorrected SSPRK2 for every trajectory", "dtau": args.dtau, "tau_end": args.tau_end},
        "checkpoint": str(args.checkpoint), "reference_cache": {"path": str(args.reference_cache_dir / case_name), "loaded": cache_used},
        "trajectory_diagnostics": diagnostics,
        "q_evaluation_timing": {"scope": "batch-1 GPU-resident Q evaluation", "includes_host_device_transfer": False, "includes_gpu_synchronization": True, "excludes_initialization_and_warmup": True, "methods": timings},
        "final_errors": {f"spectral_a{accuracy}": {metric: final[f"spectral_a{accuracy}_{metric}"] if np.isfinite(final[f"spectral_a{accuracy}_{metric}"]) else None for metric in metrics} for accuracy in args.angular_accuracy},
        "files": {"errors": "angular_accuracy_time_series.csv", "absolute_error_figure": "angular_accuracy_absolute_error_facets.png", "relative_error_figure": "angular_accuracy_relative_error_facets.png"},
    }
    summary["final_errors"]["neural_operator"] = {metric: final[f"neural_operator_{metric}"] for metric in metrics}
    (output_dir / "angular_accuracy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    final_row = {"case": case_name, **final}
    for method, values in diagnostics.items():
        if method != f"spectral_a{REFERENCE_ACCURACY}":
            final_row[f"{method}_status"] = values["status"]
            final_row[f"{method}_last_finite_tau"] = values["last_finite_tau"]
    for method, values in timings.items():
        final_row[f"{method}_mean_q_seconds"] = values["mean_seconds_per_sample"]
    return summary, final_row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Compare low angular accuracies with a={REFERENCE_ACCURACY} for stream relaxation.")
    parser.add_argument("--case", nargs="+", choices=("two_stream", "three_stream"), default=["two_stream", "three_stream"])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "angular_accuracy_sweep")
    parser.add_argument("--angular-accuracy", type=int, nargs="+", default=list(DEFAULT_ACCURACIES))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--R", type=float, default=9.0)
    parser.add_argument("--angular-chunk", type=int, default=16)
    parser.add_argument("--reference-cache-dir", type=Path, default=Path(__file__).parent / "reference_cache")
    parser.add_argument("--refresh-reference-cache", action="store_true")
    parser.add_argument("--dtau", type=float, default=0.1)
    parser.add_argument("--tau-end", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timing-repeats", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.reference_cache_dir = args.reference_cache_dir.resolve()
    args.angular_accuracy = sorted(set(args.angular_accuracy))
    grid = make_grid(args.n, checkpoint_velocity_half_width(args.checkpoint))
    results = [run_case(args, case_name, grid) for case_name in args.case]
    summaries = {summary["case"]: summary for summary, _ in results}
    final_rows = [row for _, row in results]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing_rows = []
    method_order = ["neural_operator", *[f"spectral_a{accuracy}" for accuracy in args.angular_accuracy], f"spectral_a{REFERENCE_ACCURACY}"]
    for case_name, summary in summaries.items():
        case_rows = []
        for method in method_order:
            timing = summary["q_evaluation_timing"]["methods"][method]
            row = {"case": case_name, "method": method, "angular_accuracy": int(method.removeprefix("spectral_a")) if method.startswith("spectral_a") else "", "repeats": timing["repeats"], "mean_seconds_per_sample": timing["mean_seconds_per_sample"], "mean_milliseconds_per_sample": 1.0e3 * timing["mean_seconds_per_sample"], "std_seconds_per_sample": timing["std_seconds_per_sample"], "minimum_seconds_per_sample": timing["minimum_seconds_per_sample"]}
            case_rows.append(row)
            timing_rows.append(row)
        with (args.output_dir / case_name / "q_evaluation_timing.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
            writer.writeheader()
            writer.writerows(case_rows)
        summary["files"]["q_evaluation_timing"] = "q_evaluation_timing.csv"
        (args.output_dir / case_name / "angular_accuracy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "q_evaluation_timing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    write_summary_tables(args.output_dir, final_rows, args.angular_accuracy, args.tau_end)


if __name__ == "__main__":
    main()
