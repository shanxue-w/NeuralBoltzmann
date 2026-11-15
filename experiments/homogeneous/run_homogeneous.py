"""Homogeneous two-stream and three-stream relaxation benchmarks."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from boltzmann_discrete_maxwellian import DiscreteMaxwellian
from boltzmann_fft_gpu import BoltzmannFFT
from boltzmann_simplified_gate_fno import load_simplified_gate_model

DEFAULT_CHECKPOINT = REPO_ROOT / "runs/boltzmann_tail_simplified_gate_fno_w36_l3_neuralop_10000_b8_s1234_bkw/best.pt"
REFERENCE_ACCURACY = 23


@dataclass(frozen=True)
class VelocityGrid:
    n: int
    L: float
    dv: float
    dv3: float
    v: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    v2: np.ndarray
    psi: np.ndarray


def make_grid(n: int, L: float) -> VelocityGrid:
    dv = 2.0 * L / n
    v = (np.arange(n, dtype=np.float64) - n // 2) * dv
    vx, vy, vz = np.meshgrid(v, v, v, indexing="ij")
    v2 = vx * vx + vy * vy + vz * vz
    psi = np.stack((np.ones_like(vx), vx / L, vy / L, vz / L, v2 / (L * L)), axis=-1)
    return VelocityGrid(n, L, dv, dv**3, v, vx, vy, vz, v2, psi)


def checkpoint_velocity_half_width(checkpoint_path: Path) -> float:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return float(checkpoint["model_config"]["L"])


def integrate(value: np.ndarray, grid: VelocityGrid) -> float:
    return float(grid.dv3 * value.sum())


def moments(value: np.ndarray, grid: VelocityGrid) -> np.ndarray:
    return grid.dv3 * (value.reshape(-1, 1) * grid.psi.reshape(-1, 5)).sum(axis=0)


def normalized_maxwellian(grid: VelocityGrid, center: tuple[float, float, float], temperature: float) -> np.ndarray:
    exponent = -((grid.vx - center[0]) ** 2 + (grid.vy - center[1]) ** 2 + (grid.vz - center[2]) ** 2) / (2.0 * temperature)
    value = np.exp(exponent) / (2.0 * np.pi * temperature) ** 1.5
    return value / integrate(value, grid)


def initial_condition(name: str, grid: VelocityGrid) -> tuple[np.ndarray, dict[str, Any]]:
    data = {
        "two_stream": ((0.5, 0.5), ((2.5, 0.0, 0.0), (-2.5, 0.0, 0.0))),
        "three_stream": ((0.4, 0.35, 0.25), ((2.5, 0.0, 0.0), (-1.6, 1.4, 0.0), (-1.76, -1.96, 0.0))),
    }
    weights, centers = data[name]
    temperature = 0.6
    value = sum(weight * normalized_maxwellian(grid, center, temperature) for weight, center in zip(weights, centers))
    return value, {"weights": list(weights), "centers": [list(center) for center in centers], "component_temperature": temperature}


def macroscopic_state(value: np.ndarray, grid: VelocityGrid) -> tuple[float, np.ndarray, float]:
    physical = moments(value, grid)
    rho = float(physical[0])
    velocity = grid.L * physical[1:4] / rho
    return rho, velocity, (grid.L * grid.L * physical[4] / rho - float(np.dot(velocity, velocity))) / 3.0


def error_metrics(value: np.ndarray, reference: np.ndarray, grid: VelocityGrid) -> dict[str, float]:
    difference = value - reference
    absolute_l1 = float(grid.dv3 * np.abs(difference).sum())
    absolute_l2 = float(np.sqrt(grid.dv3 * np.square(difference).sum()))
    reference_l1 = float(grid.dv3 * np.abs(reference).sum())
    reference_l2 = float(np.sqrt(grid.dv3 * np.square(reference).sum()))
    return {"absolute_l1": absolute_l1, "absolute_l2": absolute_l2, "relative_l1": absolute_l1 / reference_l1, "relative_l2": absolute_l2 / reference_l2}


class SpectralCollision:
    def __init__(self, grid: VelocityGrid, R: float, angular_accuracy: int, angular_chunk: int) -> None:
        import cupy as cp

        self.cp = cp
        self.solver = BoltzmannFFT(n=grid.n, L=grid.L, R=R, radial_accuracy=grid.n, angular_accuracy=angular_accuracy, omega=1.0, kernel_constant=1.0 / (4.0 * math.pi), backend="cupy", dtype=np.float64, loss_mode="sinc", angular_chunk=angular_chunk, use_rfft=True, enforce_mass=True, lebedev_dir=REPO_ROOT / "Lebedev")

    def __call__(self, value: np.ndarray) -> np.ndarray:
        collision = self.solver.collision(value, batch_chunk=1)
        self.cp.cuda.get_current_stream().synchronize()
        return self.cp.asnumpy(collision).astype(np.float64, copy=False)

    def close(self) -> None:
        del self.solver
        self.cp.get_default_memory_pool().free_all_blocks()
        self.cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()


class NeuralCollision:
    def __init__(self, checkpoint_path: Path, grid: VelocityGrid, device: str) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model, _ = load_simplified_gate_model(checkpoint, n=grid.n)
        self.model.to(device=self.device, dtype=torch.float32).eval()

    def __call__(self, value: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))[None].to(self.device)
        with self.torch.inference_mode():
            collision = self.model(tensor)
        return collision[0].cpu().numpy().astype(np.float64, copy=False)

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()
        gc.collect()


def discrete_equilibrium(initial: np.ndarray, grid: VelocityGrid) -> np.ndarray:
    import torch

    with torch.inference_mode():
        equilibrium = DiscreteMaxwellian(n=grid.n, L=grid.L).double()(torch.from_numpy(initial)[None])
    return equilibrium[0].numpy()


def integrate_ssprk2(name: str, rhs: Any, initial: np.ndarray, dt: float, steps: int, snapshot_indices: set[int]) -> tuple[np.ndarray, dict[int, np.ndarray], float]:
    states = np.empty((steps + 1, *initial.shape), dtype=np.float64)
    states[0] = initial
    q_snapshots: dict[int, np.ndarray] = {}
    minimum = float(initial.min())
    stride = max(1, steps // 10)
    for step in range(steps):
        current = states[step]
        q0 = rhs(current)
        if step in snapshot_indices:
            q_snapshots[step] = q0.copy()
        first = current + dt * q0
        following = 0.5 * current + 0.5 * (first + dt * rhs(first))
        minimum = min(minimum, float(first.min()), float(following.min()))
        states[step + 1] = following
        if (step + 1) % stride == 0:
            print(f"  {name}: {step + 1:02d}/{steps:02d}", flush=True)
    if steps in snapshot_indices:
        q_snapshots[steps] = rhs(states[-1])
    return states, q_snapshots, minimum


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir: Path, grid: VelocityGrid, tau: np.ndarray, trajectories: dict[str, np.ndarray], rows: list[dict[str, Any]], snapshot_tau: np.ndarray, q_snapshots: dict[str, np.ndarray], map_vmax: float, baseline_method: str, baseline_accuracy: int) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/q_learning_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D

    labels = {"reference": rf"Spectral $a={REFERENCE_ACCURACY}$", "fno": "Neural operator", baseline_method: rf"Spectral $a={baseline_accuracy}$"}
    colors = {"reference": "#202020", "fno": "#007C91", baseline_method: "#6D44A6"}
    compared = ("fno", baseline_method)
    positive = tau > 0.0
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), constrained_layout=True)
    for method in compared:
        for axis, norm in zip(axes, ("l1", "l2")):
            values = np.asarray([row[f"{method}_relative_{norm}"] for row in rows])
            axis.semilogy(tau[positive], values[positive], marker="o", markersize=3.2, color=colors[method], label=labels[method])
    for axis, norm in zip(axes, ("1", "2")):
        axis.set(xlabel=r"dimensionless time $\tau=\rho t$", ylabel=rf"relative $L^{norm}_v$ error")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    figure.savefig(output_dir / "state_error_vs_reference.png", dpi=220)
    plt.close(figure)

    reference_final = trajectories["reference"][-1]
    mask = np.abs(grid.v) <= map_vmax
    velocity = grid.v[mask]
    signed, absolute = {}, {}
    for method in compared:
        difference = trajectories[method][-1] - reference_final
        signed[method] = grid.dv * difference.sum(axis=2)[np.ix_(mask, mask)]
        absolute[method] = grid.dv * np.abs(difference).sum(axis=2)[np.ix_(mask, mask)]
    signed_limit = max(np.finfo(np.float64).tiny, *(float(np.abs(value).max()) for value in signed.values()))
    absolute_limit = max(np.finfo(np.float64).tiny, *(float(value.max()) for value in absolute.values()))
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), constrained_layout=True)
    for column, method in enumerate(compared):
        signed_image = axes[0, column].pcolormesh(velocity, velocity, signed[method].T, shading="nearest", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-signed_limit, vmax=signed_limit))
        absolute_image = axes[1, column].pcolormesh(velocity, velocity, absolute[method].T, shading="nearest", cmap="viridis", vmin=0.0, vmax=absolute_limit)
        axes[0, column].set_title(labels[method])
        for axis in axes[:, column]:
            axis.set(xlabel=r"$v_x$", ylabel=r"$v_y$")
            axis.set_aspect("equal")
    figure.colorbar(signed_image, ax=axes[0], shrink=0.82, label=rf"$\int(f_{{\rm method}}-f_{{a={REFERENCE_ACCURACY}}})\,dv_z$")
    figure.colorbar(absolute_image, ax=axes[1], shrink=0.82, label=rf"$\int|f_{{\rm method}}-f_{{a={REFERENCE_ACCURACY}}}|\,dv_z$")
    figure.savefig(output_dir / "final_vx_vy_difference_maps.png", dpi=220)
    plt.close(figure)

    cmap = LinearSegmentedColormap.from_list("collision_ocean_teal", ("#06184F", "#075985", "#0891B2", "#9FE3D4", "#19B77A", "#07543A"))
    q_marginals = {method: grid.dv * values.sum(axis=3) for method, values in q_snapshots.items()}
    figure, axes = plt.subplots(1, len(snapshot_tau), figsize=(4.5 * len(snapshot_tau), 4.8), constrained_layout=True, squeeze=False)
    for column, time_value in enumerate(snapshot_tau):
        reference_q = q_marginals["reference"][column]
        limit = max(float(np.abs(q_marginals[method][column]).max()) for method in ("reference", "fno", baseline_method))
        levels = np.linspace(-0.8 * limit, 0.8 * limit, 9)
        levels = levels[np.abs(levels) > 1.0e-14 * limit]
        axis = axes[0, column]
        image = axis.contourf(grid.v, grid.v, reference_q.T, levels=np.linspace(-limit, limit, 65), cmap=cmap, extend="neither")
        for method, color, style, width in (("reference", "#171717", "dashed", 0.85), ("fno", "#F59E0B", "solid", 1.35), (baseline_method, "#7C3AED", "solid", 1.1)):
            axis.contour(grid.v, grid.v, q_marginals[method][column].T, levels=levels, colors=color, linestyles=style, linewidths=width)
        axis.set(xlim=(-6.0, 6.0), ylim=(-6.0, 6.0), xlabel=r"$v_x$", title=rf"$\tau={time_value:g}$")
        axis.set_aspect("equal")
        if column == 0:
            axis.set_ylabel(r"$v_y$")
        colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", fraction=0.045, pad=0.11, shrink=0.68, aspect=28, ticks=(-limit, 0.0, limit))
        colorbar.ax.tick_params(labelsize=8)
    axes[0, 0].legend(handles=(Line2D([], [], color="#171717", linestyle="--", label=labels["reference"]), Line2D([], [], color="#F59E0B", label=labels["fno"]), Line2D([], [], color="#7C3AED", label=labels[baseline_method])), loc="lower left", fontsize=7.5, framealpha=0.78)
    figure.savefig(output_dir / "collision_vx_vy_marginals_over_time.pdf", format="pdf", bbox_inches="tight")
    plt.close(figure)


def load_or_compute_reference(args: argparse.Namespace, case_name: str, initial: np.ndarray, grid: VelocityGrid, steps: int, snapshot_indices: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any], bool]:
    cache_dir = args.reference_cache_dir / case_name
    states_path = cache_dir / "states.npy"
    collision_path = cache_dir / "collision.npy"
    manifest_path = cache_dir / "manifest.json"
    signature = {"schema_version": 1, "case": case_name, "n": grid.n, "L": grid.L, "R": args.R, "radial_accuracy": grid.n, "angular_accuracy": REFERENCE_ACCURACY, "angular_chunk": args.angular_chunk, "kernel_constant": "1/(4*pi)", "loss_mode": "sinc", "dtype": "float64", "integrator": "uncorrected SSPRK2", "dtau": args.dtau, "tau_end": args.tau_end, "steps": steps}
    if not args.refresh_reference_cache and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest["signature"].get(key) == value for key, value in signature.items()):
            states = np.load(states_path)
            collision = np.load(collision_path)
            print(f"[{case_name}] loaded cached spectral a={REFERENCE_ACCURACY} reference", flush=True)
            return states, {index: collision[index].copy() for index in snapshot_indices}, manifest["trajectory_diagnostics"], True

    print(f"[{case_name}] spectral a={REFERENCE_ACCURACY} reference", flush=True)
    rhs = SpectralCollision(grid, args.R, REFERENCE_ACCURACY, args.angular_chunk)
    for _ in range(args.warmup):
        rhs(initial)
    states, all_collisions, minimum = integrate_ssprk2(f"spectral a={REFERENCE_ACCURACY}", rhs, initial, args.dtau, steps, set(range(steps + 1)))
    rhs.close()
    collision = np.stack([all_collisions[index] for index in range(steps + 1)])
    diagnostics = {"positivity_correction": False, "total_stages": 2 * steps, "minimum_stage_value": minimum, "minimum_saved_state": float(states.min())}
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(states_path, states)
    np.save(collision_path, collision)
    manifest_path.write_text(json.dumps({"signature": signature, "trajectory_diagnostics": diagnostics}, indent=2) + "\n", encoding="utf-8")
    print(f"[{case_name}] saved spectral a={REFERENCE_ACCURACY} reference cache", flush=True)
    return states, {index: collision[index].copy() for index in snapshot_indices}, diagnostics, False


def run_case(
    args: argparse.Namespace, name: str, grid: VelocityGrid, *, include_bgk: bool = True
) -> dict[str, Any]:
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    initial, initial_metadata = initial_condition(name, grid)
    equilibrium = discrete_equilibrium(initial, grid) if include_bgk else None
    steps = round(args.tau_end / args.dtau)
    tau = np.arange(steps + 1, dtype=np.float64) * args.dtau
    snapshot_indices = [round(value / args.dtau) for value in args.snapshot_tau]
    snapshot_set = set(snapshot_indices)
    baseline_accuracy = args.baseline_angular_accuracy
    baseline_method = f"spectral_a{baseline_accuracy}"

    reference_states, reference_q, reference_diagnostics, cache_used = load_or_compute_reference(args, name, initial, grid, steps, snapshot_indices)
    trajectories = {"reference": reference_states}
    collisions = {"reference": reference_q}
    diagnostics = {"reference": reference_diagnostics}

    print(f"[{name}] spectral a={baseline_accuracy}", flush=True)
    rhs = SpectralCollision(grid, args.R, baseline_accuracy, args.angular_chunk)
    for _ in range(args.warmup):
        rhs(initial)
    states, q_values, minimum = integrate_ssprk2(f"spectral a={baseline_accuracy}", rhs, initial, args.dtau, steps, snapshot_set)
    rhs.close()
    trajectories[baseline_method] = states
    collisions[baseline_method] = q_values
    diagnostics[baseline_method] = {"positivity_correction": False, "total_stages": 2 * steps, "minimum_stage_value": minimum, "minimum_saved_state": float(states.min())}

    print(f"[{name}] neural operator", flush=True)
    neural = NeuralCollision(args.checkpoint, grid, args.device)
    for _ in range(args.warmup):
        neural(initial)
    trajectories["fno"], collisions["fno"], _ = integrate_ssprk2("neural operator", neural, initial, args.dtau, steps, snapshot_set)
    neural.close()

    if include_bgk:
        rho = integrate(initial, grid)
        bgk_rhs = lambda value: 0.5 * rho * (equilibrium - value)
        trajectories["bgk"], collisions["bgk"], _ = integrate_ssprk2("BGK", bgk_rhs, initial, args.dtau, steps, snapshot_set)

    compared_methods = ("fno", "bgk", baseline_method) if include_bgk else ("fno", baseline_method)
    rows: list[dict[str, Any]] = []
    for index, time_value in enumerate(tau):
        row: dict[str, Any] = {"tau": time_value}
        reference = trajectories["reference"][index]
        for method in compared_methods:
            row.update({f"{method}_{key}": value for key, value in error_metrics(trajectories[method][index], reference, grid).items()})
        rows.append(row)
    write_csv(output_dir / "time_series.csv", rows)

    selector = np.asarray(snapshot_indices, dtype=np.int64)
    methods = ("reference", *compared_methods)
    payload: dict[str, np.ndarray] = {"tau": tau[selector], "initial": initial}
    if include_bgk:
        payload["equilibrium"] = equilibrium
    for method in methods:
        payload[f"{method}_states"] = trajectories[method][selector]
        payload[f"{method}_collision"] = np.stack([collisions[method][index] for index in snapshot_indices])
    np.savez_compressed(output_dir / "snapshots.npz", **payload)
    make_plots(output_dir, grid, tau, trajectories, rows, tau[selector], {method: payload[f"{method}_collision"] for method in methods}, args.map_vmax, baseline_method, baseline_accuracy)

    final_row = rows[-1]
    metrics = ("absolute_l1", "absolute_l2", "relative_l1", "relative_l2")
    summary = {
        "case": name, "initial_condition": initial_metadata,
        "grid": {"n": grid.n, "L": grid.L, "R": args.R, "radial_accuracy": grid.n, "reference_angular_accuracy": REFERENCE_ACCURACY, "baseline_angular_accuracy": baseline_accuracy, "kernel_constant": "1/(4*pi)", "loss_mode": "sinc"},
        "time_integration": {"integrator": f"SSPRK2 for all {len(methods)} trajectories", "spectral_state_correction": "none", "dtau": args.dtau, "tau_end": args.tau_end, "snapshot_tau": args.snapshot_tau},
        "reference": f"independent spectral a={REFERENCE_ACCURACY} trajectory", "reference_cache": {"path": str(args.reference_cache_dir / name), "loaded": cache_used},
        "checkpoint": str(args.checkpoint), "spectral_positivity_repairs": diagnostics,
        "final_errors": {method: {key: final_row[f"{method}_{key}"] for key in metrics} for method in compared_methods},
        "files": {"errors": "time_series.csv", "difference_maps": "final_vx_vy_difference_maps.png", "collision_marginals": "collision_vx_vy_marginals_over_time.pdf", "snapshots": "snapshots.npz"},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    final_report = ", ".join(
        f"{method}={final_row[f'{method}_relative_l2']:.6e}" for method in compared_methods
    )
    print(f"[{name}] final relative L2: {final_report}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Homogeneous two-stream and three-stream relaxation benchmark.")
    parser.add_argument("--case", nargs="+", choices=("two_stream", "three_stream"), default=["two_stream", "three_stream"])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--R", type=float, default=9.0)
    parser.add_argument("--baseline-angular-accuracy", type=int, default=9)
    parser.add_argument("--angular-chunk", type=int, default=16)
    parser.add_argument("--reference-cache-dir", type=Path, default=Path(__file__).parent / "reference_cache")
    parser.add_argument("--refresh-reference-cache", action="store_true")
    parser.add_argument("--dtau", type=float, default=0.1)
    parser.add_argument("--tau-end", type=float, default=5.0)
    parser.add_argument("--snapshot-tau", type=float, nargs="+", default=[0, 0.5, 1, 2, 5])
    parser.add_argument("--map-vmax", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.reference_cache_dir = args.reference_cache_dir.resolve()
    grid = make_grid(args.n, checkpoint_velocity_half_width(args.checkpoint))
    summaries = {name: run_case(args, name, grid) for name in args.case}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
