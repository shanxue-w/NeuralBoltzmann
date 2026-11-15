"""Compare the simplified Gate-FNO and BGK with the analytic BKW flow."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from boltzmann_discrete_maxwellian import DiscreteMaxwellian
from boltzmann_simplified_gate_fno import load_simplified_gate_model

DEFAULT_CHECKPOINT = REPO_ROOT / "runs/boltzmann_tail_simplified_gate_fno_w36_l3_neuralop_10000_b8_s1234_bkw/best.pt"


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


def make_grid(n: int, L: float) -> VelocityGrid:
    dv = 2.0 * L / n
    v = (np.arange(n, dtype=np.float64) - n // 2) * dv
    vx, vy, vz = np.meshgrid(v, v, v, indexing="ij")
    return VelocityGrid(n, L, dv, dv**3, v, vx, vy, vz, vx * vx + vy * vy + vz * vz)


def bkw_state(grid: VelocityGrid, K: float) -> np.ndarray:
    gaussian = np.exp(-grid.v2 / (2.0 * K)) / (2.0 * np.pi * K) ** 1.5
    polynomial = (5.0 * K - 3.0) / (2.0 * K) + (1.0 - K) * grid.v2 / (2.0 * K * K)
    return np.maximum(gaussian * polynomial, 0.0)


def error_metrics(value: np.ndarray, reference: np.ndarray, grid: VelocityGrid) -> dict[str, float]:
    difference = value - reference
    absolute_l1 = float(grid.dv3 * np.abs(difference).sum())
    absolute_l2 = float(np.sqrt(grid.dv3 * np.square(difference).sum()))
    reference_l1 = float(grid.dv3 * np.abs(reference).sum())
    reference_l2 = float(np.sqrt(grid.dv3 * np.square(reference).sum()))
    return {"absolute_l1": absolute_l1, "absolute_l2": absolute_l2, "relative_l1": absolute_l1 / reference_l1, "relative_l2": absolute_l2 / reference_l2}


def macroscopic_state(value: np.ndarray, grid: VelocityGrid) -> tuple[float, np.ndarray, float]:
    rho = float(grid.dv3 * value.sum())
    velocity = grid.dv3 * np.asarray([(grid.vx * value).sum(), (grid.vy * value).sum(), (grid.vz * value).sum()]) / rho
    radius2 = (grid.vx - velocity[0]) ** 2 + (grid.vy - velocity[1]) ** 2 + (grid.vz - velocity[2]) ** 2
    return rho, velocity, float(grid.dv3 * np.sum(radius2 * value) / (3.0 * rho))


def fourth_moment(value: np.ndarray, grid: VelocityGrid) -> float:
    rho, velocity, _ = macroscopic_state(value, grid)
    radius2 = (grid.vx - velocity[0]) ** 2 + (grid.vy - velocity[1]) ** 2 + (grid.vz - velocity[2]) ** 2
    return float(grid.dv3 * np.sum(radius2**2 * value) / rho)


class NeuralCollision:
    def __init__(self, checkpoint_path: Path, device: str, dtype: str, n: int) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.numpy_dtype = np.dtype(dtype)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model, config = load_simplified_gate_model(checkpoint, n=n)
        self.L = float(config["L"])
        self.model.to(device=self.device, dtype=getattr(torch, dtype)).eval()

    def __call__(self, states: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(np.ascontiguousarray(states, dtype=self.numpy_dtype)).to(self.device)
        with self.torch.inference_mode():
            collision = self.model(tensor)
        return collision.cpu().numpy().astype(np.float64, copy=False)


def discrete_equilibria(initial: np.ndarray, grid: VelocityGrid) -> np.ndarray:
    import torch

    with torch.inference_mode():
        equilibrium = DiscreteMaxwellian(n=grid.n, L=grid.L).double()(torch.from_numpy(initial))
    return equilibrium.numpy()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir: Path, grid: VelocityGrid, rows: list[dict[str, Any]], snapshots: dict[str, np.ndarray], profile_vmax: float, map_vmax: float) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/q_learning_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    labels = {"fno": "Neural", "bgk": "BGK", "exact": "Analytic BKW"}
    colors = {"fno": "#007C91", "bgk": "#E68613", "exact": "#202020"}
    times = np.asarray([row["time"] for row in rows])
    positive = times > 0.0

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    for method in ("fno", "bgk"):
        for axis, norm in zip(axes, ("l1", "l2")):
            values = np.asarray([row[f"{method}_relative_{norm}"] for row in rows])
            axis.semilogy(times[positive], values[positive], marker="o", markersize=3.5, color=colors[method], label=labels[method])
    for axis, norm in zip(axes, ("1", "2")):
        axis.set(xlabel=r"physical time $t$", ylabel=rf"relative $L^{norm}_v$ error")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    figure.savefig(output_dir / "state_error_vs_exact.png", dpi=220)
    plt.close(figure)

    center = grid.n // 2
    profile_mask = np.abs(grid.v) <= profile_vmax
    snapshot_times = snapshots["time"]
    figure, axes = plt.subplots(1, len(snapshot_times), figsize=(4.0 * len(snapshot_times), 3.45), sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    for column, time_value in enumerate(snapshot_times):
        axis = axes[0, column]
        for method in ("exact", "fno", "bgk"):
            axis.plot(grid.v[profile_mask], snapshots[method][column, :, center, center][profile_mask], color=colors[method], linewidth=1.8, label=labels[method])
        axis.set(title=rf"$t={time_value:g}$", xlabel=r"$v_x$ at $v_y=v_z=0$")
        axis.grid(alpha=0.18)
    axes[0, 0].set_ylabel(r"$f(t,v_x,0,0)$")
    axes[0, -1].legend(frameon=False, fontsize=9)
    figure.savefig(output_dir / "bkw_profiles.png", dpi=220)
    plt.close(figure)

    profile_indices = []
    for target_time in (1.0, 2.0):
        matches = np.flatnonzero(np.isclose(snapshot_times, target_time))
        if len(matches) != 1:
            raise ValueError("--snapshot-times must contain t=1 and t=2")
        profile_indices.append(int(matches[0]))

    profile_values = np.concatenate(
        [
            snapshots[method][snapshot_index, :, center, center][profile_mask]
            for snapshot_index in profile_indices
            for method in ("exact", "fno", "bgk")
        ]
    )
    profile_padding = 0.04 * (float(profile_values.max()) - float(profile_values.min()))
    profile_ylim = (float(profile_values.min()) - profile_padding, float(profile_values.max()) + profile_padding)
    for snapshot_index, filename in zip(
        profile_indices, ("bkw_profile_t1.pdf", "bkw_profile_t2.pdf")
    ):
        figure, axis = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
        for method in ("exact", "fno", "bgk"):
            axis.plot(
                grid.v[profile_mask],
                snapshots[method][snapshot_index, :, center, center][profile_mask],
                color=colors[method],
                linewidth=1.8,
                label=labels[method],
            )
        axis.set(xlabel=r"$v_x$", ylabel=r"$f(t,v_x,0,0)$", ylim=profile_ylim)
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, fontsize=9)
        figure.savefig(output_dir / filename, format="pdf", bbox_inches="tight")
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
    for method in ("fno", "bgk"):
        values = np.asarray([row[f"{method}_relative_l2"] for row in rows])
        axis.semilogy(
            times[positive],
            values[positive],
            marker="o",
            markersize=3.5,
            color=colors[method],
            label=labels[method],
        )
    axis.set(xlabel=r"$t$", ylabel=r"relative $L^2_v$ error")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=9)
    figure.savefig(output_dir / "bkw_relative_l2.pdf", format="pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.1, 3.9), constrained_layout=True)
    for method in ("exact", "fno", "bgk"):
        axis.plot(times, [row[f"{method}_fourth_moment"] for row in rows], color=colors[method], linewidth=1.8, label=labels[method])
    axis.set(xlabel=r"physical time $t$", ylabel=r"$m_4=\rho^{-1}\int_{\mathrm{R}^3}|v-u|^4f\,dv$")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(output_dir / "fourth_moment.png", dpi=220)
    plt.close(figure)

    exact_final = snapshots["exact"][-1]
    mask = np.abs(grid.v) <= map_vmax
    velocity = grid.v[mask]
    signed, absolute = {}, {}
    for method in ("fno", "bgk"):
        difference = snapshots[method][-1] - exact_final
        signed[method] = grid.dv * difference.sum(axis=2)[np.ix_(mask, mask)]
        absolute[method] = grid.dv * np.abs(difference).sum(axis=2)[np.ix_(mask, mask)]
    signed_limit = max(np.finfo(np.float64).tiny, *(float(np.abs(value).max()) for value in signed.values()))
    absolute_limit = max(np.finfo(np.float64).tiny, *(float(value.max()) for value in absolute.values()))
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    for column, method in enumerate(("fno", "bgk")):
        signed_image = axes[0, column].pcolormesh(velocity, velocity, signed[method].T, shading="nearest", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-signed_limit, vmax=signed_limit))
        absolute_image = axes[1, column].pcolormesh(velocity, velocity, absolute[method].T, shading="nearest", cmap="viridis", vmin=0.0, vmax=absolute_limit)
        axes[0, column].set_title(labels[method])
        for axis in axes[:, column]:
            axis.set(xlabel=r"$v_x$", ylabel=r"$v_y$")
            axis.set_aspect("equal")
    figure.colorbar(signed_image, ax=axes[0], shrink=0.82, label=r"$\int(f_{\rm method}-f_{\rm BKW})\,dv_z$")
    figure.colorbar(absolute_image, ax=axes[1], shrink=0.82, label=r"$\int|f_{\rm method}-f_{\rm BKW}|\,dv_z$")
    figure.savefig(output_dir / "final_vx_vy_difference_maps.png", dpi=220)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    initial_K = np.asarray(args.K, dtype=np.float64)
    neural = NeuralCollision(args.checkpoint, args.device, args.dtype, args.n)
    grid = make_grid(args.n, neural.L)
    steps = round(args.t_end / args.dt)
    snapshot_indices = {round(value / args.dt) for value in args.snapshot_times}
    initial = np.stack([bkw_state(grid, K) for K in initial_K])
    equilibria = discrete_equilibria(initial, grid)
    densities = np.asarray([macroscopic_state(value, grid)[0] for value in initial])
    for _ in range(args.warmup):
        neural(initial)

    rows = [[] for _ in initial_K]
    snapshots = [{"time": [], "K": [], "exact": [], "fno": [], "bgk": []} for _ in initial_K]
    current = initial.copy()
    bgk_current = initial.copy()
    for step in range(steps + 1):
        time_value = step * args.dt
        K_values = 1.0 - (1.0 - initial_K) * np.exp(-time_value / 6.0)
        exact = np.stack([bkw_state(grid, K) for K in K_values])
        for case_index in range(len(initial_K)):
            fno_error = error_metrics(current[case_index], exact[case_index], grid)
            bgk_error = error_metrics(bgk_current[case_index], exact[case_index], grid)
            rows[case_index].append({"time": time_value, "K": float(K_values[case_index]), **{f"fno_{key}": value for key, value in fno_error.items()}, **{f"bgk_{key}": value for key, value in bgk_error.items()}, "exact_fourth_moment": fourth_moment(exact[case_index], grid), "fno_fourth_moment": fourth_moment(current[case_index], grid), "bgk_fourth_moment": fourth_moment(bgk_current[case_index], grid)})
            if step in snapshot_indices:
                data = snapshots[case_index]
                data["time"].append(time_value)
                data["K"].append(float(K_values[case_index]))
                for method, values in (("exact", exact), ("fno", current), ("bgk", bgk_current)):
                    data[method].append(values[case_index].copy())
        if step == steps:
            break
        first = current + args.dt * neural(current)
        current = 0.5 * current + 0.5 * (first + args.dt * neural(first))
        bgk_first = bgk_current + args.dt * 0.5 * densities[:, None, None, None] * (equilibria - bgk_current)
        bgk_current = 0.5 * bgk_current + 0.5 * (bgk_first + args.dt * 0.5 * densities[:, None, None, None] * (equilibria - bgk_first))
        print(f"step {step + 1:02d}/{steps:02d}, t={(step + 1) * args.dt:.2f}", flush=True)

    root_summary: dict[str, Any] = {"equation": "partial_t f = Q(f,f)", "K_of_time": "K(t) = 1 - (1-K0) exp(-t/6)", "checkpoint": str(args.checkpoint), "device": str(args.device), "dtype": args.dtype, "dt": args.dt, "t_end": args.t_end, "cases": {}}
    metrics = ("absolute_l1", "absolute_l2", "relative_l1", "relative_l2")
    for case_index, K0 in enumerate(initial_K):
        name = f"K{K0:g}".replace(".", "p")
        output_dir = args.output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_arrays = {key: np.asarray(value) for key, value in snapshots[case_index].items()}
        write_csv(output_dir / "time_series.csv", rows[case_index])
        np.savez_compressed(output_dir / "snapshots.npz", **snapshot_arrays)
        make_plots(output_dir, grid, rows[case_index], snapshot_arrays, args.profile_vmax, args.map_vmax)
        final_row = rows[case_index][-1]
        summary = {
            "initial_K": float(K0), "final_K": float(final_row["K"]), "grid": {"n": grid.n, "L": grid.L, "dv": grid.dv},
            "time_integration": {"neural_operator": "SSPRK2", "BGK": "SSPRK2 with nu=rho/2", "dt": args.dt, "t_end": args.t_end},
            "fourth_moment_definition": "rho^(-1) integral |v-u|^4 f dv",
            "difference_map_definition": {"signed": "integral (f_method-f_BKW) dv_z", "absolute": "integral |f_method-f_BKW| dv_z", "displayed_vx_vy_window": [-args.map_vmax, args.map_vmax]},
            "final_errors": {method: {key: final_row[f"{method}_{key}"] for key in metrics} for method in ("fno", "bgk")},
            "files": {"errors_and_moments": "time_series.csv", "profiles": "bkw_profiles.png", "profile_t1": "bkw_profile_t1.pdf", "profile_t2": "bkw_profile_t2.pdf", "relative_l2": "bkw_relative_l2.pdf", "fourth_moment": "fourth_moment.png", "difference_maps": "final_vx_vy_difference_maps.png", "snapshots": "snapshots.npz"},
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        root_summary["cases"][name] = summary
        print(f"{name}: final relative L2, neural operator={final_row['fno_relative_l2']:.6e}, BGK={final_row['bgk_relative_l2']:.6e}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(root_summary, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare the neural operator and BGK with analytic BKW flows.")
    parser.add_argument("--K", type=float, nargs="+", default=[0.6])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--snapshot-times", type=float, nargs="+", default=[0, 1, 2])
    parser.add_argument("--profile-vmax", type=float, default=5.0)
    parser.add_argument("--map-vmax", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    run(args)


if __name__ == "__main__":
    main()
