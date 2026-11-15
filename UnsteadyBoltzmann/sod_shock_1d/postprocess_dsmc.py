"""Terminal DSMC arrays and profile plots."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from UnsteadyBoltzmann.runtime import json_ready


REPO_ROOT = Path(__file__).resolve().parents[2]


def save_results(
    output_dir: Path,
    *,
    solver: Any,
    instantaneous_moments: np.ndarray,
    run_configuration: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if solver.sample_moments:
        history = np.stack(solver.sample_moments)
        averaged = history.mean(axis=0)
        standard_error = history.std(axis=0, ddof=1) / math.sqrt(history.shape[0]) if history.shape[0] > 1 else np.full_like(averaged, np.nan)
        averaged_vx = solver.vx_marginal_sum / history.shape[0]
    else:
        history = np.empty((0, solver.nx, 5), dtype=np.float64)
        averaged = instantaneous_moments.copy()
        standard_error = np.full_like(averaged, np.nan)
        averaged_vx = np.zeros_like(solver.vx_marginal_sum)

    dv = 2.0 * solver.velocity_half_width / solver.velocity_bins
    vx_centers = -solver.velocity_half_width + (np.arange(solver.velocity_bins, dtype=np.float64) + 0.5) * dv
    positions = solver.positions[: solver.particle_count]
    velocities = solver.velocities[: solver.particle_count]
    cell = np.floor((positions - solver.x_left) * solver.inverse_dx).astype(np.int64)
    np.clip(cell, 0, solver.nx - 1, out=cell)
    vx_bin = np.floor((velocities[:, 0] + solver.velocity_half_width) / dv).astype(np.int64)
    inside = (vx_bin >= 0) & (vx_bin < solver.velocity_bins)
    flat = cell[inside] * solver.velocity_bins + vx_bin[inside]
    final_vx = np.bincount(flat, minlength=solver.nx * solver.velocity_bins).reshape(solver.nx, solver.velocity_bins)
    final_vx = final_vx * (solver.particle_weight / (solver.dx * dv))

    np.save(output_dir / "x.npy", solver.x_centers)
    np.save(output_dir / "final_moments.npy", instantaneous_moments)
    np.save(output_dir / "averaged_moments.npy", averaged)
    np.save(output_dir / "moment_standard_error.npy", standard_error)
    np.save(output_dir / "vx_centers.npy", vx_centers)
    np.save(output_dir / "final_vx_marginal.npy", final_vx)
    np.save(output_dir / "averaged_vx_marginal.npy", averaged_vx)
    np.savez(output_dir / "sampling_history.npz", times=np.asarray(solver.sample_times, dtype=np.float64), moments=history)
    if solver.full_distribution_sum is not None and solver.sample_count:
        np.save(output_dir / "averaged_f.npy", solver.full_distribution_sum / solver.sample_count)
        full_dv = 2.0 * solver.velocity_half_width / solver.full_distribution_nv
        centers = -solver.velocity_half_width + (np.arange(solver.full_distribution_nv, dtype=np.float64) + 0.5) * full_dv
        np.save(output_dir / "velocity_centers.npy", centers)

    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for column, ylabel, title, filename in (
        (0, r"$\rho$", "Final DSMC density", "rho_x.pdf"),
        (1, r"$u_x$", "Final DSMC velocity", "u_x.pdf"),
        (4, r"$T$", "Final DSMC temperature", "T_x.pdf"),
    ):
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot(solver.x_centers, instantaneous_moments[:, column], linewidth=1.8)
        axis.set(xlabel=r"$x$", ylabel=ylabel, title=title)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(figure)

    (output_dir / "run_config.json").write_text(
        json.dumps(json_ready(run_configuration), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
