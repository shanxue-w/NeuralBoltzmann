"""Check conservation, nonnegativity, and entropy along the three-stream rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.homogeneous.run_homogeneous import (
    NeuralCollision,
    integrate_ssprk2,
    make_grid,
    moments,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir", type=Path,
        default=Path(__file__).parent / "outputs" / "three_stream",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    output_dir = (args.output_dir or case_dir / "structure_verification").resolve()
    summary = json.loads((case_dir / "summary.json").read_text())
    checkpoint = Path(summary["checkpoint"])
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if checkpoint_hash != summary["checkpoint_sha256"]:
        raise ValueError("The checkpoint does not match the archived three-stream run.")

    with np.load(case_dir / "snapshots.npz", allow_pickle=False) as data:
        initial = data["initial"]
        equilibrium = data["equilibrium"]
        saved_tau = data["tau"]
        saved_states = data["fno_states"]
    grid = make_grid(summary["grid"]["n"], summary["grid"]["L"])
    dtau = summary["time_integration"]["dtau"]
    tau_end = summary["time_integration"]["tau_end"]
    steps = round(tau_end / dtau)
    tau = np.arange(steps + 1) * dtau

    # Reuse the original FP32 collision wrapper and FP64 SSPRK2 state updates.
    # The time integrator applies neither clipping nor moment renormalization.
    rhs = NeuralCollision(checkpoint, grid, args.device)
    try:
        states, _, minimum_stage = integrate_ssprk2(
            "three-stream structure check", rhs, initial, dtau, steps, set()
        )
    finally:
        rhs.close()
    if not np.isfinite(states).all():
        raise ValueError("The rollout contains nonfinite distribution values.")

    physical_scale = np.array([1.0, grid.L, grid.L, grid.L, 0.5 * grid.L**2])
    physical_moments = np.stack([moments(f, grid) * physical_scale for f in states])
    drift = physical_moments - physical_moments[0]
    minima = states.reshape(steps + 1, -1).min(axis=1)
    entropy = np.full(steps + 1, np.nan)
    relative_entropy = np.full(steps + 1, np.nan)
    if not (np.isfinite(equilibrium).all() and np.all(equilibrium > 0.0)):
        raise ValueError("The archived moment-matched equilibrium must be positive.")
    log_equilibrium = np.log(equilibrium)
    for index, f in enumerate(states):
        if minima[index] < 0.0:
            continue
        positive = f > 0.0
        # Evaluate x log x with 0 log 0 = 0, without flooring positive values.
        entropy[index] = grid.dv3 * np.sum(f[positive] * np.log(f[positive]))
        relative_entropy[index] = grid.dv3 * np.sum(
            f[positive] * (np.log(f[positive]) - log_equilibrium[positive])
        )

    moment_names = ("mass", "momentum_x", "momentum_y", "momentum_z", "energy")
    rows = []
    for index, time_value in enumerate(tau):
        row = {"tau": float(time_value)}
        for column, name in enumerate(moment_names):
            row[name] = float(physical_moments[index, column])
            row[f"{name}_drift"] = float(drift[index, column])
        row.update(
            minimum_f=float(minima[index]),
            entropy=float(entropy[index]),
            relative_entropy=float(relative_entropy[index]),
        )
        rows.append(row)

    saved_indices = np.rint(saved_tau / dtau).astype(int)
    if not np.allclose(tau[saved_indices], saved_tau):
        raise ValueError("The archived snapshot times are off the timestep grid.")
    differences = states[saved_indices] - saved_states
    relative_l2 = np.linalg.norm(differences.reshape(len(saved_tau), -1), axis=1)
    relative_l2 /= np.linalg.norm(saved_states.reshape(len(saved_tau), -1), axis=1)
    entropy_finite = bool(np.isfinite(entropy).all())
    report = {
        "case": summary["case"],
        "source_case_dir": str(case_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if torch.device(args.device).type == "cuda" else None,
        "collision_dtype": "float32",
        "state_dtype": "float64",
        "grid": summary["grid"],
        "time_integration": {"method": "SSPRK2", "dtau": dtau, "tau_end": tau_end},
        "saved_states_checked": steps + 1,
        "rk_stages_checked": 2 * steps,
        "state_clipping": False,
        "state_moment_renormalization": False,
        "max_absolute_moment_drift": {
            name: float(np.abs(drift[:, column]).max())
            for column, name in enumerate(moment_names)
        },
        "max_relative_mass_drift": float(np.abs(drift[:, 0]).max() / physical_moments[0, 0]),
        "max_relative_energy_drift": float(np.abs(drift[:, 4]).max() / physical_moments[0, 4]),
        "minimum_over_states_and_stages": minimum_stage,
        "minimum_state_at_positive_time": float(minima[1:].min()),
        "negative_saved_values": int(np.count_nonzero(states < 0.0)),
        "nonnegative_in_all_states_and_stages": bool(minimum_stage >= 0.0),
        "entropy_definition": "dv^3 sum(f log f), 0 log 0 = 0",
        "entropy_initial": float(entropy[0]),
        "entropy_final": float(entropy[-1]),
        "max_entropy_increment": float(np.diff(entropy).max()) if entropy_finite else None,
        "entropy_nonincreasing_at_every_step": bool(entropy_finite and np.all(np.diff(entropy) <= 0.0)),
        "relative_entropy_initial": float(relative_entropy[0]),
        "relative_entropy_final": float(relative_entropy[-1]),
        "max_snapshot_relative_l2_difference": float(relative_l2.max()),
        "archived_snapshot_times": saved_tau.tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "time_series.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
