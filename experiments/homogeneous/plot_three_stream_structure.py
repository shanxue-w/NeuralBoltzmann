"""Plot the three-stream conservation, minimum distribution, and entropy histories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/q_learning_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    Path(__file__).parent
    / "outputs/three_stream/structure_verification/time_series.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "paper/figures")
    args = parser.parse_args()
    data = np.genfromtxt(args.data, delimiter=",", names=True)
    time = data["tau"] / data["mass"][0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5.0, 4.4), constrained_layout=True)
    curves = (
        ("mass_drift", r"$\rho$", "#007C91", "o"),
        ("momentum_x_drift", r"$\rho u_1$", "#F59E0B", "s"),
        ("momentum_y_drift", r"$\rho u_2$", "#7C3AED", "^"),
        ("momentum_z_drift", r"$\rho u_3$", "#2E7D32", "v"),
        ("energy_drift", r"$E$", "#202020", "D"),
    )
    for field, label, color, marker in curves:
        # Exact zeros, including the initial error, have no logarithmic ordinate.
        error = np.ma.masked_less_equal(np.abs(data[field]), 0.0)
        ax.semilogy(
            time, error, label=label, color=color, linewidth=1.7,
            marker=marker, markevery=5, markersize=3.5,
        )
    ax.set(xlabel=r"$t$", ylabel="Absolute conservation error", xlim=(0.0, time[-1]))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.savefig(args.output_dir / "three_stream_conservation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.4), constrained_layout=True)
    ax.plot(
        time, data["minimum_f"], color="#007C91", linewidth=2.0,
        marker="o", markevery=5, markersize=4,
    )
    ax.set(xlabel=r"$t$", ylabel=r"$\min_j f_j(t)$", xlim=(0.0, time[-1]))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.22)
    fig.savefig(args.output_dir / "three_stream_minimum_f.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.4), constrained_layout=True)
    ax.plot(
        time, data["relative_entropy"], color="#007C91", linewidth=2.0,
        marker="o", markevery=5, markersize=4,
    )
    ax.set(
        xlabel=r"$t$", ylabel=r"$\mathscr{H}_h(f(t)\mid M[f_0])$",
        xlim=(0.0, time[-1]),
    )
    ax.grid(alpha=0.22)
    fig.savefig(args.output_dir / "three_stream_entropy.pdf")
    plt.close(fig)
    print(f"Saved three PDF figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
