"""Plot saved discontinuous-experiment data without rerunning the solvers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/q_learning_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent


def save_pdf(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(path)


def plot_final(output, v, fields, baseline_label):
    limit = max(float(np.abs(q).max()) for q in fields.values())
    levels = np.linspace(-0.8 * limit, 0.8 * limit, 9)
    levels = levels[np.abs(levels) > 1e-14 * limit]
    levels = np.concatenate((levels[levels < 0], levels[levels > 0][:3]))
    fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")
    background = ax.contourf(
        v, v, fields["reference"].T,
        levels=np.linspace(-limit, limit, 65), cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    styles = (
        ("spectral", "#7C3AED", "-", 1.5, baseline_label.replace("$", "")),
        ("fno", "#F59E0B", "-", 1.5, "Neural"),
        ("reference", "#171717", "--", 1.0, "Reference p=23"),
    )
    handles = []
    for method, color, style, width, label in styles:
        ax.contour(
            v, v, fields[method].T, levels=levels,
            colors=color, linestyles=style, linewidths=width,
        )
        handles.append(Line2D([], [], color=color, ls=style, lw=width, label=label))
    ax.set(xlim=(-6, 6), ylim=(-6, 6), xlabel=r"$v_x$", ylabel=r"$v_y$")
    ax.set_aspect("equal")
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    fig.colorbar(background, ax=ax, shrink=0.85)
    save_pdf(fig, output / "final_collision_field.pdf")


def plot_errors(output, table, baseline, baseline_label):
    # All data have unit initial mass, so the stored tau equals physical t.
    t = table["tau"]
    for norm in (1, 2):
        fig, ax = plt.subplots(figsize=(4.8, 3.7), layout="constrained")
        for method, color, label in (
            ("fno", "#007C91", "Neural"),
            (baseline, "#7C3AED", baseline_label),
        ):
            error = table[f"{method}_relative_l{norm}"]
            positive = (t > 0) & (error > 0)
            ax.semilogy(t[positive], error[positive], color=color, lw=1.8, label=label)
        ax.set(xlabel=r"$t$", ylabel=rf"Relative $L^{norm}$ error", xlim=(0, t[-1]))
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        save_pdf(fig, output / f"relative_l{norm}_error.pdf")


def plot_initial_field(output, filename, v, field, norm, jump_ratio=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    options = dict(shading="gouraud", cmap="turbo", norm=norm, rasterized=True)
    if jump_ratio is None:
        image = ax.pcolormesh(v, v, field.T, **options)
    else:
        # Interpolate colors separately on each side, never across the jump.
        zero = int(np.flatnonzero(v == 0)[0])
        left = np.column_stack((field[:, :zero], jump_ratio * field[:, zero]))
        ax.pcolormesh(v, v[:zero + 1], left.T, **options)
        image = ax.pcolormesh(v, v[zero:], field[:, zero:].T, **options)
    ax.set(
        xlim=(-4.5, 4.5), ylim=(-3, 3),
        xlabel=r"$v_x$", ylabel=r"$v_y$", aspect="equal",
        xticks=[-4, -2, 0, 2, 4], yticks=[-2, 0, 2],
    )
    colorbar = fig.colorbar(
        image, cax=ax.inset_axes([1.04, 0, 0.045, 1]),
        ticks=np.linspace(norm.vmin, norm.vmax, 5),
    )
    colorbar.formatter = ScalarFormatter(useMathText=True)
    colorbar.formatter.set_powerlimits((0, 0))
    colorbar.update_ticks()
    save_pdf(fig, output / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=HERE / "outputs" / "two_stream_jump_eps0.25",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.input_dir / "summary.json").read_text())
    n, L = summary["grid"]["n"], summary["grid"]["L"]
    dv = 2 * L / n
    v = (np.arange(n) - n // 2) * dv
    accuracy = summary["grid"]["baseline_angular_accuracy"]
    baseline = f"spectral_a{accuracy}"
    baseline_label = rf"Spectral $p={accuracy}$"
    with np.load(args.input_dir / "snapshots.npz") as data:
        index = int(np.argmax(data["tau"]))
        fields = {
            label: dv * data[f"{method}_collision"][index].sum(axis=2)
            for label, method in (
                ("reference", "reference"), ("fno", "fno"), ("spectral", baseline),
            )
        }
    with np.load(args.input_dir / "initial_collision.npz") as data:
        initial = {key: dv * data[key].sum(axis=2) for key in ("initial", "reference", "neural")}
    table = np.atleast_1d(np.genfromtxt(
        args.input_dir / "time_series.csv", delimiter=",", names=True,
    ))
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "pdf.fonttype": 42})
    plot_final(output, v, fields, baseline_label)
    plot_errors(output, table, baseline, baseline_label)
    f_norm = Normalize(vmin=0, vmax=float(initial["initial"].max()))
    parameters = summary["initial_condition"]
    jump_ratio = None
    if parameters["kind"] == "two_stream_jump":
        epsilon = parameters["epsilon"]
        jump_ratio = (1 - epsilon) / (1 + epsilon)
    plot_initial_field(
        output, "initial_f_2d.pdf", v, initial["initial"], f_norm, jump_ratio,
    )
    q_min = min(float(initial[key].min()) for key in ("reference", "neural"))
    q_max = max(float(initial[key].max()) for key in ("reference", "neural"))
    q_norm = Normalize(vmin=q_min, vmax=q_max)
    for method, label in (("reference", "ref"), ("neural", "neural")):
        plot_initial_field(
            output, f"initial_Q_{label}_2d.pdf", v, initial[method],
            q_norm,
        )


if __name__ == "__main__":
    main()
