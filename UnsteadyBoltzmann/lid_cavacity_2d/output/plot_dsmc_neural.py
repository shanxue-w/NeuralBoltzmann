#!/usr/bin/env python3
"""Compare exact-time Neural/DSMC moments in separate or combined figures.

Run without arguments for t=0.5, 1, 5. The shortest completed run containing
the requested time is preferred. Snapshots are selected by exact filename
and checked against their stored time; final arrays are used only at their
actual terminal time. Add --layout paper for new colors and separate flow plots.
Use --layout combined for time rows and four moment columns, plus a flow figure.
No checkpoint, model, or particle data are loaded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


DSMC_DIRS = tuple(ROOT / (
    f"nx32_ny32_kn1_tend{end}_dt0p003125_stretch0_Ulid1_Tw1_Tstd_"
    "dsmc_ensemble200_ppc2000_seed1234"
) for end in (1, 5))
NEURAL_DIRS = tuple(ROOT / (
    f"nx32_ny32_nv64_kn1_tend{end}_dt0p005_stretch0_L6_Ulid1_Tw1_csl3_tct_gate_euler"
) for end in (1, 5))


@dataclass
class MomentState:
    directory: Path
    source: Path
    time: float
    x: np.ndarray
    y: np.ndarray
    moments: np.ndarray
    configuration: dict


def time_slug(value):
    return format(float(value), ".12g").replace("-", "m").replace("+", "").replace(".", "p")


def same_time(left, right):
    return math.isclose(left, right, rel_tol=0, abs_tol=8 * math.ulp(max(1.0, abs(right))))


def terminal_time(configuration):
    return float(configuration.get("T_end", configuration.get("t_end", math.inf)))


def load_state(directory, time_value):
    directory = Path(directory)
    configuration = json.loads((directory / "run_config.json").read_text())
    source = directory / "snapshots" / f"moments_t{time_slug(time_value)}.npz"
    if source.is_file():
        with np.load(source, allow_pickle=False) as saved:
            actual_time = float(saved["time"])
            x, y, moments = saved["x"], saved["y"], saved["moments"]
            if "metadata_json" in saved:
                configuration = {**configuration, **json.loads(str(saved["metadata_json"]))}
    elif same_time(terminal_time(configuration), time_value):
        source = directory / "final_moments.npy"
        actual_time = terminal_time(configuration)
        x, y = np.load(directory / "x.npy"), np.load(directory / "y.npy")
        moments = np.load(source)
    else:
        raise FileNotFoundError(f"no exact t={time_value:g} snapshot in {directory}")
    if not same_time(actual_time, time_value):
        raise ValueError(f"stored time {actual_time!r} does not match t={time_value:g}: {source}")
    if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
        raise ValueError(f"invalid spatial coordinates: {source}")
    if moments.shape != (x.size, y.size, 5) or not np.isfinite(moments).all():
        raise ValueError(f"invalid/nonfinite five-moment array: {source}")
    for grid in (x, y):
        if not np.isfinite(grid).all() or not np.all(np.diff(grid) > 0) or grid[0] <= 0 or grid[-1] >= 1:
            raise ValueError(f"expected increasing cell centers inside the unit square: {source}")
    return MomentState(directory, source, actual_time, x, y, moments, configuration)


def select_state(directories, time_value):
    candidates = []
    for directory in directories:
        directory = Path(directory)
        config_path = directory / "run_config.json"
        if not config_path.is_file():
            continue
        end = terminal_time(json.loads(config_path.read_text()))
        if end >= time_value or same_time(end, time_value):
            candidates.append((end, directory))
    for _, directory in sorted(candidates):
        try:
            return load_state(directory, time_value)
        except FileNotFoundError:
            continue
    searched = "\n  ".join(str(path) for path in directories)
    raise FileNotFoundError(f"no exact t={time_value:g} data in:\n  {searched}")


def validate_pair(neural, dsmc):
    np.testing.assert_allclose(neural.x, dsmc.x, rtol=0, atol=1e-12, err_msg="x grids differ")
    np.testing.assert_allclose(neural.y, dsmc.y, rtol=0, atol=1e-12, err_msg="y grids differ")
    if not same_time(neural.time, dsmc.time):
        raise ValueError("Neural and DSMC times differ")
    for aliases in (("Kn", "kn"), ("lid_speed",), ("wall_temperature",), ("stretch",)):
        values = [next((state.configuration[key] for key in aliases if key in state.configuration), None)
                  for state in (neural, dsmc)]
        if any(value is None for value in values) or not math.isclose(*map(float, values), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"missing or mismatched {aliases[0]}: Neural={values[0]}, DSMC={values[1]}")


def centerline(coordinates, field, axis):
    """Interpolate current-time cell-center data to the exact line x/y=0.5."""
    upper = int(np.searchsorted(coordinates, 0.5))
    if upper == 0 or upper == len(coordinates):
        return np.take(field, 0 if upper == 0 else -1, axis=axis)
    lower = upper - 1
    weight = (0.5 - coordinates[lower]) / (coordinates[upper] - coordinates[lower])
    return ((1 - weight) * np.take(field, lower, axis=axis)
            + weight * np.take(field, upper, axis=axis))


def save_figure(figure, output, name, formats, dpi):
    for extension in formats:
        figure.savefig(output / f"{name}.{extension}", bbox_inches="tight", dpi=dpi)
    plt.close(figure)


def draw_contours(axis, x, y, field, levels, **style):
    # A constant field has no isolines; do not manufacture an artificial one.
    levels = levels[(levels > field.min()) & (levels < field.max())]
    if levels.size:
        axis.contour(x, y, field.T, levels=levels, **style)


def contour_settings(background, signed, line_count):
    """Original per-time Neural scale and common Neural/DSMC contour levels."""
    if signed:
        limit = max(float(np.abs(background).max()), 1e-15)
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = float(background.min()), float(background.max())
        if vmax == vmin:
            vmin, vmax = vmin - 1e-12, vmax + 1e-12
    return vmin, vmax, np.linspace(vmin, vmax, line_count + 2)[1:-1]


def plot_comparison(neural, dsmc, output, formats=("pdf", "png"), dpi=200,
                    show_time=False, show_legend=True, paper_style=False):
    validate_pair(neural, dsmc)
    output.mkdir(parents=True, exist_ok=True)
    x, y = neural.x, neural.y
    n, d = neural.moments, dsmc.moments
    # Match the reference's display-only edge padding, clipped to [0,1]^2.
    plot_x = np.r_[2 * x[0] - x[1], x, 2 * x[-1] - x[-2]]
    plot_y = np.r_[2 * y[0] - y[1], y, 2 * y[-1] - y[-2]]
    fields = (
        (n[..., 0], d[..., 0], r"$\rho$", "rho_contours", False, 10),
        (n[..., 1], d[..., 1], r"$u_x$", "ux_contours", True, 10),
        (n[..., 2], d[..., 2], r"$u_y$", "uy_contours", True, 10),
        (n[..., 4], d[..., 4], r"$T$", "T_contours", False, 6),
        (np.hypot(n[..., 1], n[..., 2]), np.hypot(d[..., 1], d[..., 2]),
         r"$|\mathbf{u}_{xy}|$", "speed_contours", False, 6),
    )
    for background, comparison, label, name, signed, line_count in fields:
        # As in the reference, Neural defines the color scale and the common
        # contour levels. DSMC is never independently renormalized.
        vmin, vmax, levels = contour_settings(background, signed, line_count)
        cmap = ({"rho_contours": "Blues", "ux_contours": "RdBu_r", "uy_contours": "RdBu_r",
                 "T_contours": "YlOrRd", "speed_contours": "magma"}[name]
                if paper_style else "viridis")
        figure, axis = plt.subplots(figsize=(3.6, 3.15), constrained_layout=True)
        image = axis.pcolormesh(plot_x, plot_y, np.pad(background, 1, mode="edge").T,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                shading="gouraud", rasterized=True)
        draw_contours(axis, x, y, background, levels, colors="#F59E0B", linewidths=1.0, zorder=2)
        draw_contours(axis, x, y, comparison, levels, colors="#171717", linestyles="dashed",
                      linewidths=0.8, zorder=3)
        axis.set(xlabel=r"$x$", ylabel=r"$y$", xlim=(0, 1), ylim=(0, 1), aspect="equal")
        axis.set_xticks([0, 0.5, 1])
        axis.set_yticks([0, 0.5, 1])
        axis.tick_params(top=False, right=False, length=0)
        if show_legend:
            axis.legend(handles=(
                Line2D([], [], color="#F59E0B", linewidth=1.0, label="Neural"),
                Line2D([], [], color="#171717", linestyle="dashed", linewidth=0.8, label="DSMC"),
            ), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005),
                ncol=2, borderaxespad=0, handlelength=2, handletextpad=0.4, columnspacing=1)
        if show_time:
            axis.text(0.03, 0.97, rf"$t={neural.time:g}$", transform=axis.transAxes,
                      va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
        colorbar = figure.colorbar(image, ax=axis, shrink=0.88, fraction=0.05, pad=0.04)
        colorbar.locator = MaxNLocator(nbins=4)
        colorbar.update_ticks()
        colorbar.ax.tick_params(length=0)
        save_figure(figure, output, name, formats, dpi)

    for coordinate, nvalues, dvalues, xlabel, ylabel, name in (
        (y, centerline(x, n[..., 1], 0), centerline(x, d[..., 1], 0),
         r"$y$", r"$u_x$", "ux_vertical_centerline"),
        (x, centerline(y, n[..., 2], 1), centerline(y, d[..., 2], 1),
         r"$x$", r"$u_y$", "uy_horizontal_centerline"),
    ):
        figure, axis = plt.subplots(figsize=(3.6, 2.65), constrained_layout=True)
        axis.plot(coordinate, nvalues, color="#0072B2", linewidth=1.6, label="Neural")
        axis.plot(coordinate, dvalues, linestyle="none", marker="x", color="black",
                  markersize=3.5, markeredgewidth=0.8, label="DSMC")
        axis.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1))
        axis.set_xticks([0, 0.5, 1])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.grid(alpha=0.12, linewidth=0.6)
        axis.tick_params(top=False, right=False, length=0)
        if show_legend:
            axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005),
                        ncol=2, borderaxespad=0, handlelength=2, handletextpad=0.4, columnspacing=1)
        if show_time:
            axis.text(0.03, 0.97, rf"$t={neural.time:g}$", transform=axis.transAxes, va="top", fontsize=9)
        save_figure(figure, output, name, formats, dpi)


def regular_velocity_grid(state):
    """Streamplot needs a uniform grid; interpolate spatially if necessary."""
    x = np.linspace(state.x[0], state.x[-1], state.x.size)
    y = np.linspace(state.y[0], state.y[-1], state.y.size)
    components = []
    for index in (1, 2):
        field = state.moments[..., index]
        along_x = np.stack([np.interp(x, state.x, field[:, j]) for j in range(state.y.size)], axis=1)
        components.append(np.stack([np.interp(y, state.y, line) for line in along_x]).T)
    return x, y, *components


def plot_streamlines(neural, dsmc, output, formats=("pdf", "png"), dpi=200, show_time=False):
    """Export each method as one panel, with a common speed scale at this time."""
    speed_limit = max(float(np.hypot(state.moments[..., 1], state.moments[..., 2]).max())
                      for state in (neural, dsmc))
    speed_limit = max(speed_limit, 1e-15)
    for method, state in (("neural", neural), ("dsmc", dsmc)):
        speed = np.hypot(state.moments[..., 1], state.moments[..., 2])
        plot_x = np.r_[2 * state.x[0] - state.x[1], state.x, 2 * state.x[-1] - state.x[-2]]
        plot_y = np.r_[2 * state.y[0] - state.y[1], state.y, 2 * state.y[-1] - state.y[-2]]
        figure, axis = plt.subplots(figsize=(3.6, 3.15), constrained_layout=True)
        image = axis.pcolormesh(plot_x, plot_y, np.pad(speed, 1, mode="edge").T,
                                cmap="magma", vmin=0, vmax=speed_limit,
                                shading="gouraud", rasterized=True)
        x, y, ux, uy = regular_velocity_grid(state)
        axis.streamplot(x, y, ux, uy, color="#fff8eb", density=0.72,
                        linewidth=0.9 * (np.hypot(ux, uy) / speed_limit) ** 0.35,
                        arrowsize=0.7, maxlength=4, zorder=2)
        axis.set(xlabel=r"$x$", ylabel=r"$y$", xlim=(0, 1), ylim=(0, 1), aspect="equal")
        axis.set_xticks([0, 0.5, 1])
        axis.set_yticks([0, 0.5, 1])
        axis.tick_params(top=False, right=False, length=0)
        if show_time:
            axis.text(0.03, 0.97, rf"$t={state.time:g}$", transform=axis.transAxes,
                      va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
        colorbar = figure.colorbar(image, ax=axis, shrink=0.88, fraction=0.05, pad=0.04)
        colorbar.locator = MaxNLocator(nbins=4)
        colorbar.update_ticks()
        colorbar.ax.tick_params(length=0)
        save_figure(figure, output, f"speed_streamlines_{method}", formats, dpi)


def combined_axes(times, row_labels, show_legend=False, shared_colorbar=False,
                  rotate_labels=False, transpose=False):
    """Square panels, compact headers and room for numeric-only colorbars."""
    time_labels = [rf"$t={time_value:g}$" for time_value in times]
    row_text, column_text = (time_labels, row_labels) if transpose else (row_labels, time_labels)
    rows, columns = len(row_text), len(column_text)
    panel, gap, row_gap = 2.12, 0.32, 0.23
    left, bottom = (1.02, 0.92) if transpose else (0.82, 0.42)
    top = 0.65 if show_legend else 0.35
    width = left + columns * panel + (columns - 1) * gap + (0.18 if transpose else 0.72)
    height = bottom + rows * panel + (rows - 1) * row_gap + top
    figure = plt.figure(figsize=(width, height))
    axes = np.empty((rows, columns), dtype=object)
    for row, label in enumerate(row_text):
        y0 = bottom + (rows - row - 1) * (panel + row_gap)
        figure.text((0.24 if transpose else 0.13) / width, (y0 + panel / 2) / height, label,
                    rotation=90 if rotate_labels else 0, ha="center", va="center",
                    fontsize=10 if transpose else 11)
        for column, label in enumerate(column_text):
            x0 = left + column * (panel + gap)
            axis = figure.add_axes([x0 / width, y0 / height, panel / width, panel / height])
            axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
            axis.set_xticks([0, 0.5, 1])
            axis.set_yticks([0, 0.5, 1])
            axis.tick_params(top=False, right=False, length=0)
            if row == 0:
                axis.set_title(label, fontsize=11 if transpose else 10, pad=8)
            if row == rows - 1:
                axis.set_xlabel(r"$x$", labelpad=2)
            else:
                axis.tick_params(labelbottom=False)
            if column == 0:
                axis.set_ylabel(r"$y$", labelpad=2)
            else:
                axis.tick_params(labelleft=False)
            axes[row, column] = axis
    caxes = []
    total_height = rows * panel + (rows - 1) * row_gap
    if transpose:
        for column in range(columns):
            x0 = left + column * (panel + gap) + panel * 0.06
            caxes.append(figure.add_axes([x0 / width, 0.26 / height,
                                          panel * 0.88 / width, 0.085 / height]))
    else:
        for row in range(1 if shared_colorbar else rows):
            bar_height = total_height * 0.76 if shared_colorbar else panel * 0.88
            y0 = (bottom + (total_height - bar_height) / 2 if shared_colorbar else
                  bottom + (rows - row - 1) * (panel + row_gap) + (panel - bar_height) / 2)
            caxes.append(figure.add_axes([(width - 0.49) / width, y0 / height,
                                          0.085 / width, bar_height / height]))
    if show_legend:
        figure.legend(handles=(
            Line2D([], [], color="#F59E0B", linewidth=1.0, label="Neural"),
            Line2D([], [], color="#171717", linestyle="dashed", linewidth=0.8, label="DSMC"),
        ), frameon=False, loc="upper center", bbox_to_anchor=(0.49, 0.998),
            ncol=2, handlelength=2, handletextpad=0.4, columnspacing=1)
    return figure, axes, caxes


def combined_background(axis, state, field, cmap, vmin, vmax):
    plot_x = np.r_[2 * state.x[0] - state.x[1], state.x, 2 * state.x[-1] - state.x[-2]]
    plot_y = np.r_[2 * state.y[0] - state.y[1], state.y, 2 * state.y[-1] - state.y[-2]]
    return axis.pcolormesh(plot_x, plot_y, np.pad(field, 1, mode="edge").T,
                           cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud", rasterized=True)


def combined_colorbar(figure, image, cax, orientation="vertical"):
    colorbar = figure.colorbar(image, cax=cax, orientation=orientation)
    colorbar.locator = MaxNLocator(nbins=4)
    colorbar.update_ticks()
    colorbar.ax.tick_params(length=0)


def plot_combined(pairs, output, formats=("pdf", "png"), dpi=200, show_legend=True, cmap="turbo"):
    """Share background scales while retaining every original contour value."""
    output.mkdir(parents=True, exist_ok=True)
    times = [neural.time for neural, _ in pairs]
    fields = ((0, r"$\rho$", False, 10),
              (1, r"$u_x$", True, 10),
              (2, r"$u_y$", True, 10),
              (4, r"$T$", False, 6))
    figure, axes, caxes = combined_axes(times, [field[1] for field in fields], show_legend,
                                       transpose=True)
    for column, (index, _, signed, line_count) in enumerate(fields):
        # Only the background normalization is shared across times. The
        # original per-time contour locations, counts and styles are retained.
        settings = [contour_settings(neural.moments[..., index], signed, line_count)
                    for neural, _ in pairs]
        vmin, vmax = min(setting[0] for setting in settings), max(setting[1] for setting in settings)
        for row, ((neural, dsmc), (_, _, levels)) in enumerate(zip(pairs, settings)):
            axis = axes[row, column]
            image = combined_background(axis, neural, neural.moments[..., index], cmap, vmin, vmax)
            draw_contours(axis, neural.x, neural.y, neural.moments[..., index], levels,
                          colors="#F59E0B", linewidths=1.0, zorder=2)
            draw_contours(axis, dsmc.x, dsmc.y, dsmc.moments[..., index], levels,
                          colors="#171717", linestyles="dashed", linewidths=0.8, zorder=3)
        combined_colorbar(figure, image, caxes[column], orientation="horizontal")
    save_figure(figure, output, f"moments_{len(times)}x4", formats, dpi)

    speed_limit = max(max(float(np.hypot(state.moments[..., 1], state.moments[..., 2]).max())
                          for pair in pairs for state in pair), 1e-15)
    figure, axes, caxes = combined_axes(times, ("Neural", "DSMC"), shared_colorbar=True,
                                       rotate_labels=True)
    for column, pair in enumerate(pairs):
        for row, state in enumerate(pair):
            axis = axes[row, column]
            speed = np.hypot(state.moments[..., 1], state.moments[..., 2])
            image = combined_background(axis, state, speed, cmap, 0, speed_limit)
            x, y, ux, uy = regular_velocity_grid(state)
            axis.streamplot(x, y, ux, uy, color="#fff8eb", density=0.72,
                            linewidth=0.9 * (np.hypot(ux, uy) / speed_limit) ** 0.35,
                            arrowsize=0.7, maxlength=4, zorder=2)
    combined_colorbar(figure, image, caxes[0])
    save_figure(figure, output, f"streamlines_2x{len(times)}", formats, dpi)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--times", type=float, nargs="+", default=[0.5, 1.0, 5.0])
    parser.add_argument("--neural-dirs", type=Path, nargs="+", default=list(NEURAL_DIRS))
    parser.add_argument("--dsmc-dirs", type=Path, nargs="+", default=list(DSMC_DIRS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "comparison_dsmc_neural")
    parser.add_argument("--formats", choices=("pdf", "png"), nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show-time", action="store_true", help="add a small time annotation; omitted for paper figures")
    parser.add_argument("--no-legend", action="store_true", help="omit legends when line styles are defined in the paper caption")
    parser.add_argument("--layout", choices=("single", "paper", "combined"), default="single",
                        help="paper: separate panels; combined: four moment columns and two flow rows")
    parser.add_argument("--cmap", default="turbo", help="shared palette for combined figures (default: turbo)")
    args = parser.parse_args(argv)
    if args.dpi <= 0 or any(not math.isfinite(t) or t < 0 for t in args.times):
        parser.error("dpi must be positive and requested times must be finite and nonnegative")
    if args.cmap not in plt.colormaps():
        parser.error(f"unknown Matplotlib colormap: {args.cmap}")
    plt.rcParams.update({"font.family": "sans-serif", "mathtext.fontset": "dejavusans",
                         "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    # Load and validate every pair before creating any figures.
    pairs = [(select_state(args.neural_dirs, t), select_state(args.dsmc_dirs, t)) for t in args.times]
    for neural, dsmc in pairs:
        validate_pair(neural, dsmc)
    if args.layout == "combined":
        output = args.output_dir / "combined"
        for neural, dsmc in pairs:
            print(f"t={neural.time:g}\n  Neural: {neural.source}\n  DSMC:   {dsmc.source}", flush=True)
        plot_combined(pairs, output, formats=args.formats, dpi=args.dpi,
                      show_legend=not args.no_legend, cmap=args.cmap)
        print(f"saved two combined figures ({', '.join(args.formats)}) to {output}", flush=True)
        return 0
    paper_style = args.layout == "paper"
    output_root = args.output_dir / "paper" if paper_style else args.output_dir
    for neural, dsmc in pairs:
        output = output_root / f"t{time_slug(neural.time)}"
        print(f"t={neural.time:g}\n  Neural: {neural.source}\n  DSMC:   {dsmc.source}", flush=True)
        plot_comparison(neural, dsmc, output, formats=args.formats, dpi=args.dpi,
                        show_time=args.show_time, show_legend=not args.no_legend, paper_style=paper_style)
        if paper_style:
            plot_streamlines(neural, dsmc, output, formats=args.formats, dpi=args.dpi,
                             show_time=args.show_time)
        count = 9 if paper_style else 7
        print(f"  saved {count} separate figures ({', '.join(args.formats)}) to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
