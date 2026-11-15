"""Original cavity plot layout, with streamfunction integration on stretched grids.

Replot saved results without loading either solver:
    python -m UnsteadyBoltzmann.lid_cavacity_2d.postprocess [output_directory ...]
With no arguments, process completed results in this case's output directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UnsteadyBoltzmann.runtime import filename_float


def _interpolate_centerline(coordinates: np.ndarray, field: np.ndarray, axis: int) -> np.ndarray:
    upper = int(np.searchsorted(coordinates, 0.5))
    if upper == 0:
        return np.take(field, 0, axis=axis).copy()
    if upper == len(coordinates):
        return np.take(field, -1, axis=axis).copy()
    lower = upper - 1
    weight = (0.5 - coordinates[lower]) / (coordinates[upper] - coordinates[lower])
    return (1.0 - weight) * np.take(field, lower, axis=axis) + weight * np.take(field, upper, axis=axis)


def _mass_flux_streamfunction(x, y, density, velocity):
    """Use the original two integration paths with actual coordinate spacings."""
    mass_flux_x = density * velocity[..., 0]
    mass_flux_y = density * velocity[..., 1]
    from_bottom = np.empty_like(density)
    from_bottom[:, 0] = y[0] * mass_flux_x[:, 0]
    from_bottom[:, 1:] = from_bottom[:, [0]] + np.cumsum(
        0.5 * np.diff(y)[None, :] * (mass_flux_x[:, :-1] + mass_flux_x[:, 1:]), axis=1
    )
    from_left = np.empty_like(density)
    from_left[0] = -x[0] * mass_flux_y[0]
    from_left[1:] = from_left[[0]] - np.cumsum(
        0.5 * np.diff(x)[:, None] * (mass_flux_y[:-1] + mass_flux_y[1:]), axis=0
    )
    return 0.5 * (from_bottom + from_left)


def save_moment_plots(output_dir, *, x, y, moments, time_value, source_title, source_key=None):
    """Keep the original five PDF names, sizes, contour levels and color maps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    density, velocity, temperature = moments[..., 0], moments[..., 1:4], moments[..., 4]
    streamfunction = _mass_flux_streamfunction(x, y, density, velocity)
    horizontal = [_interpolate_centerline(y, velocity[..., component], 1) for component in range(2)]
    vertical = [_interpolate_centerline(x, velocity[..., component], 0) for component in range(2)]
    profiles = {
        "horizontal_x": x, "horizontal_ux": horizontal[0], "horizontal_uy": horizontal[1],
        "vertical_y": y, "vertical_ux": vertical[0], "vertical_uy": vertical[1],
    }
    if source_key is not None:
        profiles["source"] = np.array(source_key)
    np.savez(output_dir / "centerline_profiles.npz", **profiles)

    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speed = np.hypot(velocity[..., 0], velocity[..., 1])
    stream_levels = None
    finite_stream = streamfunction[np.isfinite(streamfunction)]
    if finite_stream.size and np.ptp(finite_stream) > 1.0e-14 * max(np.max(np.abs(finite_stream)), 1.0):
        stream_levels = np.linspace(finite_stream.min(), finite_stream.max(), 22)[1:-1]

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    for axis, field, label in zip(axes[:2], velocity[..., :2].transpose(2, 0, 1), (r"$u_x$", r"$u_y$")):
        limit = max(float(np.nanmax(np.abs(field))), 1.0e-15)
        contour = axis.contourf(x, y, field.T, levels=np.linspace(-limit, limit, 31), cmap="RdBu_r")
        figure.colorbar(contour, ax=axis, label=label)
        axis.set(title=label, xlabel="x", ylabel="y", aspect="equal")
    contour = axes[2].contourf(
        x, y, speed.T, levels=np.linspace(0.0, max(float(np.nanmax(speed)), 1.0e-15), 31), cmap="viridis"
    )
    if stream_levels is not None:
        axes[2].contour(x, y, streamfunction.T, levels=stream_levels, colors="white", linewidths=0.6)
    figure.colorbar(contour, ax=axes[2], label=r"$|u_{xy}|$")
    axes[2].set(title=r"$|u_{xy}|$ and mass-flux streamlines", xlabel="x", ylabel="y", aspect="equal")
    figure.suptitle(f"{source_title} at t={time_value:g}")
    figure.tight_layout()
    figure.savefig(output_dir / "velocity_contours_streamlines.pdf")
    plt.close(figure)

    for name, field, cmap, label, title, draw_streamlines in (
        ("density_streamlines.pdf", density, "viridis", r"$\rho$", "density and streamlines", True),
        ("temperature_contour.pdf", temperature, "inferno", r"$T$", "temperature", False),
    ):
        figure, axis = plt.subplots(figsize=(6.4, 5.4))
        contour = axis.contourf(x, y, field.T, levels=30, cmap=cmap)
        if draw_streamlines and stream_levels is not None:
            axis.contour(x, y, streamfunction.T, levels=stream_levels, colors="white", linewidths=0.6)
        figure.colorbar(contour, ax=axis, label=label)
        axis.set(xlabel="x", ylabel="y", title=f"{source_title} {title}", aspect="equal")
        figure.tight_layout()
        figure.savefig(output_dir / name)
        plt.close(figure)

    for name, coordinate, fields, xlabel, title in (
        ("velocity_horizontal_centerline.pdf", x, horizontal, "x at y=0.5", "Horizontal"),
        ("velocity_vertical_centerline.pdf", y, vertical, "y at x=0.5", "Vertical"),
    ):
        figure, axis = plt.subplots(figsize=(6.4, 4.6))
        axis.plot(coordinate, fields[0], label=r"$u_x$", linewidth=2.0)
        axis.plot(coordinate, fields[1], label=r"$u_y$", linewidth=2.0)
        axis.set(xlabel=xlabel, ylabel="velocity", title=f"{title} centerline ({source_title})")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / name)
        plt.close(figure)
    return streamfunction


def save_reference_fields_and_plots(output_dir, *, x, y, moments, time_value, source_key, source_title):
    """Original ensemble output interface, using the stretched-grid plots."""
    streamfunction = save_moment_plots(output_dir, x=x, y=y, moments=moments, time_value=time_value,
                                     source_key=source_key, source_title=f"DSMC {source_title}")
    for name, array in dict(x=x, y=y, reference_moments=moments, reference_density=moments[..., 0],
                            reference_velocity=moments[..., 1:4], reference_temperature=moments[..., 4],
                            reference_streamfunction=streamfunction).items():
        np.save(output_dir / f"{name}.npy", array)


def replot(output_dir):
    """Read only saved moments, coordinates and metadata; do not advance time."""
    output_dir = Path(output_dir)
    if output_dir.is_file():
        with np.load(output_dir, allow_pickle=False) as snapshot:
            configuration = json.loads(str(snapshot["metadata_json"]))
            x, y, moments = snapshot["x"], snapshot["y"], snapshot["moments"]
            time_value = float(snapshot["time"])
        output_dir = output_dir.parent / f"t{filename_float(time_value)}"
    else:
        configuration = json.loads((output_dir / "run_config.json").read_text())
        x, y = np.load(output_dir / "x.npy"), np.load(output_dir / "y.npy")
        moments = np.load(output_dir / "final_moments.npy")
        time_value = configuration["T_end"]
    is_dsmc = "dsmc" in configuration.get("case", "").lower()
    streamfunction = save_moment_plots(
        output_dir, x=x, y=y, moments=moments, time_value=time_value,
        source_title=configuration.get("reference_title", "DSMC terminal state" if is_dsmc else "Deterministic terminal state"),
        source_key=configuration.get("reference_source", "terminal_instantaneous_moments") if is_dsmc else None,
    )
    name = "reference_streamfunction" if is_dsmc else "final_streamfunction"
    np.save(output_dir / f"{name}.npy", streamfunction)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", type=Path, nargs="*", help="result directories, snapshot .npz files, or a snapshots directory")
    args = parser.parse_args(argv)
    results = set()
    for directory in args.directories or [CASE_DIR / "output"]:
        if directory.is_file() and directory.suffix == ".npz":
            results.add(directory)
        elif (directory / "final_moments.npy").is_file():
            results.add(directory)
        else:
            results.update(directory.glob("moments_t*.npz"))
            results.update(path.parent for path in directory.glob("*/final_moments.npy"))
    if not results:
        parser.error("no saved final_moments.npy results or moment snapshots found")
    for directory in sorted(results):
        destination = replot(directory)
        print(f"saved five cavity PDFs to {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
