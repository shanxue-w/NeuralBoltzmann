"""Plot five macroscopic moments for the configured Couette Kn sweep."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

# Keep Matplotlib's runtime cache out of a potentially read-only home directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "steady_boltzmann_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from SteadyBoltzmann.io_1d import read_json, write_json


KN_VALUES = (0.1, 1.0, 10.0)
MOMENT_SPECS = (
    ("rho", "rho.pdf", r"Density $\rho$", r"$\rho$"),
    ("ux", "u1.pdf", r"Normal velocity $u_1$", r"$u_1$"),
    ("uy", "u2.pdf", r"Tangential velocity $u_2$", r"$u_2$"),
    ("uz", "u3.pdf", r"Transverse velocity $u_3$", r"$u_3$"),
    ("temperature", "theta.pdf", r"Temperature $\theta$", r"$\theta$"),
)


def kn_directory_name(kn: float) -> str:
    return f"kn_{float(kn):g}"


def load_couette_sweep(sweep_dir: str | Path, *,
                       require_converged: bool = True) -> dict[float, dict[str, np.ndarray]]:
    """Load and validate the configured solver outputs used by every PDF."""

    sweep_dir = Path(sweep_dir).resolve()
    curves: dict[float, dict[str, np.ndarray]] = {}
    required_fields = {"x", *(spec[0] for spec in MOMENT_SPECS)}
    for kn in KN_VALUES:
        case_dir = sweep_dir / kn_directory_name(kn)
        summary_path = case_dir / "summary.json"
        solution_path = case_dir / "solution.npz"
        summary = read_json(summary_path)
        if require_converged and not bool(summary.get("converged")):
            raise RuntimeError(f"Kn={kn} is not converged; refusing to plot it")
        with np.load(solution_path) as data:
            values = {field: np.asarray(data[field], dtype=np.float64).copy() for field in required_fields}
        curves[kn] = values
    return curves


def plot_couette_moments(curves: dict[float, dict[str, np.ndarray]], output_dir: str | Path) -> list[Path]:
    """Write exactly five one-page PDFs containing every configured Kn curve."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    color_map = plt.get_cmap("tab10")
    colors = tuple(color_map(index % color_map.N) for index in range(len(KN_VALUES)))
    linestyle_cycle = ("-", "--", "-.", ":")
    linestyles = tuple(linestyle_cycle[index % len(linestyle_cycle)] for index in range(len(KN_VALUES)))
    labels = tuple(fr"$Kn={kn:g}$" for kn in KN_VALUES)
    paths: list[Path] = []
    for field, filename, title, ylabel in MOMENT_SPECS:
        figure, axis = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
        for kn, color, linestyle, label in zip(KN_VALUES, colors, linestyles, labels, strict=True):
            axis.plot(curves[kn]["x"], curves[kn][field], color=color,
                      linestyle=linestyle, linewidth=2.0, label=label)
        axis.set_title(title, fontsize=12)
        axis.set_xlabel(r"Position $x$")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.5, 0.5)
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.legend(frameon=False, ncol=min(3, len(KN_VALUES)), loc="best")
        axis.tick_params(direction="in", top=True, right=True)
        path = output_dir / filename
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def plot_from_sweep(sweep_dir: str | Path, *, allow_unconverged: bool = False) -> list[Path]:
    sweep_dir = Path(sweep_dir).resolve()
    curves = load_couette_sweep(sweep_dir, require_converged=not allow_unconverged)
    paths = plot_couette_moments(curves, sweep_dir / "figures")
    manifest = {
        "format": "couette-kn-sweep-five-moments-v2",
        "kn_values": list(KN_VALUES),
        "moments": [spec[0] for spec in MOMENT_SPECS],
        "pdfs": [path.name for path in paths],
        "source_directories": [kn_directory_name(kn) for kn in KN_VALUES],
        "require_converged": not allow_unconverged,
    }
    write_json(sweep_dir / "figures" / "plot_manifest.json", manifest)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the five moments of an existing Couette Kn sweep.")
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--allow-unconverged", action="store_true",
                        help="diagnostic use only; steady plots reject unconverged runs by default")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = plot_from_sweep(args.sweep_dir, allow_unconverged=args.allow_unconverged)
    for path in paths:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
