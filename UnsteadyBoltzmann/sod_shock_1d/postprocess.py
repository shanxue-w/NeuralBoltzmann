"""Case-local array output and plotting."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MOMENT_COLUMNS = ("rho", "ux", "uy", "uz", "temperature")


def save_results(
    output_dir: Path,
    *,
    solver: SodShock1D,
    final_state: torch.Tensor,
    run_configuration: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    moments = solver.moments(final_state)
    x = solver.x.detach().cpu().numpy()
    final_f = final_state.detach().cpu().numpy()
    final_moments = moments.detach().cpu().numpy()
    np.save(output_dir / "x.npy", x)
    np.save(output_dir / "final_f.npy", final_f)
    np.save(output_dir / "final_moments.npy", final_moments)

    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = (
        (final_moments[:, 0], r"$\rho$", "Final density", "rho_x.pdf"),
        (final_moments[:, 1], r"$u_x$", "Final velocity", "u_x.pdf"),
        (final_moments[:, 4], r"$T$", "Final temperature", "T_x.pdf"),
    )
    for values, ylabel, title, filename in profiles:
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        axis.plot(x, values, color="tab:blue", linewidth=2.0)
        axis.set_xlabel("x")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{title} at t={run_configuration['T_end']:.8g}")
        axis.set_xlim(solver.x_left, solver.x_right)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / filename)
        plt.close(figure)

    configuration = {
        **run_configuration,
        "equation": "partial_t f + v_x partial_x f = Q(f,f) / Kn",
        "spatial_domain": [solver.x_left, solver.x_right],
        "maxwellian": "rho/(2*pi*T)^(3/2)*exp(-|v-u|^2/(2*T))",
        "spatial_grid": "finite-volume cell centers",
        "boundary_condition": {
            "left_vx_positive": "fixed upstream Maxwellian M_minus",
            "left_vx_negative": "outflow reconstructed from interior",
            "right_vx_negative": "fixed downstream Maxwellian M_plus",
            "right_vx_positive": "outflow reconstructed from interior",
        },
        "time_discretization": "unsplit method of lines",
        "transport": "minmod-MUSCL finite volume",
        "initial_condition": {
            "interface": 0.0,
            "rho_minus": solver.rho_minus,
            "u_minus": solver.u_minus,
            "temperature_minus": solver.temperature_minus,
            "rho_plus": solver.rho_plus,
            "u_plus": solver.u_plus,
            "temperature_plus": solver.temperature_plus,
        },
        "moment_columns": list(MOMENT_COLUMNS),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
