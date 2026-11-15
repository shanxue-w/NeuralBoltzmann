#!/usr/bin/env python3
"""Plot the three-stream collision marginal at tau=5."""

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/q_learning_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)


HERE = Path(__file__).resolve().parent
with np.load(HERE / "snapshots.npz", allow_pickle=False) as data:
    index = int(np.flatnonzero(np.isclose(data["tau"], 5.0))[0])
    collisions = {
        "Reference": data["reference_collision"][index],
        "Neural": data["fno_collision"][index],
        r"Spectral $p=9$": data["spectral_a9_collision"][index],
    }

n, L = collisions["Neural"].shape[0], 10.0
dv = 2.0 * L / n
v = (np.arange(n) - n // 2) * dv
marginals = {name: dv * value.sum(axis=2) for name, value in collisions.items()}
limit = max(float(np.abs(value).max()) for value in marginals.values())
levels = np.linspace(-0.8 * limit, 0.8 * limit, 9)
levels = levels[np.abs(levels) > 1.0e-14 * limit]
# Keep three common positive levels, omitting the highest spectral-only loop.
levels = np.concatenate((levels[levels < 0], levels[levels > 0][:3]))

cmap = LinearSegmentedColormap.from_list(
    "collision_ocean_teal",
    ("#06184F", "#075985", "#0891B2", "#9FE3D4", "#19B77A", "#07543A"),
)
fig, ax = plt.subplots(figsize=(5.0, 4.4), constrained_layout=True)
image = ax.contourf(
    v,
    v,
    marginals["Neural"].T,
    levels=np.linspace(-limit, limit, 65),
    cmap=cmap,
)
styles = {
    r"Spectral $p=9$": ("#7C3AED", "solid", 1.8, 2),
    "Neural": ("#F59E0B", "solid", 1.3, 3),
    "Reference": ("#171717", "dashed", 0.9, 4),
}
for name, (color, style, width, order) in styles.items():
    ax.contour(
        v,
        v,
        marginals[name].T,
        levels=levels,
        colors=color,
        linestyles=style,
        linewidths=width,
        zorder=order,
    )

ax.set(xlim=(-6, 6), ylim=(-6, 6), xlabel=r"$v_x$", ylabel=r"$v_y$")
ax.set_aspect("equal")
ax.legend(
    handles=[
        Line2D([], [], color=styles[name][0], linestyle=styles[name][1], label=name)
        for name in ("Reference", "Neural", r"Spectral $p=9$")
    ],
    loc="lower left",
    framealpha=0.78,
)
cax = ax.inset_axes([1.035, 0.01, 0.035, 0.95])
colorbar = fig.colorbar(
    image,
    cax=cax,
    orientation="vertical",
    ticks=(-limit, 0.0, limit),
    format=FuncFormatter(lambda value, _: f"{1.0e3 * value:.1f}"),
)
colorbar.ax.tick_params(labelsize=10)
colorbar.ax.set_title("1e-3", fontsize=10, pad=4)
fig.savefig(HERE / "collision_vx_vy_marginal_tau5.pdf")
plt.close(fig)
