#!/usr/bin/env python3
"""Plot the spectral a=9 relative L2 error history."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
data = np.genfromtxt(
    HERE / "angular_accuracy_time_series.csv", delimiter=",", names=True
)

fig, ax = plt.subplots(figsize=(5.0, 4.4), constrained_layout=True)
ax.plot(
    data["tau"],
    data["spectral_a9_relative_l2"],
    color="#7C3AED",
    linewidth=2.0,
    marker="o",
    markevery=5,
    markersize=4,
    label=r"Spectral $p=9$",
)
ax.plot(
    data["tau"],
    data["neural_operator_relative_l2"],
    color="#007C91",
    linewidth=2.0,
    marker="s",
    markevery=5,
    markersize=4,
    label="Neural",
)
ax.set(xlabel=r"$t$", ylabel=r"relative $L_v^2$ error")
ax.grid(alpha=0.22)
ax.legend(frameon=False)
fig.savefig(HERE / "spectral_p9_relative_l2.pdf")
plt.close(fig)
