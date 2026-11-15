#!/usr/bin/env python3
"""Plot z-integrated collision outputs for the two-stream case."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "outputs" / "a40_full"


def load(path):
    with np.load(OUT / path, allow_pickle=False) as data:
        v = data["v"].copy()
        q = data["q"].copy()
    return v, q


fields = [
    ("Reference", *load("reference_N64/two_stream/reference.npz")),
    ("Neural", *load("N64/two_stream/float32/neural.npz")),
    *[
        (
            rf"$p={a}$",
            *load(f"N64/two_stream/float64/spectral_a{a}.npz"),
        )
        for a in (7, 9, 11)
    ],
]

integrals = [
    (title, v, (field.sum(axis=2) * (v[1] - v[0])).T)
    for title, v, field in fields
]
limit = max(float(np.max(np.abs(integral))) for _, _, integral in integrals)

fig, axes = plt.subplots(1, len(integrals), figsize=(17, 3.4), constrained_layout=True)
for i, (title, v, integral) in enumerate(integrals):
    image = axes[i].imshow(
        integral,
        origin="lower",
        extent=(v[0], v[-1], v[0], v[-1]),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axes[i].set_title(title, fontsize=16)
    axes[i].set_xlabel(r"$v_x$", fontsize=15)
    axes[i].set_ylabel(r"$v_y$" if i == 0 else "", fontsize=15)
    axes[i].tick_params(labelsize=12)
colorbar = fig.colorbar(
    image, ax=axes, format="%.1e", fraction=0.025, pad=0.02, shrink=0.82, aspect=25
)
colorbar.ax.tick_params(labelsize=12)

pdf = OUT / "two_stream_z_integral.pdf"
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)
print(pdf)
