#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DSMC = ROOT / "nx100_kn0p1_tend50_dt0p0109375_L10_Ma2_gam1p66666666667_rhom1_Tm1_Tstd_dsmc_ensemble50_ppc2000_seed1234"
NEURAL = ROOT / "nx100_nv32_kn0p1_tend50_dt0p035_L10_Ma2_gam1p66666666667_rhom1_Tm1_Tstd_gate_euler"
MOMENTS = ((0, r"$\rho$", "sod_rho.pdf"), (1, r"$u_x$", "sod_u.pdf"), (4, r"$T$", "sod_T.pdf"))

x = np.load(DSMC / "x.npy")
x_neural = np.load(NEURAL / "x.npy")
np.testing.assert_allclose(x, x_neural)
dsmc = np.load(DSMC / "final_moments.npy")
standard_error = np.load(DSMC / "moments_standard_error.npy")
neural = np.load(NEURAL / "final_moments.npy")

for column, ylabel, filename in MOMENTS:
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    ax.fill_between(x, dsmc[:, column] - 2 * standard_error[:, column],
                    dsmc[:, column] + 2 * standard_error[:, column],
                    color="0.75", alpha=0.2, linewidth=0, zorder=1)
    ax.plot(x, neural[:, column], color="tab:orange", linewidth=2.2,
            label="Neural", zorder=2)
    ax.plot(x, dsmc[:, column], linestyle="none", marker="x",
            color="black", markersize=3.2, markeredgewidth=0.9,
            label="DSMC", zorder=3)
    ax.set(xlabel=r"$x$", ylabel=ylabel, xlim=(x[0], x[-1]))
    ax.grid(alpha=0.12, linewidth=0.6)
    ax.tick_params(direction="in", top=True, right=True, length=4)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False)
    fig.savefig(ROOT / filename, bbox_inches="tight")
    plt.close(fig)
