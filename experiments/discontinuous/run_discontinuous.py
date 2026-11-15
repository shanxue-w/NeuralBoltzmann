"""Evolve the two-stream initial state with a vy=0 jump of epsilon=0.25."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "homogeneous"))
import run_homogeneous as h

CASE_NAME = "two_stream_jump_eps0.25"


def initial_condition(grid):
    parameters = {
        "kind": "two_stream_jump",
        "epsilon": 0.25,
        "temperature": 0.6,
        "centers": [[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
        "weights": [0.5, 0.5],
    }
    base = sum(
        w * h.normalized_maxwellian(grid, tuple(c), parameters["temperature"])
        for w, c in zip(parameters["weights"], parameters["centers"])
    )
    value = base * np.where(grid.vy < 0, 0.75, 1.25)
    normalization = 1 / h.integrate(value, grid)
    return normalization * value, {
        **parameters,
        "normalization": normalization,
        "mass_normalization": "discrete Nv64",
        "plane_boundary": "vy=0 belongs to plus side",
    }


def plot_initial(output, grid, reference, neural):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/q_learning_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mask = np.abs(grid.v) <= 5
    x, y = np.meshgrid(grid.v[mask], grid.v[mask], indexing="ij")
    values = [grid.dv * q.sum(axis=2)[np.ix_(mask, mask)] for q in (reference, neural)]
    low, high = min(q.min() for q in values), max(q.max() for q in values)
    limit = max(abs(low), abs(high), np.finfo(float).tiny)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for method, title, q in zip(
        ("reference", "neural"),
        ("Spectral reference ($N_v=64$, $p=23$)", "Neural operator"),
        values,
    ):
        fig = plt.figure(figsize=(7, 5.6), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
        surface = ax.plot_surface(
            x, y, q, cmap="RdBu_r", norm=norm,
            rstride=1, cstride=1, linewidth=0, antialiased=True,
        )
        ax.set(
            xlabel=r"$v_x$", ylabel=r"$v_y$",
            zlabel=r"$\int Q[f_0,f_0]\mathrm{d}v_z$",
            title=title, zlim=(low, high),
        )
        ax.view_init(elev=28, azim=-55)
        fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.12)
        fig.savefig(output / f"initial_collision_{method}_3d.pdf", bbox_inches="tight")
        fig.savefig(output / f"initial_collision_{method}_3d.png", dpi=170)
        plt.close(fig)


def plot_final_collision(output, grid, baseline_accuracy):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import TwoSlopeNorm

    with np.load(output / "snapshots.npz") as data:
        index = int(np.argmax(data["tau"]))
        tau = float(data["tau"][index])
        methods = ("reference", "fno", f"spectral_a{baseline_accuracy}")
        q = {m: data[f"{m}_collision"][index] for m in methods}
    marginals = {m: grid.dv * a.sum(axis=2) for m, a in q.items()}
    errors = {
        m: {
            "full_q_relative_l2": float(
                np.linalg.norm(q[m] - q["reference"]) / np.linalg.norm(q["reference"])
            ),
            "marginal_relative_l2": float(
                np.linalg.norm(marginals[m] - marginals["reference"])
                / np.linalg.norm(marginals["reference"])
            ),
        }
        for m in methods[1:]
    }
    limit = max(float(np.abs(a).max()) for a in marginals.values())
    levels = np.linspace(-0.8 * limit, 0.8 * limit, 9)
    levels = levels[np.abs(levels) > 1e-14 * limit]
    # Use the same three positive contours for all methods.
    levels = np.concatenate((levels[levels < 0], levels[levels > 0][:3]))
    fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")
    im = ax.contourf(
        grid.v, grid.v, marginals["reference"].T,
        levels=np.linspace(-limit, limit, 65), cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    styles = (
        (methods[2], "#7C3AED", "-", 1.5, f"Spectral p={baseline_accuracy}"),
        ("fno", "#F59E0B", "-", 1.5, "Neural"),
        ("reference", "#171717", "--", 1.0, "Reference p=23"),
    )
    handles = []
    for m, color, style, width, label in styles:
        ax.contour(
            grid.v, grid.v, marginals[m].T,
            levels=levels, colors=color, linestyles=style, linewidths=width,
        )
        handles.append(Line2D([], [], color=color, linestyle=style, linewidth=width, label=label))
    ax.set(
        xlim=(-6, 6), ylim=(-6, 6), xlabel=r"$v_x$", ylabel=r"$v_y$",
        title=rf"Collision marginal at $\tau={tau:g}$",
    )
    ax.set_aspect("equal")
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    for extension in ("pdf", "png"):
        fig.savefig(
            output / f"collision_vx_vy_marginal_tau{tau:g}.{extension}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    return {
        "tau": tau,
        "comparison": "each method on its own evolved state; includes trajectory error",
        "errors": errors,
    }


def run_case(args, name, grid):
    output = args.output_dir / name
    output.mkdir(parents=True, exist_ok=True)
    initial, metadata = initial_condition(grid)
    steps = round(args.tau_end / args.dtau)
    tau = np.arange(steps + 1, dtype=np.float64) * args.dtau
    indices = [round(t / args.dtau) for t in args.snapshot_tau]
    baseline = f"spectral_a{args.baseline_angular_accuracy}"
    methods = ("reference", "fno", baseline)

    states, q, diagnostics, cached = h.load_or_compute_reference(
        args, name, initial, grid, steps, indices,
    )
    trajectories = {"reference": states}
    collisions = {"reference": q}
    spectral_diagnostics = {"reference": diagnostics}

    for method in (baseline, "fno"):
        print(f"[{name}] {method}", flush=True)
        rhs = (
            h.NeuralCollision(args.checkpoint, grid, args.device)
            if method == "fno"
            else h.SpectralCollision(
                grid, args.R, args.baseline_angular_accuracy, args.angular_chunk,
            )
        )
        try:
            for _ in range(args.warmup):
                rhs(initial)
            states, q, minimum = h.integrate_ssprk2(
                method, rhs, initial, args.dtau, steps, set(indices),
            )
        finally:
            rhs.close()
        trajectories[method] = states
        collisions[method] = q
        if method == baseline:
            spectral_diagnostics[method] = {
                "positivity_correction": False,
                "total_stages": 2 * steps,
                "minimum_stage_value": minimum,
                "minimum_saved_state": float(states.min()),
            }

    rows = []
    for index, time_value in enumerate(tau):
        row = {"tau": time_value}
        for method in methods[1:]:
            errors = h.error_metrics(
                trajectories[method][index], trajectories["reference"][index], grid,
            )
            row.update({f"{method}_{key}": value for key, value in errors.items()})
        rows.append(row)
    h.write_csv(output / "time_series.csv", rows)

    selector = np.asarray(indices, dtype=np.int64)
    payload = {"tau": tau[selector], "initial": initial}
    for method in methods:
        payload[f"{method}_states"] = trajectories[method][selector]
        payload[f"{method}_collision"] = np.stack(
            [collisions[method][index] for index in indices]
        )
    np.savez_compressed(output / "snapshots.npz", **payload)
    h.make_plots(
        output, grid, tau, trajectories, rows, tau[selector],
        {method: payload[f"{method}_collision"] for method in methods},
        args.map_vmax, baseline, args.baseline_angular_accuracy,
    )
    metrics = ("absolute_l1", "absolute_l2", "relative_l1", "relative_l2")
    summary = {
        "case": name,
        "initial_condition": metadata,
        "grid": {
            "n": grid.n, "L": grid.L, "R": args.R,
            "radial_accuracy": grid.n,
            "reference_angular_accuracy": h.REFERENCE_ACCURACY,
            "baseline_angular_accuracy": args.baseline_angular_accuracy,
            "kernel_constant": "1/(4*pi)", "loss_mode": "sinc",
        },
        "time_integration": {
            "integrator": "SSPRK2 for all three trajectories",
            "spectral_state_correction": "none",
            "dtau": args.dtau, "tau_end": args.tau_end,
            "snapshot_tau": args.snapshot_tau,
        },
        "reference": f"independent spectral p={h.REFERENCE_ACCURACY} trajectory",
        "reference_cache": {
            "path": str(args.reference_cache_dir / name), "loaded": cached,
        },
        "checkpoint": str(args.checkpoint),
        "spectral_positivity_repairs": spectral_diagnostics,
        "final_errors": {
            method: {key: rows[-1][f"{method}_{key}"] for key in metrics}
            for method in methods[1:]
        },
        "files": {
            "errors": "time_series.csv",
            "difference_maps": "final_vx_vy_difference_maps.png",
            "collision_marginals": "collision_vx_vy_marginals_over_time.pdf",
            "snapshots": "snapshots.npz",
        },
    }
    final_report = ", ".join(
        f"{method}={summary['final_errors'][method]['relative_l2']:.6e}"
        for method in methods[1:]
    )
    print(f"[{name}] final relative L2: {final_report}", flush=True)
    return summary


def main():
    parser = h.build_parser()
    parser.description = __doc__
    # Keep all numerical defaults from homogeneous; isolate data and caches.
    for action in parser._actions:
        if action.dest == "case":
            action.choices = (CASE_NAME,)
            action.default = [CASE_NAME]
    parser.set_defaults(output_dir=HERE / "outputs", reference_cache_dir=HERE / "reference_cache")
    parser.add_argument("--screen-only", action="store_true", help="Only evaluate the fixed initial state.")
    args = parser.parse_args()
    if args.n != 64:
        parser.error("This experiment requires Nv=64 for reference and neural grids.")
    if (
        args.dtau <= 0
        or args.tau_end < 0
        or not np.isclose(args.tau_end / args.dtau, round(args.tau_end / args.dtau))
    ):
        parser.error("tau-end must be a nonnegative integer multiple of positive dtau")
    if any(
        t < 0 or t > args.tau_end or not np.isclose(t / args.dtau, round(t / args.dtau))
        for t in args.snapshot_tau
    ):
        parser.error("snapshot times must be time steps within [0, tau-end]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = h.make_grid(args.n, h.checkpoint_velocity_half_width(args.checkpoint))
    initial, _ = initial_condition(grid)
    reference = h.SpectralCollision(grid, args.R, 23, args.angular_chunk)
    try:
        reference_q = reference(initial)
    finally:
        reference.close()
    neural = h.NeuralCollision(args.checkpoint, grid, args.device)
    try:
        neural_q = neural(initial)
    finally:
        neural.close()
    errors = {"case": CASE_NAME, **h.error_metrics(neural_q, reference_q, grid)}
    print(f"[{CASE_NAME}] initial Q relative L2={errors['relative_l2']:.6g}", flush=True)
    output = args.output_dir / CASE_NAME
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "initial_collision.npz",
        v=grid.v, initial=initial, reference=reference_q, neural=neural_q,
    )
    plot_initial(output, grid, reference_q, neural_q)
    if not args.screen_only:
        summary = run_case(args, CASE_NAME, grid)
        summary["initial_q_errors"] = errors
        summary["final_collision"] = plot_final_collision(output, grid, args.baseline_angular_accuracy)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")



if __name__ == "__main__":
    main()
