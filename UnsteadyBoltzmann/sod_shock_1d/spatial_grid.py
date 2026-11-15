"""Shared physical-space cell edges for deterministic and particle shocks."""
import math
import numpy as np


def grid_edges(nx, *, mesh="sinh", stretch=5.0, refine_factor=1,
               refine_center=0.0, refine_width=10.0):
    if nx < 2 or int(nx) != nx:
        raise ValueError("nx must be an integer >= 2")
    if mesh == "sinh":
        if not math.isfinite(stretch) or not 0 <= stretch <= 10:
            raise ValueError("stretch must lie in [0, 10]")
        xi = np.linspace(-1.0, 1.0, nx + 1)
        return 35 * (xi if stretch == 0 else np.sinh(stretch * xi) / np.sinh(stretch))
    if mesh != "local":
        raise ValueError("mesh must be sinh or local")
    if refine_factor < 1 or int(refine_factor) != refine_factor:
        raise ValueError("refine-factor must be a positive integer")
    if not math.isfinite(refine_center) or not math.isfinite(refine_width) or refine_width <= 0:
        raise ValueError("finite refinement center and positive width required")
    base = np.linspace(-35.0, 35.0, nx + 1)
    edges = []
    for a, b in zip(base[:-1], base[1:]):
        count = refine_factor if abs((a + b) / 2 - refine_center) <= refine_width / 2 else 1
        edges.extend(np.linspace(a, b, count + 1)[:-1])
    return np.asarray(edges + [35.0])


def add_mesh_arguments(parser):
    parser.add_argument("--mesh", choices=("sinh", "local"), default="sinh",
                        help="global center-focused sinh map, or legacy local subdivision")
    parser.add_argument("--stretch", type=float, default=5.0,
                        help="sinh concentration near zero; 0 is uniform, default 5")
    return parser
