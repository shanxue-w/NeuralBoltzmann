"""Thirteen velocity moments, without distributions or particle histories.

Raw columns are rho, rho*u (3), symmetric integral(v_i*v_j*f) (6),
and integral(|v|^2*v_i*f)/2 (3). Averaging these linear moments before
centering gives consistent ensemble pressure and heat flux.
"""

import json
import os
from pathlib import Path

import numpy as np

from UnsteadyBoltzmann.runtime import filename_float, json_ready


GRAD13_COLUMNS = ("rho", "ux", "uy", "uz", "temperature", "stress_xx", "stress_yy",
                  "stress_xy", "stress_xz", "stress_yz", "heat_flux_x", "heat_flux_y", "heat_flux_z")
SECOND_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def fields_from_raw_moments(raw):
    rho = raw[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = raw[..., 1:4] / rho[..., None]
        second = np.empty((*rho.shape, 3, 3))
        for index, (i, j) in enumerate(SECOND_COMPONENTS, 4):
            second[..., i, j] = second[..., j, i] = raw[..., index]
        pressure = second - rho[..., None, None] * velocity[..., :, None] * velocity[..., None, :]
        temperature = np.trace(pressure, axis1=-2, axis2=-1) / (3 * rho)
        stress = pressure - (rho * temperature)[..., None, None] * np.eye(3)
        heat_flux = (raw[..., 10:13] - np.einsum("...ij,...j->...i", pressure, velocity)
                     - 0.5 * np.trace(second, axis1=-2, axis2=-1)[..., None] * velocity)
    moments = np.concatenate((rho[..., None], velocity, temperature[..., None]), axis=-1)
    conservative = np.concatenate((raw[..., :4],
                                   0.5 * np.trace(second, axis1=-2, axis2=-1)[..., None]), axis=-1)
    grad13 = np.concatenate((moments, stress[..., 0, 0, None], stress[..., 1, 1, None],
                             stress[..., 0, 1, None], stress[..., 0, 2, None], stress[..., 1, 2, None],
                             heat_flux), axis=-1)
    return dict(moments=moments, conservative_moments=conservative, pressure_tensor=pressure,
                stress_tensor=stress, heat_flux=heat_flux, raw_moments=raw, grad13=grad13,
                grad13_columns=np.array(GRAD13_COLUMNS))


def moment_snapshot_path(directory, time_value):
    return Path(directory) / f"moments_t{filename_float(time_value)}.npz"


def save_moment_snapshot(directory, *, time_value, moments, x, y, x_faces, y_faces,
                         metadata, **arrays):
    # The packed Grad representation remains available from raw_moments on
    # demand; do not duplicate it in any deterministic or DSMC archive.
    arrays = {key: value for key, value in arrays.items()
              if key not in ("grad13", "grad13_columns")}
    path = moment_snapshot_path(directory, time_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, time=np.array(time_value), moments=moments,
                 density=moments[..., 0], velocity=moments[..., 1:4], temperature=moments[..., 4],
                 x=x, y=y, x_faces=x_faces, y_faces=y_faces,
                 metadata_json=np.array(json.dumps(json_ready(metadata), sort_keys=True)),
                 **arrays)
    os.replace(temporary, path)
    return path
