"""Sample Maxwellian-centered states for the FFT Boltzmann dataset."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from boltzmann_fft_gpu import BoltzmannFFT


KERNEL_CONSTANT = 1.0 / (4.0 * math.pi)
TARGET_MASS = 10.0
ALPHA_BANDS = ((0.05, 0.20), (0.20, 0.45), (0.45, 0.75), (0.75, 1.00), (1.00, 1.00))
COMPONENT_COUNTS = (1, 2, 3, 4)
POLYNOMIAL_MODULATION_PROBABILITY = 0.5
MASS_FLOOR = 1e-30
SAMPLING_FAMILY = "mixed_gaussian"


def synchronize_backend(backend: str):
    if backend == "cupy":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def to_numpy(array):
    return array.get() if getattr(array, "__module__", "").startswith("cupy") else np.asarray(array)


def truncation_condition_value(L: float, R: float) -> float:
    return (3.0 + math.sqrt(2.0)) * R / 4.0


def build_solver(
    *, n: int, L: float, R: float, angular_accuracy: int | None,
    backend: str, angular_chunk: int, use_rfft: bool,
) -> BoltzmannFFT:
    return BoltzmannFFT(
        n=n, L=L, R=R, radial_accuracy=n, angular_accuracy=angular_accuracy,
        omega=1.0, kernel_constant=KERNEL_CONSTANT, backend=backend,
        angular_chunk=angular_chunk, use_rfft=use_rfft,
    )


def _random_direction(rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=3)
    return direction / np.linalg.norm(direction)


def velocity_step(v: np.ndarray) -> float:
    return float(abs(v[1] - v[0]))


def discrete_moments(f: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, dv3: float) -> np.ndarray:
    moments = [np.sum(f), np.sum(f * vx), np.sum(f * vy), np.sum(f * vz),
               np.sum(f * (vx * vx + vy * vy + vz * vz))]
    return dv3 * np.array(moments, dtype=np.float64)


def normalize_to_mass(f: np.ndarray, dv3: float, mass: float) -> np.ndarray:
    return mass * f / (dv3 * np.sum(f))


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    rotation, triangular = np.linalg.qr(rng.normal(size=(3, 3)))
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    rotation = rotation @ np.diag(signs)
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def _random_covariance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues = np.exp(rng.uniform(np.log(0.30), np.log(1.80), size=3))
    rotation = _random_rotation(rng)
    return rotation @ np.diag(eigenvalues) @ rotation.T, eigenvalues


def _random_center(rng: np.random.Generator, L: float, R: float) -> np.ndarray:
    max_radius = min(4.7, 0.55 * R, 0.50 * L)
    return max_radius * rng.random() ** (1.0 / 3.0) * _random_direction(rng)


def _gaussian(vx, vy, vz, center: np.ndarray, covariance: np.ndarray):
    inv_cov = np.linalg.inv(covariance)
    dx, dy, dz = vx - center[0], vy - center[1], vz - center[2]
    diagonal = inv_cov[0, 0] * dx * dx + inv_cov[1, 1] * dy * dy + inv_cov[2, 2] * dz * dz
    cross = inv_cov[0, 1] * dx * dy + inv_cov[0, 2] * dx * dz + inv_cov[1, 2] * dy * dz
    normalizer = (2.0 * math.pi) ** 1.5 * math.sqrt(float(np.linalg.det(covariance)))
    return np.exp(-0.5 * (diagonal + 2.0 * cross)) / normalizer


def _apply_polynomial_modulation(
    f: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply ``epsilon + (1-epsilon)|v-u|^2/(3T)`` with probability one half."""

    if rng.random() >= POLYNOMIAL_MODULATION_PROBABILITY:
        return f, {"polynomial_applied": False, "polynomial_kind": "none"}

    mass = float(f.sum())
    center = np.array([np.sum(f * vx) / mass, np.sum(f * vy) / mass, np.sum(f * vz) / mass])
    dx, dy, dz = vx - center[0], vy - center[1], vz - center[2]
    temperature = float(np.sum(f * (dx * dx + dy * dy + dz * dz)) / (3.0 * mass))
    epsilon_draw = rng.random()
    if epsilon_draw < 0.10:
        epsilon = 0.0
    elif epsilon_draw < 0.30:
        epsilon = float(rng.uniform(0.0, 0.10))
    else:
        epsilon = float(rng.uniform(0.10, 1.0))
    radial_coefficient = (1.0 - epsilon) / (3.0 * temperature)
    polynomial = epsilon + radial_coefficient * (dx * dx + dy * dy + dz * dz)
    return f * polynomial, {
        "polynomial_applied": True,
        "polynomial_kind": "P_epsilon",
        "polynomial_center": center.tolist(),
        "polynomial_epsilon": epsilon,
        "polynomial_radial_coefficient": radial_coefficient,
        "polynomial_temperature": temperature,
    }


def _sample_gaussian_mixture(vx, vy, vz, rng: np.random.Generator, L: float, R: float):
    component_count = int(rng.choice(COMPONENT_COUNTS))
    bulk_cap = min(2.0, 0.22 * R, 0.20 * L)
    bulk_velocity = bulk_cap * rng.random() ** (1.0 / 3.0) * _random_direction(rng)
    if component_count == 1:
        weights = np.ones(1, dtype=np.float64)
        centers = bulk_velocity[None, :]
    else:
        weights = rng.dirichlet(np.full(component_count, 0.8))
        center_cap = min(4.8, 0.55 * R, 0.50 * L)
        relative_center_cap = max(1.5, center_cap - float(np.linalg.norm(bulk_velocity)))
        centers = np.zeros((component_count, 3), dtype=np.float64)
        for _ in range(100):
            raw_centers = np.stack([_random_center(rng, L, R) for _ in range(component_count)])
            candidate = raw_centers - np.sum(weights[:, None] * raw_centers, axis=0, keepdims=True)
            largest_radius = float(np.max(np.linalg.norm(candidate, axis=1)))
            if largest_radius > relative_center_cap:
                candidate *= relative_center_cap / largest_radius
            pair_distances = [np.linalg.norm(candidate[i] - candidate[j])
                              for i in range(component_count) for j in range(i)]
            centers = candidate + bulk_velocity[None, :]
            if min(pair_distances) >= 1.5:
                break

    distribution = np.zeros_like(vx)
    components = []
    for weight, center in zip(weights, centers, strict=True):
        covariance, eigenvalues = _random_covariance(rng)
        distribution += float(weight) * _gaussian(vx, vy, vz, center, covariance)
        components.append({
            "weight": float(weight), "center": center.tolist(),
            "theta_eigenvalues": eigenvalues.tolist(), "covariance": covariance.tolist(),
        })

    return distribution, {
        "base_kind": SAMPLING_FAMILY,
        "component_count": component_count,
        "bulk_velocity": bulk_velocity.tolist(),
        "mixture_velocity": np.sum(weights[:, None] * centers, axis=0).tolist(),
        "components": components,
    }


def _sample_base_distribution(vx, vy, vz, rng: np.random.Generator, L: float, R: float, dv3: float):
    distribution, metadata = _sample_gaussian_mixture(vx, vy, vz, rng, L, R)
    distribution, modulation = _apply_polynomial_modulation(distribution, vx, vy, vz, rng)
    distribution = normalize_to_mass(np.maximum(distribution, MASS_FLOOR), dv3, TARGET_MASS)
    metadata.update(modulation)
    return distribution, metadata


def _maxwellian_from_lambda(lambda_vec: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(phi @ lambda_vec, -745.0, 80.0))


def _initial_maxwellian_lambda(target: np.ndarray) -> np.ndarray:
    rho = float(target[0])
    velocity = target[1:4] / rho
    temperature = max(float((target[4] / rho - np.dot(velocity, velocity)) / 3.0), 1e-8)
    return np.array([
        math.log(rho / ((2.0 * math.pi * temperature) ** 1.5))
        - float(np.dot(velocity, velocity)) / (2.0 * temperature),
        velocity[0] / temperature, velocity[1] / temperature, velocity[2] / temperature,
        -1.0 / (2.0 * temperature),
    ], dtype=np.float64)


def _relative_moment_error(residual: np.ndarray, target: np.ndarray) -> float:
    rho = max(float(abs(target[0])), 1e-30)
    rms_speed = math.sqrt(max(float(target[4]) / rho, 1e-30))
    scales = np.array([rho, rho * rms_speed, rho * rms_speed, rho * rms_speed, rho * rms_speed**2])
    return float(np.max(np.abs(residual) / scales))


def _discrete_maxwellian_from_mesh(
    distribution: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
    dv3: float, max_iter: int = 30, tol: float = 1e-11,
) -> tuple[np.ndarray, dict[str, object]]:
    target = discrete_moments(distribution, vx, vy, vz, dv3)
    v2 = vx * vx + vy * vy + vz * vz
    phi = np.stack((np.ones_like(vx).ravel(), vx.ravel(), vy.ravel(), vz.ravel(), v2.ravel()), axis=1)
    lambda_vec = _initial_maxwellian_lambda(target)
    converged = False
    relative_error = math.inf

    for iteration in range(max_iter):
        maxwellian_flat = _maxwellian_from_lambda(lambda_vec, phi)
        residual = dv3 * (maxwellian_flat @ phi) - target
        relative_error = _relative_moment_error(residual, target)
        if relative_error < tol:
            converged = True
            break

        jacobian = dv3 * (phi.T @ (maxwellian_flat[:, None] * phi))
        try:
            delta = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            break
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = lambda_vec - damping * delta
            candidate_residual = dv3 * (_maxwellian_from_lambda(candidate, phi) @ phi) - target
            if _relative_moment_error(candidate_residual, target) < relative_error:
                lambda_vec = candidate
                break
        else:
            lambda_vec -= 0.03125 * delta

    maxwellian = _maxwellian_from_lambda(lambda_vec, phi).reshape(distribution.shape)
    maxwellian = normalize_to_mass(np.maximum(maxwellian, MASS_FLOOR), dv3, float(target[0]))
    moment_error = discrete_moments(maxwellian, vx, vy, vz, dv3) - target
    return maxwellian, {
        "converged": converged, "iterations": iteration + 1,
        "relative_error": relative_error, "moment_error": moment_error.tolist(),
    }


def discrete_maxwellian_matching_moments(
    distribution: np.ndarray, v: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    vx, vy, vz = np.meshgrid(v, v, v, indexing="ij")
    return _discrete_maxwellian_from_mesh(distribution, vx, vy, vz, velocity_step(v) ** 3)


def sample_distributions(
    v: np.ndarray, *, count: int, rng: np.random.Generator,
    L: float, R: float, alpha_slots: Sequence[int],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Sample independent states ``(1-alpha) M[g] + alpha g``."""

    vx, vy, vz = np.meshgrid(v, v, v, indexing="ij")
    dv3 = velocity_step(v) ** 3
    samples = np.empty((count, v.size, v.size, v.size), dtype=np.float64)
    metadata: list[dict[str, object]] = []
    for sample_index, alpha_slot in zip(range(count), alpha_slots, strict=True):
        distribution, sample_metadata = _sample_base_distribution(vx, vy, vz, rng, L, R, dv3)
        maxwellian, maxwellian_info = _discrete_maxwellian_from_mesh(distribution, vx, vy, vz, dv3)
        lower, upper = ALPHA_BANDS[int(alpha_slot)]
        alpha = float(rng.uniform(lower, upper))
        state = (1.0 - alpha) * maxwellian + alpha * distribution
        samples[sample_index] = normalize_to_mass(np.maximum(state, MASS_FLOOR), dv3, TARGET_MASS)
        metadata.append({
            **sample_metadata,
            "sample_index": sample_index,
            "alpha": alpha,
            "alpha_slot": int(alpha_slot),
            "maxwellian_newton_converged": bool(maxwellian_info["converged"]),
            "maxwellian_newton_iterations": int(maxwellian_info["iterations"]),
            "maxwellian_moment_error": maxwellian_info["moment_error"],
            "maxwellian_relative_error": float(maxwellian_info["relative_error"]),
        })
    return samples, metadata
