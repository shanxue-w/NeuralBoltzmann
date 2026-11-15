"""GPU-first FFT spectral Boltzmann collision operator on a uniform velocity grid."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Literal

import numpy as np

ArrayBackend = Literal["cupy", "numpy"]
LossMode = Literal["angular", "sinc"]


def recommended_collision_radius(L: float) -> float:
    return 0.9 * float(L)


def jacobi_values_and_derivatives(
    alpha: float, beta: float, degree: int, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Jacobi P_degree^(alpha,beta) and its derivative.

    This mirrors the three-term recurrence used in ``quadrature.c``.
    """

    x = np.asarray(x, dtype=np.float64)
    if degree == 0:
        return np.ones_like(x), np.zeros_like(x)

    apb = alpha + beta
    p_prev = np.ones_like(x)
    dp_prev = np.zeros_like(x)
    p_cur = (alpha - beta + (apb + 2.0) * x) / 2.0
    dp_cur = np.full_like(x, (apb + 2.0) / 2.0)
    if degree == 1:
        return p_cur, dp_cur

    for i in range(2, degree + 1):
        a1 = 2.0 * i * (i + apb) * (2.0 * i + apb - 2.0)
        a2 = (2.0 * i + apb - 1.0) * (alpha * alpha - beta * beta)
        b3 = 2.0 * i + apb - 2.0
        a3 = b3 * (b3 + 1.0) * (b3 + 2.0)
        a4 = 2.0 * (i + alpha - 1.0) * (i + beta - 1.0) * (2.0 * i + apb)
        p_next = ((a2 + a3 * x) * p_cur - a4 * p_prev) / a1
        dp_next = ((a2 + a3 * x) * dp_cur - a4 * dp_prev + a3 * p_cur) / a1
        p_prev, p_cur = p_cur, p_next
        dp_prev, dp_cur = dp_cur, dp_next

    return p_cur, dp_cur


def jacobi_gauss_nodes(alpha: float, beta: float, degree: int) -> np.ndarray:
    """Return Gauss-Jacobi nodes for degree ``degree``.

    The returned array has ``degree + 1`` nodes and uses the same Golub-Welsch
    tridiagonal coefficients as ``Jacobi_Gauss_quadrature`` in ``quadrature.c``.
    """

    if degree == 0:
        return np.array([(alpha - beta) / (alpha + beta + 2.0)], dtype=np.float64)

    n = degree + 1
    diag = np.empty(n, dtype=np.float64)
    subdiag = np.empty(n - 1, dtype=np.float64)
    apb = alpha + beta
    for i in range(n):
        denom = (2.0 + apb + 2.0 * i) * (apb + 2.0 * i)
        diag[i] = (beta * beta - alpha * alpha) / denom
    if abs(apb) < 1e-15:
        diag[0] = 0.0
    for i in range(n - 1):
        numerator = (i + 1.0) * (i + 1.0 + apb) * (i + 1.0 + alpha) * (i + 1.0 + beta)
        denominator = (2.0 * i + 1.0 + apb) * (2.0 * i + 3.0 + apb)
        subdiag[i] = 2.0 * math.sqrt(numerator / denominator) / (2.0 + apb + 2.0 * i)

    tri = np.diag(diag) + np.diag(subdiag, 1) + np.diag(subdiag, -1)
    return np.linalg.eigvalsh(tri)


def jacobi_radau_quadrature(alpha: float, beta: float, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Jacobi-Radau quadrature with the left endpoint ``x=-1``.

    It approximates ``int_-1^1 (1-x)^alpha (1+x)^beta phi(x) dx`` and matches
    the ``Jacobi_Radau_quadrature`` routine used by ``QuadratureOnBall.h``.
    """

    x = np.empty(degree + 1, dtype=np.float64)
    w = np.empty(degree + 1, dtype=np.float64)
    x[0] = -1.0

    if degree == 0:
        log_w0 = (alpha + beta + 1.0) * math.log(2.0)
        log_w0 += math.lgamma(beta + 1.0) + math.lgamma(alpha + 1.0)
        w[0] = math.exp(log_w0 - math.lgamma(alpha + beta + 2.0))
        return x, w

    x[1:] = jacobi_gauss_nodes(alpha, beta + 1.0, degree - 1)
    _, derivative = jacobi_values_and_derivatives(alpha, beta + 1.0, degree, x[1:])

    log_gn = (alpha + beta + 2.0) * math.log(2.0)
    log_gn += math.lgamma(degree + alpha + 1.0) + math.lgamma(degree + beta + 2.0)
    log_gn -= math.lgamma(degree + 1.0) + math.lgamma(degree + alpha + beta + 2.0)
    gn = math.exp(log_gn)
    w[1:] = gn / ((1.0 + x[1:]) * (1.0 - x[1:] * x[1:]) * derivative * derivative)
    log_w0 = (alpha + beta + 1.0) * math.log(2.0) + 2.0 * math.lgamma(beta + 1.0)
    log_w0 += math.lgamma(degree + 1.0) + math.lgamma(degree + alpha + 1.0)
    log_w0 -= math.lgamma(degree + beta + 2.0) + math.lgamma(degree + alpha + beta + 2.0)
    w[0] = math.exp(log_w0) * (beta + 1.0)
    return x, w


def radial_g2_quadrature(radius: float, accuracy: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature for ``int_0^R g^2 psi(g) dg``.

    ``accuracy`` follows ``QuadratureOnBall.h``: the Radau degree is
    ``(accuracy + 1) // 2``.
    """

    degree = (accuracy + 1) // 2
    x, w = jacobi_radau_quadrature(alpha=2.0, beta=0.0, degree=degree)
    r = (1.0 - x) / 2.0
    nodes = radius * r
    weights = (radius**3) * w / 8.0
    return nodes, weights


def lebedev_hemisphere_quadrature(
    accuracy: int, directory: str | Path = "Lebedev"
) -> tuple[np.ndarray, np.ndarray]:
    """Load antipodal-paired Lebedev nodes on one hemisphere.

    The repository's Lebedev files store normalized full-sphere weights whose
    sum is one.  For the paper's convention
    ``int_{S^2_+} phi(sigma) d sigma ~= 2*pi*sum_m w_m phi(sigma_m)``,
    the returned hemisphere weights also sum to one.
    """

    directory = Path(directory)
    candidates = sorted(int(p.stem.split("_")[-1]) for p in directory.glob("lebedev_*.txt"))
    chosen = next((value for value in candidates if value >= accuracy), None)
    if chosen is None:
        raise ValueError(f"Lebedev quadrature with accuracy >= {accuracy} not found in {directory}")

    data = np.loadtxt(directory / f"lebedev_{chosen:03d}.txt", dtype=np.float64)
    theta = np.deg2rad(data[:, 0])
    phi = np.deg2rad(data[:, 1])
    points = np.column_stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)))
    weights = data[:, 2]

    used = np.zeros(points.shape[0], dtype=bool)
    hemi_points: list[np.ndarray] = []
    hemi_weights: list[float] = []
    tol = 1e-12

    def choose_positive(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        if p[2] > tol:
            return p
        if q[2] > tol:
            return q
        for axis in range(3):
            if p[axis] > tol:
                return p
            if q[axis] > tol:
                return q
        return p

    for i, point in enumerate(points):
        if used[i]:
            continue
        distances = np.linalg.norm(points + point, axis=1)
        distances[used] = np.inf
        distances[i] = np.inf
        j = int(np.argmin(distances))
        used[i] = True
        used[j] = True
        hemi_points.append(choose_positive(points[i], points[j]))
        hemi_weights.append(float(weights[i] + weights[j]))

    return np.asarray(hemi_points, dtype=np.float64), np.asarray(hemi_weights, dtype=np.float64)


def _import_backend(name: ArrayBackend):
    if name == "numpy":
        return np
    import cupy as cp

    return cp


class BoltzmannFFT:
    """Fast Fourier spectral evaluator for ``Q[f, f]``.

    Parameters use the notation of the paper: the velocity domain is
    ``[-L, L)^3`` and the collision kernel is truncated at ``R``.
    The input to :meth:`collision` may have shape ``(n, n, n)`` or
    ``(..., n, n, n)`` for batched spatial points.
    """

    def __init__(
        self,
        n: int,
        L: float,
        R: float | None = None,
        radial_accuracy: int | None = None,
        angular_accuracy: int | None = None,
        *,
        omega: float = 1.0,
        kernel_constant: float = 1.0,
        collision_kernel: Callable[[np.ndarray], np.ndarray] | None = None,
        backend: ArrayBackend = "cupy",
        dtype=np.float64,
        lebedev_dir: str | Path = "Lebedev",
        loss_mode: LossMode = "sinc",
        enforce_mass: bool = True,
        angular_chunk: int = 8,
        use_rfft: bool = True,
    ) -> None:
        self.n = int(n)
        self.L = float(L)
        self.R = recommended_collision_radius(L) if R is None else float(R)
        self.radial_accuracy = int(n if radial_accuracy is None else radial_accuracy)
        self.angular_accuracy = int(n if angular_accuracy is None else angular_accuracy)
        self.omega = float(omega)
        self.kernel_constant = float(kernel_constant)
        self.backend = backend
        self.xp = _import_backend(backend)
        self.dtype = dtype
        self.loss_mode = loss_mode
        self.enforce_mass = bool(enforce_mass)
        self.angular_chunk = int(angular_chunk)
        self.use_rfft = bool(use_rfft)
        self.axes = (-3, -2, -1)
        self.dv = 2.0 * self.L / self.n
        self.dv3 = self.dv * self.dv * self.dv
        self.grid_size = self.n**3

        radial_nodes, radial_weights = radial_g2_quadrature(self.R, self.radial_accuracy)
        angular_nodes, angular_weights = lebedev_hemisphere_quadrature(self.angular_accuracy, lebedev_dir)
        kernel_values = (self.kernel_constant * radial_nodes ** (2.0 * (1.0 - self.omega))
                         if collision_kernel is None else np.asarray(collision_kernel(radial_nodes), dtype=np.float64))

        self.radial_nodes_cpu = radial_nodes
        self.radial_weights_cpu = radial_weights
        self.angular_nodes_cpu = angular_nodes
        self.angular_weights_cpu = angular_weights
        self.kernel_values_cpu = kernel_values

        xp = self.xp
        self.radial_nodes = xp.asarray(radial_nodes, dtype=dtype)
        self.radial_factor = xp.asarray(radial_weights * kernel_values, dtype=dtype)
        self.angular_nodes = xp.asarray(angular_nodes, dtype=dtype)
        self.angular_weights = xp.asarray(angular_weights, dtype=dtype)

        modes = np.arange(-self.n // 2, self.n // 2, dtype=np.float64)
        kx, ky, kz = np.meshgrid(modes, modes, modes, indexing="ij")
        self.kx = xp.asarray(kx, dtype=dtype)
        self.ky = xp.asarray(ky, dtype=dtype)
        self.kz = xp.asarray(kz, dtype=dtype)
        self.k_norm = xp.sqrt(self.kx * self.kx + self.ky * self.ky + self.kz * self.kz)
        half_slice = (slice(None), slice(None), slice(0, self.n // 2 + 1))
        kx_u = np.fft.ifftshift(kx, axes=(0, 1, 2))[half_slice]
        ky_u = np.fft.ifftshift(ky, axes=(0, 1, 2))[half_slice]
        kz_u = np.fft.ifftshift(kz, axes=(0, 1, 2))[half_slice]
        self.kx_half = xp.asarray(kx_u, dtype=dtype)
        self.ky_half = xp.asarray(ky_u, dtype=dtype)
        self.kz_half = xp.asarray(kz_u, dtype=dtype)
        self.k_norm_half = xp.sqrt(self.kx_half**2 + self.ky_half**2 + self.kz_half**2)
        self.loss_kernel = self._build_loss_kernel()
        self.loss_kernel_half = self._build_loss_kernel_half()
        self.coeff = 2.0 * math.pi * math.pi / (self.L**3)
        self.center = (self.n // 2, self.n // 2, self.n // 2)

    def velocity_grid(self):
        """Return the one-dimensional grid for each velocity component."""

        return self.xp.asarray((np.arange(self.n, dtype=np.float64) - self.n // 2) * self.dv, dtype=self.dtype)

    def collision(self, f, *, batch_chunk: int | None = None):
        """Return ``Q[f, f]`` on the physical velocity grid.

        ``f`` may have shape ``(n, n, n)`` or ``(..., n, n, n)``.  Leading
        dimensions are treated as independent spatial/batch points.
        """

        values = self.xp.asarray(f, dtype=self.dtype)
        lead_shape = tuple(int(v) for v in values.shape[:-3])
        flat = values.reshape((-1, self.n, self.n, self.n))
        pieces = []
        step = batch_chunk or flat.shape[0]
        for start in range(0, flat.shape[0], step):
            chunk = flat[start : start + step]
            if self.use_rfft and self.xp.isrealobj(chunk):
                pieces.append(self._collision_values_rfft_flat(chunk))
            else:
                f_hat = self._values_to_hat(chunk)
                q_hat = self._collision_hat_flat(f_hat)
                pieces.append(self._hat_to_values(q_hat))
        q_flat = pieces[0] if len(pieces) == 1 else self.xp.concatenate(pieces, axis=0)
        q = q_flat.reshape((*lead_shape, self.n, self.n, self.n))
        if self.xp.isrealobj(values):
            q = self.xp.real(q)
        return q

    def collision_hat_from_values(self, f, *, batch_chunk: int | None = None):
        """Return Fourier coefficients of ``Q[f, f]`` from physical values."""

        values = self.xp.asarray(f, dtype=self.dtype)
        lead_shape = tuple(int(v) for v in values.shape[:-3])
        flat = values.reshape((-1, self.n, self.n, self.n))
        pieces = []
        step = batch_chunk or flat.shape[0]
        for start in range(0, flat.shape[0], step):
            pieces.append(self._collision_hat_flat(self._values_to_hat(flat[start : start + step])))
        q_hat = pieces[0] if len(pieces) == 1 else self.xp.concatenate(pieces, axis=0)
        return q_hat.reshape((*lead_shape, self.n, self.n, self.n))

    def collision_hat(self, f_hat, *, batch_chunk: int | None = None):
        """Return Fourier coefficients of ``Q[f, f]`` from Fourier coefficients."""

        coeffs = self.xp.asarray(f_hat)
        lead_shape = tuple(int(v) for v in coeffs.shape[:-3])
        flat = coeffs.reshape((-1, self.n, self.n, self.n))
        pieces = []
        step = batch_chunk or flat.shape[0]
        for start in range(0, flat.shape[0], step):
            pieces.append(self._collision_hat_flat(flat[start : start + step]))
        q_hat = pieces[0] if len(pieces) == 1 else self.xp.concatenate(pieces, axis=0)
        return q_hat.reshape((*lead_shape, self.n, self.n, self.n))

    def _values_to_hat(self, values):
        shifted = self.xp.fft.ifftshift(values, axes=self.axes)
        return self.dv3 * self.xp.fft.fftshift(self.xp.fft.fftn(shifted, axes=self.axes), axes=self.axes)

    def _values_to_hat_half(self, values):
        shifted = self.xp.fft.ifftshift(values, axes=self.axes)
        return self.dv3 * self.xp.fft.rfftn(shifted, axes=self.axes)

    def _hat_to_values(self, coeffs):
        shifted = self.xp.fft.ifftshift(coeffs, axes=self.axes)
        return self.xp.fft.fftshift(self.xp.fft.ifftn(shifted, axes=self.axes), axes=self.axes) / self.dv3

    def _hat_half_to_values(self, coeffs):
        values = self.xp.fft.irfftn(coeffs, s=(self.n, self.n, self.n), axes=self.axes)
        return self.xp.fft.fftshift(values, axes=self.axes) / self.dv3

    def _cyclic_convolution(self, left, right):
        left_u = self.xp.fft.ifftshift(left, axes=self.axes)
        right_u = self.xp.fft.ifftshift(right, axes=self.axes)
        product = self.xp.fft.fftn(left_u, axes=self.axes) * self.xp.fft.fftn(right_u, axes=self.axes)
        conv_u = self.xp.fft.ifftn(product, axes=self.axes)
        return self.xp.fft.fftshift(conv_u, axes=self.axes)

    def _cyclic_convolution_half(self, left, right):
        left_values = self.xp.fft.irfftn(left, s=(self.n, self.n, self.n), axes=self.axes)
        right_values = self.xp.fft.irfftn(right, s=(self.n, self.n, self.n), axes=self.axes)
        return self.grid_size * self.xp.fft.rfftn(left_values * right_values, axes=self.axes)

    def _gain_convolution_half(self, f_hat, phase):
        # Match BinaryCollision.h: split exp(i phase) into cos and i*sin so
        # the RFFT half-spectrum keeps the Hermitian structure at Nyquist modes.
        shape = (self.n, self.n, self.n)
        cos_values = self.xp.fft.irfftn(
            f_hat[:, None] * self.xp.cos(phase)[None], s=shape, axes=self.axes
        )
        sin_values = self.xp.fft.irfftn(
            f_hat[:, None] * (1j * self.xp.sin(phase)[None]), s=shape, axes=self.axes
        )
        return self.grid_size * self.xp.fft.rfftn(cos_values * cos_values - sin_values * sin_values, axes=self.axes)

    def _sinc(self, x):
        xp = self.xp
        safe_x = xp.where(xp.abs(x) < 1e-14, 1.0, x)
        return xp.where(xp.abs(x) < 1e-14, 1.0, xp.sin(safe_x) / safe_x)

    @staticmethod
    def _sigma_dot_k(sigma, kx, ky, kz):
        sx = sigma[:, 0, None, None, None]
        sy = sigma[:, 1, None, None, None]
        sz = sigma[:, 2, None, None, None]
        return sx * kx + sy * ky + sz * kz

    def _build_loss_kernel(self):
        xp = self.xp
        psi = xp.zeros((self.n, self.n, self.n), dtype=self.dtype)
        if self.loss_mode == "sinc":
            for gj, factor in zip(self.radial_nodes, self.radial_factor):
                psi = psi + factor * self._sinc(math.pi * gj * self.k_norm / self.L)
            return psi

        for gj, factor in zip(self.radial_nodes, self.radial_factor):
            angular_sum = xp.zeros_like(psi)
            for start in range(0, int(self.angular_weights.shape[0]), self.angular_chunk):
                stop = min(start + self.angular_chunk, int(self.angular_weights.shape[0]))
                sigma = self.angular_nodes[start:stop]
                weight = self.angular_weights[start:stop]
                sigma_dot_k = self._sigma_dot_k(sigma, self.kx, self.ky, self.kz)
                phase = math.pi * gj * sigma_dot_k / self.L
                angular_sum += xp.sum(weight[:, None, None, None] * xp.cos(phase), axis=0)
            psi += factor * angular_sum
        return psi

    def _build_loss_kernel_half(self):
        xp = self.xp
        psi = xp.zeros((self.n, self.n, self.n // 2 + 1), dtype=self.dtype)
        if self.loss_mode == "sinc":
            for gj, factor in zip(self.radial_nodes, self.radial_factor):
                psi = psi + factor * self._sinc(math.pi * gj * self.k_norm_half / self.L)
            return psi

        for gj, factor in zip(self.radial_nodes, self.radial_factor):
            angular_sum = xp.zeros_like(psi)
            for start in range(0, int(self.angular_weights.shape[0]), self.angular_chunk):
                stop = min(start + self.angular_chunk, int(self.angular_weights.shape[0]))
                sigma = self.angular_nodes[start:stop]
                weight = self.angular_weights[start:stop]
                sigma_dot_k = self._sigma_dot_k(sigma, self.kx_half, self.ky_half, self.kz_half)
                phase = math.pi * gj * sigma_dot_k / self.L
                angular_sum += xp.sum(weight[:, None, None, None] * xp.cos(phase), axis=0)
            psi += factor * angular_sum
        return psi

    def _collision_values_rfft_flat(self, values):
        xp = self.xp
        f_hat = self._values_to_hat_half(values)
        complex_dtype = xp.result_type(f_hat.dtype, xp.complex128)
        f_hat = f_hat.astype(complex_dtype, copy=False)
        q_hat = xp.zeros_like(f_hat, dtype=complex_dtype)

        for gj, factor in zip(self.radial_nodes, self.radial_factor):
            phi = factor * self._sinc(math.pi * gj * self.k_norm_half / (2.0 * self.L))
            for start in range(0, int(self.angular_weights.shape[0]), self.angular_chunk):
                stop = min(start + self.angular_chunk, int(self.angular_weights.shape[0]))
                sigma = self.angular_nodes[start:stop]
                weight = self.angular_weights[start:stop]
                sigma_dot_k = self._sigma_dot_k(sigma, self.kx_half, self.ky_half, self.kz_half)
                phase = math.pi * gj * sigma_dot_k / (2.0 * self.L)
                conv = self._gain_convolution_half(f_hat, phase)
                q_hat += xp.sum(weight[None, :, None, None, None] * conv, axis=1) * phi[None]

        loss = self._cyclic_convolution_half(f_hat, f_hat * self.loss_kernel_half[None, :, :, :])
        q_hat = self.coeff * (q_hat - loss)

        if self.enforce_mass:
            q_hat[(slice(None), 0, 0, 0)] = 0.0
        return self._hat_half_to_values(q_hat)

    def _collision_hat_flat(self, f_hat):
        xp = self.xp
        complex_dtype = xp.result_type(f_hat.dtype, xp.complex128)
        f_hat = f_hat.astype(complex_dtype, copy=False)
        q_hat = xp.zeros_like(f_hat, dtype=complex_dtype)

        for gj, factor in zip(self.radial_nodes, self.radial_factor):
            phi = factor * self._sinc(math.pi * gj * self.k_norm / (2.0 * self.L))
            for start in range(0, int(self.angular_weights.shape[0]), self.angular_chunk):
                stop = min(start + self.angular_chunk, int(self.angular_weights.shape[0]))
                sigma = self.angular_nodes[start:stop]
                weight = self.angular_weights[start:stop]
                sigma_dot_k = self._sigma_dot_k(sigma, self.kx, self.ky, self.kz)
                phase = math.pi * gj * sigma_dot_k / (2.0 * self.L)
                exp_phase = xp.exp(1j * phase)[None, ...]
                weighted_f_plus = f_hat[:, None, :, :, :] * exp_phase
                weighted_f_minus = f_hat[:, None, :, :, :] * xp.conj(exp_phase)
                conv = self._cyclic_convolution(weighted_f_plus, weighted_f_minus)
                q_hat += xp.sum(weight[None, :, None, None, None] * conv, axis=1) * phi[None]

        loss = self._cyclic_convolution(f_hat, f_hat * self.loss_kernel[None, :, :, :])
        q_hat = self.coeff * (q_hat - loss)

        if self.enforce_mass:
            q_hat[(slice(None), *self.center)] = 0.0
        return q_hat
