"""CuPy implementation of ``LinearCollision.h`` for ``Q[M,f] + Q[f,M]``."""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np

from boltzmann_fft_gpu import jacobi_radau_quadrature


ArrayBackend = Literal["cupy", "numpy"]


def _import_backend(name: ArrayBackend):
    if name == "numpy":
        return np
    import cupy as cp

    return cp


class LinearCollisionGPU:
    """Fast Fourier spectral evaluator matching ``LinearCollision.h``.

    Parameters correspond exactly to ``LinearCollision::reinit(B, R, L, N,
    acc_r)``.  Arrays have shape ``(n, n, n)`` and use the unshifted FFT order
    returned by the C++ ``get_velocity()`` method:
    ``[0, dv, ..., -2*dv, -dv]`` on each even-length axis.

    ``collision_kernel`` is the C++ radial kernel ``B(g)``.  It is evaluated
    once at each Jacobi--Radau node on the CPU, just as the templated C++
    callback is evaluated during ``reinit``.
    """

    def __init__(
        self, n: int, L: float, R: float, radial_accuracy: int,
        collision_kernel: Callable[[float], float] | None = None, *, backend: ArrayBackend = "cupy",
    ) -> None:
        self.n = int(n)
        self.L = float(L)
        self.R = float(R)
        self.radial_accuracy = int(radial_accuracy)
        self.backend = backend
        self.xp = _import_backend(backend)
        # LinearCollision.h and FFT.h are hard-wired to double precision.
        self.dtype = np.float64
        self.axes = (-3, -2, -1)
        self.grid_size = self.n**3
        self.dv = 2.0 * self.L / self.n

        # C++ uses N=(acc_r+1)/2 and then allocates N+1 Radau nodes.
        radau_degree = (self.radial_accuracy + 1) // 2
        radau_x, radau_w = jacobi_radau_quadrature(2.0, 0.0, radau_degree)
        radial_nodes = self.R * (1.0 - radau_x) / 2.0
        kernel_values = (np.ones_like(radial_nodes) if collision_kernel is None
                         else np.asarray([collision_kernel(float(g)) for g in radial_nodes], dtype=np.float64))

        self.radial_nodes_cpu = np.asarray(radial_nodes, dtype=np.float64)
        self.radau_weights_cpu = np.asarray(radau_w, dtype=np.float64)
        self.kernel_values_cpu = kernel_values

        # Reproduce the centered integer loops followed by inverse_fft_shift.
        left = self.n // 2
        centered_modes = np.arange(-left, self.n - left, dtype=np.float64)
        mx, my, mz = np.meshgrid(centered_modes, centered_modes, centered_modes, indexing="ij")
        xi_norm_centered = (math.pi / self.L) * np.sqrt(mx * mx + my * my + mz * mz)
        xi_norm = np.fft.ifftshift(xi_norm_centered)

        velocity_1d = np.fft.ifftshift(centered_modes * self.dv)
        vx, vy, vz = np.meshgrid(velocity_1d, velocity_1d, velocity_1d, indexing="ij")
        xp = self.xp
        self.velocity_1d = xp.asarray(velocity_1d, dtype=self.dtype)
        self.vx = xp.asarray(vx, dtype=self.dtype)
        self.vy = xp.asarray(vy, dtype=self.dtype)
        self.vz = xp.asarray(vz, dtype=self.dtype)
        self.xi_norm = xp.asarray(xi_norm, dtype=self.dtype)

        gain_term_data1 = []
        gain_term_data2 = []
        loss_term_data = xp.zeros((self.n, self.n, self.n), dtype=self.dtype)
        for g_i, w_i, kernel_i in zip(self.radial_nodes_cpu, self.radau_weights_cpu, self.kernel_values_cpu):
            # These are the exact coefficients in LinearCollision::reinit.
            coef_l = 2.0 * math.pi**2 * self.R**3 * w_i * kernel_i
            coef_g = math.sqrt(2.0 * coef_l)

            tmp = coef_g * self._sinc(g_i * self.xi_norm / 2.0)
            tmp_c = xp.fft.fftn(tmp, axes=self.axes) / self.grid_size
            gain_data1_i = xp.real(tmp_c)
            gain_term_data1.append(gain_data1_i)
            gain_term_data2.append(xp.fft.rfftn(gain_data1_i, axes=self.axes) / self.grid_size)

            tmp = coef_l * self._sinc(g_i * self.xi_norm)
            tmp_c = xp.fft.fftn(tmp, axes=self.axes) / self.grid_size
            loss_term_data = loss_term_data + xp.real(tmp_c)

        self.gain_term_data1 = xp.stack(gain_term_data1, axis=0)
        self.gain_term_data2 = xp.stack(gain_term_data2, axis=0)
        self.loss_term_data = loss_term_data

    def _sinc(self, x):
        """Match the two-branch C++ ``sinc`` helper, including its cutoff."""

        xp = self.xp
        small = xp.abs(x) < 1e-4
        safe_x = xp.where(small, 1.0, x)
        return xp.where(small, 1.0 - x * x / 6.0, xp.sin(safe_x) / safe_x)

    def get_velocity(self):
        """Return C++-ordered velocity points with shape ``(n**3, 3)``."""

        return self.xp.stack((self.vx, self.vy, self.vz), axis=-1).reshape((-1, 3))

    def velocity_grid(self):
        """Return the one-dimensional velocity axis in the C++ FFT order."""

        return self.velocity_1d.copy()

    def _convolve(self, left, right):
        """Match ``FFT_R2C::convolve`` including its final size scaling."""

        product = self.xp.fft.rfftn(left, axes=self.axes) * self.xp.fft.rfftn(right, axes=self.axes)
        return self.xp.fft.irfftn(product, s=(self.n, self.n, self.n), axes=self.axes)

    def linear_collision(
        self, f, density: float, velocity, temperature: float, *, conserve_mass: bool = False
    ):
        """Return ``Q[M, f] + Q[f, M]`` for one C++-ordered grid.

        The input and output are point values, matching the comment and API in
        ``LinearCollision.h``.  All calculations use float64/complex128 when
        the default dtype is retained.
        """

        xp = self.xp
        values = xp.asarray(f, dtype=self.dtype)

        density = float(density)
        temperature = float(temperature)
        bulk_velocity = xp.asarray(velocity, dtype=self.dtype)

        v_sqr_norm = self.vx * self.vx + self.vy * self.vy + self.vz * self.vz
        cx = self.vx - bulk_velocity[0]
        cy = self.vy - bulk_velocity[1]
        cz = self.vz - bulk_velocity[2]
        c_sqr_norm = cx * cx + cy * cy + cz * cz

        coef_f = math.pow(2.0 * math.pi * temperature, -1.5)
        coef_g = math.pow(4.0 * math.pi * temperature, -1.5)
        coef_h = math.pow(math.pi * temperature, -1.5)
        maxwellian_f = coef_f * xp.exp(-c_sqr_norm / (2.0 * temperature))
        maxwellian_g = coef_g * xp.exp(-v_sqr_norm / temperature)
        maxwellian_h = coef_h * xp.exp(-c_sqr_norm / temperature)

        # This threshold and the division by n**3 are literal translations of
        # the non-quadruple-precision C++ branch.
        ratio = xp.zeros_like(values)
        retained = maxwellian_f >= 1e-9
        ratio[retained] = values[retained] / maxwellian_f[retained] / self.grid_size

        ratio_k = xp.fft.rfftn(ratio, axes=self.axes)
        half_shape = (self.n, self.n, self.n // 2 + 1)
        linear_k = xp.zeros(half_shape, dtype=xp.complex128)
        for gain_data1_i, gain_data2_i in zip(self.gain_term_data1, self.gain_term_data2):
            work = maxwellian_g * gain_data1_i
            work_k = xp.fft.rfftn(work, axes=self.axes)
            work_k = work_k * ratio_k

            # FFTW's c2r inverse is unnormalized, unlike NumPy/CuPy irfftn.
            work = self.grid_size * xp.fft.irfftn(work_k, s=(self.n, self.n, self.n), axes=self.axes)
            work = work * maxwellian_h
            work_k = xp.fft.rfftn(work, axes=self.axes)
            work_k = work_k * gain_data2_i
            linear_k = linear_k + density * work_k

        # The final C++ inverse_transform is also deliberately unnormalized.
        result = self.grid_size * xp.fft.irfftn(linear_k, s=(self.n, self.n, self.n), axes=self.axes)

        loss_against_maxwellian = self._convolve(self.loss_term_data, maxwellian_f)
        loss_against_f = self._convolve(self.loss_term_data, values)
        result -= density * (loss_against_f * maxwellian_f + values * loss_against_maxwellian)

        if conserve_mass:
            result = result - xp.sum(result) / self.grid_size
        return result

    def linear_collision_batch(
        self, f, density, velocity, temperature, *, conserve_mass: bool = False
    ):
        """Batched ``Q[M_i,f_i]+Q[f_i,M_i]`` on resident arrays.

        This is the batch analogue of :meth:`linear_collision`.  It preserves
        the same C++ FFT ordering and arithmetic while vectorizing independent
        spatial cells over the leading dimension.
        """

        xp = self.xp
        values = xp.asarray(f, dtype=self.dtype)
        batch = values.shape[0]

        density_values = xp.asarray(density, dtype=self.dtype)
        velocity_values = xp.asarray(velocity, dtype=self.dtype)
        temperature_values = xp.asarray(temperature, dtype=self.dtype)
        scalar_shape = (batch, 1, 1, 1)
        rho = density_values.reshape(scalar_shape)
        theta = temperature_values.reshape(scalar_shape)
        ux = velocity_values[:, 0].reshape(scalar_shape)
        uy = velocity_values[:, 1].reshape(scalar_shape)
        uz = velocity_values[:, 2].reshape(scalar_shape)

        v_sqr_norm = self.vx * self.vx + self.vy * self.vy + self.vz * self.vz
        cx = self.vx[None] - ux
        cy = self.vy[None] - uy
        cz = self.vz[None] - uz
        c_sqr_norm = cx * cx + cy * cy + cz * cz

        coef_f = xp.power(2.0 * math.pi * theta, -1.5)
        coef_g = xp.power(4.0 * math.pi * theta, -1.5)
        coef_h = xp.power(math.pi * theta, -1.5)
        maxwellian_f = coef_f * xp.exp(-c_sqr_norm / (2.0 * theta))
        maxwellian_g = coef_g * xp.exp(-v_sqr_norm[None] / theta)
        maxwellian_h = coef_h * xp.exp(-c_sqr_norm / theta)

        ratio = xp.zeros_like(values)
        retained = maxwellian_f >= 1e-9
        ratio[retained] = values[retained] / maxwellian_f[retained] / self.grid_size

        ratio_k = xp.fft.rfftn(ratio, axes=self.axes)
        half_shape = (batch, self.n, self.n, self.n // 2 + 1)
        linear_k = xp.zeros(half_shape, dtype=xp.complex128)
        for gain_data1_i, gain_data2_i in zip(self.gain_term_data1, self.gain_term_data2):
            work = maxwellian_g * gain_data1_i[None]
            work_k = xp.fft.rfftn(work, axes=self.axes) * ratio_k
            work = self.grid_size * xp.fft.irfftn(work_k, s=(self.n, self.n, self.n), axes=self.axes)
            work = work * maxwellian_h
            work_k = xp.fft.rfftn(work, axes=self.axes)
            linear_k = linear_k + rho * work_k * gain_data2_i[None]

        result = self.grid_size * xp.fft.irfftn(linear_k, s=(self.n, self.n, self.n), axes=self.axes)
        loss_against_maxwellian = self._convolve(self.loss_term_data, maxwellian_f)
        loss_against_f = self._convolve(self.loss_term_data, values)
        result -= rho * (loss_against_f * maxwellian_f + values * loss_against_maxwellian)

        if conserve_mass:
            result -= xp.sum(result, axis=self.axes, keepdims=True) / self.grid_size
        return result

# The shorter name mirrors the C++ class while keeping the GPU-first module
# name explicit at import sites.
LinearCollision = LinearCollisionGPU
