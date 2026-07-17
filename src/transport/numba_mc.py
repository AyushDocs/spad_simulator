"""Numba-accelerated Monte Carlo transport kernel (pure-Python fallback if numba absent)."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(f, **kwargs):  # type: ignore[no-redef]
        return f


@njit(cache=True)
def mc_step(x: np.ndarray, v: np.ndarray,
            F: np.ndarray, dx: float,
            mu: float, vsat: float,
            dt: float, q_val: float,
            T: float, kB_val: float,
            m_eff: float) -> None:
    """Single Monte Carlo step for all particles.

    Updates positions and velocities in-place.
    """
    n = len(x)
    vth = np.sqrt(kB_val * T / m_eff)

    for i in range(n):
        # Drift
        xi = min(max(x[i] / dx, 0.0), len(F) - 1)
        idx = int(xi)
        frac = xi - idx
        if idx < len(F) - 1:
            E = F[idx] * (1 - frac) + F[idx + 1] * frac
        else:
            E = F[idx]

        # Acceleration
        a = q_val * E / m_eff
        v[i] += a * dt

        # Scattering (simplified)
        if np.random.random() < 0.1:
            v[i] = np.random.normal(0, vth)

        # Velocity saturation
        if abs(v[i]) > vsat:
            v[i] = np.sign(v[i]) * vsat

        # Update position
        x[i] += v[i] * dt * 1e2  # m/s -> cm/s (dt in s, x in cm)
