"""Poisson-only solver for 1D pn-junctions (no drift-diffusion).

Removed: Gummel iteration, Slotboom-variable continuity, and all
carrier-transport coupling that could not be made numerically stable.
The Poisson solver (Newton with Boltzmann statistics) works correctly.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from ..core.constants import q, VT
from ..utils._logging import get_logger

log = get_logger("transport.continuum")


def _harmonic_mean(arr: np.ndarray) -> np.ndarray:
    """Harmonic mean at half-points for interface quantities."""
    return 2.0 * arr[:-1] * arr[1:] / (arr[:-1] + arr[1:] + 1e-30)


def _equilibrium_np(net: float, ni: float) -> tuple[float, float]:
    """Equilibrium carrier densities from net doping and intrinsic density."""
    if net >= 0:
        n = 0.5 * (net + np.sqrt(net**2 + 4.0 * ni**2))
        p = ni**2 / max(n, 1e-10)
    else:
        p = 0.5 * (-net + np.sqrt(net**2 + 4.0 * ni**2))
        n = ni**2 / max(p, 1e-10)
    return n, p


def _solve_tridiagonal(a: np.ndarray, b: np.ndarray,
                       c: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve tridiagonal system using banded linear solver."""
    N = len(b)
    ab = np.zeros((3, N))
    ab[1, :] = b
    ab[0, 1:] = c[:-1]
    ab[2, :-1] = a[1:]
    return solve_banded((1, 1), ab, rhs)


class ContinuumSolver:
    """Newton-Poisson solver with Boltzmann carrier statistics.

    No drift-diffusion / continuity coupling.  Carrier densities are
    evaluated from the quasi-Fermi levels passed in by the user (or
    assumed flat at zero bias).
    """

    def __init__(self, grid, doping, material, T: float = 300.0):
        self.x = np.asarray(grid.x, dtype=float)
        self.dx = float(grid.dx)
        self.N = grid.no_of_nodes
        self.doping = doping
        self.T = T
        self.vth = float(VT(T))
        self.q_val = float(q.to("C").magnitude)

        self.ni = np.asarray(material.ni, dtype=float)
        self.eps = np.asarray(material.eps, dtype=float)

    def _contact_bias(self, Vbias: float) -> tuple[float, float]:
        """Ohmic-contact potentials at left (p) and right (n) terminals."""
        net = self.doping.net_doping(self.x)
        na0 = max(-net[0], self.ni[0])
        ndL = max(net[-1], self.ni[-1])
        phi_0 = -self.vth * np.log(na0 / self.ni[0]) if na0 > self.ni[0] else 0.0
        phi_L = Vbias + self.vth * np.log(ndL / self.ni[-1]) if ndL > self.ni[-1] else Vbias
        return float(phi_0), float(phi_L)

    def _quasi_fermi(self, phi, n, p):
        """Quasi-Fermi levels from potential and densities (post-processing)."""
        phi_n = phi - self.vth * np.log(np.maximum(n / self.ni, 1e-300))
        phi_p = phi + self.vth * np.log(np.maximum(p / self.ni, 1e-300))
        return phi_n, phi_p

    def _step_guess(self, Vbias: float) -> np.ndarray:
        """Tanh-step initial guess for pn-junction potential."""
        x = self.x
        N = self.N
        L = x[-1] - x[0]
        phi_0, phi_L = self._contact_bias(Vbias)

        net = self.doping.net_doping(x)
        signs = np.where(net >= 0, 1.0, -1.0)
        xj = L / 2.0
        for i in range(N - 1):
            if signs[i] < 0 and signs[i + 1] >= 0:
                frac = -net[i] / (net[i + 1] - net[i] + 1e-30)
                xj = float(x[i] + frac * (x[i + 1] - x[i]))
                break

        Vbi = max(phi_L - phi_0, 0.01)
        n_mid = float(np.max(np.abs(net[int(N * 0.4):int(N * 0.6)])))
        if n_mid > 1e10:
            eps_mid = float(np.mean(self.eps))
            W_dep = float(np.sqrt(2.0 * eps_mid * Vbi / (self.q_val * n_mid)))
            W_dep = float(np.clip(W_dep, L / 200.0, L / 2.0))
        else:
            W_dep = L / 4.0
        sig_len = max(W_dep / 3.0, self.dx * 2.0)
        sig = 0.5 * (1.0 + np.tanh((x - xj) / sig_len))
        return phi_0 + (phi_L - phi_0) * sig

    def _solve_poisson(self, Vbias: float,
                       phi_n: np.ndarray, phi_p: np.ndarray,
                       guess: np.ndarray | None = None) -> np.ndarray:
        """Solve Poisson equation with Boltzmann carrier statistics.

        Uses Newton-Raphson with position-dependent quasi-Fermi levels
        phi_n(x) and phi_p(x).
        """
        N = self.N
        vth = self.vth
        phi_0, phi_L = self._contact_bias(Vbias)

        if guess is not None:
            phi = guess.copy()
        else:
            phi = self._step_guess(Vbias)

        net = self.doping.net_doping(self.x)
        eps_half = 0.5 * (self.eps[:-1] + self.eps[1:])
        ep = eps_half[1:]
        em = eps_half[:-1]
        dx2 = self.dx**2

        q_val = self.q_val
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        rhs = np.zeros(N)

        for newton_iter in range(50):
            n = self.ni * np.exp(np.clip((phi - phi_n) / vth, -100, 100))
            p = self.ni * np.exp(np.clip((phi_p - phi) / vth, -100, 100))
            rho = q_val * (p - n + net)
            drho = -q_val / vth * (n + p)

            a[1:-1] = em / dx2
            b[1:-1] = -(ep + em) / dx2 + drho[1:-1]
            c[1:-1] = ep / dx2
            flux_div = (ep * (phi[2:] - phi[1:-1])
                        - em * (phi[1:-1] - phi[:-2])) / dx2
            rhs[1:-1] = -(flux_div + rho[1:-1])

            b[0] = 1.0
            c[0] = 0.0
            rhs[0] = -(phi[0] - phi_0)
            b[-1] = 1.0
            a[-1] = 0.0
            rhs[-1] = -(phi[-1] - phi_L)

            delta = _solve_tridiagonal(a, b, c, rhs)

            norm = float(np.max(np.abs(delta)))
            if norm < 1e-6:
                break

            if norm > 10.0:
                delta *= 10.0 / norm

            phi = phi + delta

        return phi

    def solve(self, Vbias: float, **kwargs) -> dict:
        """Solve Poisson with flat quasi-Fermi levels (no injection).

        Returns the potential, equilibrium-like carrier densities, and
        the built-in field.  No continuity / current computation.
        """
        x = self.x
        N = self.N
        net = self.doping.net_doping(x)

        n = np.zeros(N, dtype=float)
        p = np.zeros(N, dtype=float)
        for i in range(N):
            n[i], p[i] = _equilibrium_np(net[i], self.ni[i])

        phi_n = np.zeros(N, dtype=float)
        phi_p = np.zeros(N, dtype=float)
        phi = self._solve_poisson(Vbias, phi_n, phi_p)
        n = self.ni * np.exp(np.clip((phi - phi_n) / self.vth, -100, 100))
        p = self.ni * np.exp(np.clip((phi_p - phi) / self.vth, -100, 100))

        phi_n_out, phi_p_out = self._quasi_fermi(phi, n, p)

        return {
            "phi": phi,
            "n": n,
            "p": p,
            "phi_n": phi_n_out,
            "phi_p": phi_p_out,
            "Vbias": Vbias,
        }
