from __future__ import annotations

from typing import Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator
from scipy.linalg import solve_banded

from ..core.grid import Grid1D
from ..core.doping import DopingProfile
from ..core.constants import q, VT
from ..utils._exceptions import ConvergenceError, PhysicsError
from ..utils._logging import get_logger
from ..utils.pydantic_types import NDArray
from .numba_solver import newton_iteration

log = get_logger("poisson")


class PoissonSolver(BaseModel):
    """
    Nonlinear 1-D Poisson solver with Newton-Raphson iteration.

    Boltzmann carrier statistics:
        n(x) = ni(x) exp((phi - phi_n) / V_T)
        p(x) = ni(x) exp((phi_p - phi) / V_T)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: Grid1D
    T: float
    doping: DopingProfile
    eps_grid: NDArray
    ni_grid: NDArray
    tol: float = 5e-4
    max_iter: int = 300
    max_dphi: float = 50.0
    damp: float = 0.5

    _eps: np.ndarray = PrivateAttr()
    _ni: np.ndarray = PrivateAttr()
    _eps_half: np.ndarray = PrivateAttr()
    _q_val: float = PrivateAttr()

    @field_validator("T")
    @classmethod
    def _T_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Temperature must be positive, got {v}")
        return v

    @field_validator("tol")
    @classmethod
    def _tol_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Tolerance must be positive, got {v}")
        return v

    @field_validator("max_iter")
    @classmethod
    def _iter_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"max_iter must be positive, got {v}")
        return v

    @field_validator("damp")
    @classmethod
    def _damp_range(cls, v: float) -> float:
        if v <= 0 or v > 1:
            raise ValueError(f"damp must be in (0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _init_arrays(self):
        self._eps = np.asarray(self.eps_grid, dtype=float)
        self._ni = np.asarray(self.ni_grid, dtype=float)
        self._eps_half = 0.5 * (self._eps[:-1] + self._eps[1:])
        self._q_val = float(q.to("C").magnitude)
        return self

    def _contact_bias(self, Vbias: float) -> Tuple[float, float]:
        x = self.grid.x
        net = self.doping.net_doping(x)
        vth = VT(self.T)
        na0 = max(-net[0], self._ni[0])
        ndL = max(net[-1], self._ni[-1])
        phi_0 = -vth * np.log(na0 / self._ni[0]) if na0 > self._ni[0] else 0.0
        phi_L = Vbias + vth * np.log(ndL / self._ni[-1]) if ndL > self._ni[-1] else Vbias
        return float(phi_0), float(phi_L)

    def _carrier_densities(self, phi: np.ndarray,
                            phi_n: float, phi_p: float
                            ) -> Tuple[np.ndarray, np.ndarray]:
        vth = VT(self.T)
        max_arg = np.log(np.finfo(float).max / np.max(self._ni))
        arg_n = np.clip((phi - phi_n) / vth, -max_arg, max_arg)
        arg_p = np.clip((phi_p - phi) / vth, -max_arg, max_arg)
        return self._ni * np.exp(arg_n), self._ni * np.exp(arg_p)

    def _charge(self, phi: np.ndarray,
                phi_n: float, phi_p: float,
                rho_ext: np.ndarray | None = None) -> np.ndarray:
        n, p = self._carrier_densities(phi, phi_n, phi_p)
        net = self.doping.net_doping(self.grid.x)
        rho = self._q_val * (p - n + net)
        if rho_ext is not None:
            rho = rho + rho_ext
        return rho

    def _drho_dphi(self, phi: np.ndarray,
                   phi_n: float, phi_p: float) -> np.ndarray:
        vth = VT(self.T)
        n, p = self._carrier_densities(phi, phi_n, phi_p)
        return -self._q_val / vth * (n + p)

    def _residual(self, phi: np.ndarray, Vbias: float,
                  phi_n: float, phi_p: float,
                  rho_ext: np.ndarray | None = None) -> np.ndarray:
        phi_0, phi_L = self._contact_bias(Vbias)
        dx2 = self.grid.dx ** 2
        rho = self._charge(phi, phi_n, phi_p, rho_ext)
        F = np.zeros_like(phi)
        ep = self._eps_half[1:]
        em = self._eps_half[:-1]
        F[1:-1] = (ep * (phi[2:] - phi[1:-1])
                   - em * (phi[1:-1] - phi[:-2])) / dx2 + rho[1:-1]
        F[0] = phi[0] - phi_0
        F[-1] = phi[-1] - phi_L
        return F

    def _jacobian(self, phi: np.ndarray, Vbias: float,
                  phi_n: float, phi_p: float
                  ) -> Tuple[np.ndarray, ...]:
        dx2 = self.grid.dx ** 2
        drho = self._drho_dphi(phi, phi_n, phi_p)
        N = self.grid.no_of_nodes
        dl, d, du = np.zeros(N), np.zeros(N), np.zeros(N)
        ep = self._eps_half[1:]
        em = self._eps_half[:-1]
        dl[1:-1] = em / dx2
        d[1:-1] = -(ep + em) / dx2 + drho[1:-1]
        du[1:-1] = ep / dx2
        d[0] = d[-1] = 1.0
        return dl, d, du

    @staticmethod
    def _tridiagonal_solve(dl: np.ndarray, d: np.ndarray,
                           du: np.ndarray, rhs: np.ndarray
                           ) -> np.ndarray:
        N = len(d)
        ab = np.zeros((3, N))
        ab[1, :] = d
        ab[0, 1:] = du[:-1]
        ab[2, :-1] = dl[1:]
        return solve_banded((1, 1), ab, rhs)

    def _depletion_guess(self, x: np.ndarray,
                         phi_0: float, phi_L: float,
                         x_center: float) -> np.ndarray:
        net = self.doping.net_doping(x)
        idx_j = int(np.searchsorted(x, x_center))
        n_doping = float(np.max(np.abs(net[max(0, idx_j):min(len(x), idx_j + 100)])))
        Vbi = max(phi_L - phi_0, 0.01)
        if n_doping > 1e10:
            W_dep = float(np.sqrt(2.0 * self._eps[idx_j] * Vbi
                                  / (self._q_val * n_doping)))
        else:
            W_dep = self.grid.L / 4.0
        W_dep = float(np.clip(W_dep, self.grid.L / 200.0, self.grid.L / 2.0))
        sig_len = max(W_dep / 3.0, self.grid.dx * 2.0)
        sig = 0.5 * (1.0 + np.tanh((x - x_center) / sig_len))
        return phi_0 + (phi_L - phi_0) * sig

    def _step_guess(self, Vbias: float,
                    x_center: float | None = None) -> np.ndarray:
        L, x = self.grid.L, self.grid.x
        if x_center is None:
            x_center = self._find_junction()
        phi_0, phi_L = self._contact_bias(Vbias)
        Vbi = phi_L - phi_0
        if Vbias < Vbi:
            return self._depletion_guess(x, phi_0, phi_L, x_center)
        sig_len = max(L / 8.0, 1.0e-5)
        sig = 0.5 * (1.0 + np.tanh((x - x_center) / sig_len))
        return phi_0 + (phi_L - phi_0) * sig

    def _find_junction(self) -> float:
        x = self.grid.x
        net = self.doping.net_doping(x)
        signs = np.where(net >= 0, 1.0, -1.0)
        for i in range(len(x) - 1):
            if signs[i] < 0 and signs[i + 1] >= 0:
                frac = -net[i] / (net[i + 1] - net[i] + 1e-30)
                return float(x[i] + frac * (x[i + 1] - x[i]))
        return self.grid.L / 2

    def solve(self, Vbias: float, phi_n: float | None = None,
              phi_p: float = 0.0,
              guess: np.ndarray | None = None,
              rho_ext: np.ndarray | None = None
              ) -> Tuple[np.ndarray, dict]:
        if guess is None:
            phi = self._step_guess(Vbias)
        else:
            phi = guess.value.copy() if hasattr(guess, 'value') else guess.copy()

        if phi_n is None:
            phi_n = Vbias

        V0, VL = self._contact_bias(Vbias)

        dx2 = self.grid.dx ** 2
        vth = VT(self.T)

        for it in range(self.max_iter):
            phi, norm, nls, _step = newton_iteration(
                phi, Vbias, phi_n, phi_p, V0, VL,
                dx2, self._eps_half, self._ni, vth,
                self.doping.net_doping(self.grid.x),
                self.max_dphi, self._q_val,
            )

            if norm < self.tol:
                log.info("Poisson converged  V=%.2f  %d iters  residual=%.2e",
                         Vbias, it, norm)
                return phi, {"converged": True, "iterations": it, "residual_norm": norm}

            if not np.isfinite(norm):
                raise PhysicsError(
                    f"Non-finite Poisson residual ({norm:.2e}) at iteration {it}, V={Vbias:.2f}")

        raise ConvergenceError(
            f"Poisson did not converge at V={Vbias:.2f} V "
            f"(residual {norm:.2e}, {self.max_iter} iterations)")
