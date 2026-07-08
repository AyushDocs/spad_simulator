from __future__ import annotations

import numpy as np

from ..core.grid import Grid1D


class TriggerSolver:
    """
    Trigger probability via multiplication-factor integrals
    (McIntyre 1966, 1973).

    For pure electron injection:
        mu(x) = exp(int_x^W (alpha - beta) ds)
        M(x) = mu(x) / (1 - int_x^W beta(s) mu(s) ds)
        P_e(x) = 1 - 1/M(x)

    For pure hole injection:
        mu_h(x) = exp(int_x^W (beta - alpha) ds)
        M_p(x) = mu_h(x) / (1 - int_x^W alpha(s) mu_h(s) ds)
        P_h(x) = 1 - 1/M_p(x)
    """

    def __init__(self, grid: Grid1D) -> None:
        self.grid = grid

    def solve(self, E: np.ndarray, alpha: np.ndarray,
              beta: np.ndarray, x: np.ndarray,
              field_threshold: float = 1e4
              ) -> tuple[np.ndarray, np.ndarray]:
        dx = self.grid.dx
        N = len(x)

        mu = np.ones(N)
        for i in range(N - 2, -1, -1):
            integ = (alpha[i] - beta[i] + alpha[i + 1] - beta[i + 1]) / 2.0
            mu[i] = mu[i + 1] * np.exp(integ * dx)

        cum = 0.0
        ibm = np.zeros(N)
        for i in range(N - 2, -1, -1):
            avg = (beta[i] * mu[i] + beta[i + 1] * mu[i + 1]) / 2.0
            cum += avg * dx
            ibm[i] = cum

        Mn = mu / np.clip(1.0 - ibm, 1e-300, None)
        Pe = np.clip(1.0 - 1.0 / Mn, 0.0, 1.0)

        mu_h = np.ones(N)
        for i in range(1, N):
            integ = (beta[i - 1] - alpha[i - 1] + beta[i] - alpha[i]) / 2.0
            mu_h[i] = mu_h[i - 1] * np.exp(integ * dx)

        cum = 0.0
        iam = np.zeros(N)
        for i in range(1, N):
            avg = (alpha[i - 1] * mu_h[i - 1] + alpha[i] * mu_h[i]) / 2.0
            cum += avg * dx
            iam[i] = cum

        Mp = mu_h / np.clip(1.0 - iam, 1e-300, None)
        Ph = np.clip(1.0 - 1.0 / Mp, 0.0, 1.0)

        Pe[np.abs(E) < field_threshold] = 0.0
        Ph[np.abs(E) < field_threshold] = 0.0

        return Pe, Ph
