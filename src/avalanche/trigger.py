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

        ab_diff = alpha - beta
        avg_diff = (ab_diff[:-1] + ab_diff[1:]) / 2.0
        exponents = avg_diff * dx
        cum_exponents = np.concatenate([[0.0], np.cumsum(exponents[::-1])[::-1]])
        mu = np.exp(cum_exponents)

        avg_bm = (beta[:-1] * mu[:-1] + beta[1:] * mu[1:]) / 2.0
        ibm = np.concatenate([[0.0], np.cumsum(avg_bm[::-1])[::-1]]) * dx

        Mn = mu / np.clip(1.0 - ibm, 1e-300, None)
        Pe = np.clip(1.0 - 1.0 / Mn, 0.0, 1.0)

        ba_diff = beta - alpha
        avg_ba = (ba_diff[:-1] + ba_diff[1:]) / 2.0
        exponents_h = avg_ba * dx
        cum_exponents_h = np.concatenate([[0.0], np.cumsum(exponents_h)])
        mu_h = np.exp(cum_exponents_h)

        avg_am = (alpha[:-1] * mu_h[:-1] + alpha[1:] * mu_h[1:]) / 2.0
        iam = np.concatenate([[0.0], np.cumsum(avg_am) * dx])

        Mp = mu_h / np.clip(1.0 - iam, 1e-300, None)
        Ph = np.clip(1.0 - 1.0 / Mp, 0.0, 1.0)

        Pe[np.abs(E) < field_threshold] = 0.0
        Ph[np.abs(E) < field_threshold] = 0.0

        return Pe, Ph
