"""Avalanche trigger probability (McIntyre model)."""

from __future__ import annotations

import numpy as np


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

    Dead space support:
        Pass effective alpha/beta (with dead space correction) as `alpha` and `beta`.
        Use IonizationCoefficients.effective_alpha_n/p() to compute them.
    """

    def __init__(self, grid) -> None:
        self.grid = grid

    def solve(self, E: np.ndarray, alpha: np.ndarray,
              beta: np.ndarray, x: np.ndarray,
              field_threshold: float = 1e4,
              ) -> tuple[np.ndarray, np.ndarray]:
        """McIntyre (1966) trigger probability for electrons (Pe) and holes (Ph).

        Boundary conditions:
          mu(x) = exp(∫_x^W (α-β) ds)  ->  mu[W] = exp(0) = 1  (rightmost = 0)
          ibm(x) = ∫_x^W β(s)·mu(s) ds  ->  ibm[W] = 0  (rightmost = 0)
          mu_h(x) = exp(∫_0^x (β-α) ds)  ->  mu_h[0] = exp(0) = 1  (leftmost = 0)
          iam(x) = ∫_0^x α(s)·mu_h(s) ds  ->  iam[0] = 0  (leftmost = 0)
        """
        dx = self.grid.dx
        N = len(x)

        # --- Electron trigger probability Pe(x) ---
        # mu(x) = exp(∫_x^W (α-β) ds): integrate from x rightward to W.
        # Boundary: mu(W) = 1  ->  append 0 at the RIGHT end.
        ab_diff = alpha - beta
        avg_diff = (ab_diff[:-1] + ab_diff[1:]) / 2.0
        exponents = avg_diff * dx  # length N-1
        # cumsum from right; append 0 at right so mu[-1] = exp(0) = 1
        cum_exponents = np.concatenate([np.cumsum(exponents[::-1])[::-1], [0.0]])
        mu = np.exp(cum_exponents)  # length N

        # ibm(x) = ∫_x^W β·mu ds: integrate from x rightward to W.
        # Boundary: ibm(W) = 0  ->  append 0 at the RIGHT end.
        avg_bm = (beta[:-1] * mu[:-1] + beta[1:] * mu[1:]) / 2.0  # length N-1
        ibm = np.concatenate([np.cumsum(avg_bm[::-1])[::-1], [0.0]]) * dx  # length N

        # Clamp denominator away from zero/negative to prevent unphysical divergence
        denom_e = np.clip(1.0 - ibm, 1e-10, None)
        Mn = mu / denom_e
        Pe = np.clip(1.0 - 1.0 / np.clip(Mn, 1.0, None), 0.0, 1.0)

        # --- Hole trigger probability Ph(x) ---
        # mu_h(x) = exp(∫_0^x (β-α) ds): integrate from 0 leftward to x.
        # Boundary: mu_h(0) = 1  ->  prepend 0 at the LEFT end.
        ba_diff = beta - alpha
        avg_ba = (ba_diff[:-1] + ba_diff[1:]) / 2.0
        exponents_h = avg_ba * dx  # length N-1
        # cumsum from left; prepend 0 at left so mu_h[0] = exp(0) = 1
        cum_exponents_h = np.concatenate([[0.0], np.cumsum(exponents_h)])
        mu_h = np.exp(cum_exponents_h)  # length N

        # iam(x) = ∫_0^x α·mu_h ds: integrate from 0 leftward to x.
        # Boundary: iam(0) = 0  ->  prepend 0 at the LEFT end.
        avg_am = (alpha[:-1] * mu_h[:-1] + alpha[1:] * mu_h[1:]) / 2.0  # length N-1
        iam = np.concatenate([[0.0], np.cumsum(avg_am)]) * dx  # length N

        denom_h = np.clip(1.0 - iam, 1e-10, None)
        Mp = mu_h / denom_h
        Ph = np.clip(1.0 - 1.0 / np.clip(Mp, 1.0, None), 0.0, 1.0)

        # Return the continuous physical trigger probability profiles
        return Pe, Ph
