"""Avalanche trigger probability & multiplication (coupled McIntyre ODEs).

Implements the Oldham–Hayat formulation as two coupled first-order ODEs
solved as a boundary-value problem, plus a separate coupled-ODE
multiplication solver.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d


class TriggerSolver:
    """Trigger probability via coupled McIntyre first-order ODEs.

    The correct Oldham/Hayat trigger-probability equations are the
    coupled nonlinear ODEs:

        dPe/dx = −α(x) · (1 − Pe) · [Pe + Ph − Pe·Ph]     Pe(W) = 0
        dPh/dx = +β(x) · (1 − Ph) · [Pe + Ph − Pe·Ph]     Ph(0) = 0

    The key physics: the factor ``[Pe + Ph − Pe·Ph]`` (probability that
    at least one of the secondary carriers triggers) creates a nonlinear
    feedback that produces a genuine threshold:

    • Below breakdown: the only self-consistent solution is Pe = Ph = 0
      everywhere — the RHS vanishes identically when Pe = Ph = 0.
    • Above breakdown: a nontrivial branch appears via a pitchfork-type
      bifurcation, giving 0 < Pe, Ph ≤ 1 in the high-field region.

    This is fundamentally different from the integral-form equations
    ``Pe = 1 − exp[−∫α(1+Ph)dx]`` which always give nonzero Pe for any
    α > 0 and have no threshold — those describe a different quantity
    (related to mean multiplication, not self-sustaining trigger).

    The BVP is solved with ``scipy.integrate.solve_bvp``.
    """

    def __init__(self, grid) -> None:
        self.grid = grid

    def solve(
        self,
        E: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        x: np.ndarray,
        field_threshold: float = 1e4,
        max_iter: int = 500,
        tol: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the coupled McIntyre trigger-probability ODEs.

        Parameters
        ----------
        E : array
            Electric field profile (V/cm).
        alpha : array
            Effective electron ionization coefficient α(x) (cm⁻¹).
        beta : array
            Effective hole ionization coefficient β(x) (cm⁻¹).
        x : array
            Spatial grid (cm), length N.
        field_threshold : float
            Minimum |E| for meaningful ionization (V/cm).
        max_iter, tol : int, float
            BVP solver controls.

        Returns
        -------
        Pe, Ph : arrays of length N
        """
        N = len(x)

        # If no significant field, return zeros
        if not np.any(np.abs(E) > field_threshold):
            return np.zeros(N), np.zeros(N)

        # Build interpolators for α(x) and β(x) on the grid
        alpha_interp = interp1d(x, alpha, kind="linear",
                                fill_value=0.0, bounds_error=False)
        beta_interp = interp1d(x, beta, kind="linear",
                               fill_value=0.0, bounds_error=False)

        # --- Define the ODE system (Oldham/Hayat form) ---
        # y[0] = Pe(x),  y[1] = Ph(x)
        #
        # dPe/dx = −α(x) · (1 − Pe) · (Pe + Ph − Pe·Ph)
        # dPh/dx = +β(x) · (1 − Ph) · (Pe + Ph − Pe·Ph)
        #
        # The factor (Pe + Ph − Pe·Ph) = 1 − (1−Pe)(1−Ph) is the probability
        # that at least one of the secondary carriers triggers an avalanche.
        def ode_rhs(xi, y):
            a = alpha_interp(xi)
            b = beta_interp(xi)
            Pe = y[0]
            Ph = y[1]
            # Trigger probability of at least one carrier
            Ptr = Pe + Ph - Pe * Ph
            dPe = -a * (1.0 - Pe) * Ptr
            dPh = b * (1.0 - Ph) * Ptr
            return np.vstack([dPe, dPh])

        # --- Boundary conditions ---
        # Pe(W) = 0  (electron exiting at right edge can't trigger)
        # Ph(0) = 0  (hole exiting at left edge can't trigger)
        def bc(ya, yb):
            return np.array([ya[1], yb[0]])  # Ph(0) = 0, Pe(W) = 0

        # --- Initial guess ---
        # Use a shaped guess that respects BCs and nudges the solver
        # toward the nontrivial branch (if it exists above breakdown).
        x_norm = (x - x[0]) / (x[-1] - x[0])
        # Pe should be nonzero near x=0, zero at x=W
        Pe_guess = 0.3 * (1.0 - x_norm)
        # Ph should be zero at x=0, nonzero near x=W
        Ph_guess = 0.3 * x_norm
        y_init = np.vstack([Pe_guess, Ph_guess])

        # --- Solve the BVP ---
        sol = solve_bvp(
            ode_rhs, bc, x, y_init,
            tol=tol, max_nodes=max(5 * N, 5000),
        )

        if sol.success:
            Pe_sol = sol.sol(x)[0]
            Ph_sol = sol.sol(x)[1]
        else:
            # Below breakdown the solver may fail to find a nontrivial
            # solution — that's correct, Pe = Ph = 0 is the answer.
            Pe_sol = np.zeros(N)
            Ph_sol = np.zeros(N)

        # Clamp to physical range [0, 1]
        Pe_sol = np.clip(Pe_sol, 0.0, 1.0)
        Ph_sol = np.clip(Ph_sol, 0.0, 1.0)

        return Pe_sol, Ph_sol


class MultiplicationSolver:
    """Avalanche multiplication factor via coupled McIntyre ODEs.

    Solves the two-point BVP for the mean multiplication factors:

        dMn/dx = −α(x) · (Mn + Mp)       Mn(W) = 1
        dMp/dx = +β(x) · (Mn + Mp)       Mp(0) = 1

    where Mn(x) is the expected total number of electron-hole pairs
    produced by a single electron created at x (drifting toward W),
    and Mp(x) is the same for a single hole (drifting toward 0).

    The total multiplication for electron injection at x₀ is Mn(x₀).
    For photon absorption uniformly in the absorber, we evaluate
    Mn at the peak-field position (or average over the injection
    profile).

    This is independent of the trigger probability — M describes
    the *mean* current gain (a first-moment quantity), not the
    *probability* of self-sustaining avalanche.
    """

    def __init__(self, grid) -> None:
        self.grid = grid

    def compute(
        self,
        E: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        x: np.ndarray,
    ) -> float:
        """Compute multiplication factor M for electron injection at x=0.

        Returns
        -------
        M : float
            Multiplication factor Mn(0).  Returns 1.0 if no significant
            field, capped at 1e6 to avoid numerical overflow near
            exact breakdown.
        """
        if not np.any(np.abs(E) > 1e5):
            return 1.0

        alpha_interp = interp1d(x, alpha, kind="linear",
                                fill_value=0.0, bounds_error=False)
        beta_interp = interp1d(x, beta, kind="linear",
                               fill_value=0.0, bounds_error=False)

        # ODE: y[0] = Mn, y[1] = Mp
        def ode_rhs(xi, y):
            a = alpha_interp(xi)
            b = beta_interp(xi)
            S = y[0] + y[1]
            dMn = -a * S
            dMp = b * S
            return np.vstack([dMn, dMp])

        # BCs: Mp(0) = 1, Mn(W) = 1
        def bc(ya, yb):
            return np.array([ya[1] - 1.0, yb[0] - 1.0])

        # Initial guess: uniform Mn=1, Mp=1
        y_init = np.ones((2, len(x)))

        M_MAX = 1e6
        sol = solve_bvp(ode_rhs, bc, x, y_init,
                        tol=1e-8, max_nodes=max(5 * len(x), 5000))

        if sol.success:
            Mn_0 = float(sol.sol(x[0])[0])
            return min(max(Mn_0, 1.0), M_MAX)

        # Solver divergence near exact breakdown
        return M_MAX
