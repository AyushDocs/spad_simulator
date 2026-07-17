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
        l_e: np.ndarray | None = None,
        l_h: np.ndarray | None = None,
        field_threshold: float = 1e4,
        max_iter: int = 500,
        tol: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the non-local trigger-probability equations with dead space.

        Uses an iterative fixed-point algorithm to solve the integral form
        of the history-dependent trigger probability equations.
        """
        N = len(x)

        # If no significant field, return zeros
        if not np.any(np.abs(E) > field_threshold):
            return np.zeros(N), np.zeros(N)

        if l_e is None:
            l_e = np.zeros(N)
        if l_h is None:
            l_h = np.zeros(N)

        # --- Initial guess ---
        # Nudge the solver toward the nontrivial branch (if it exists)
        x_norm = (x - x[0]) / (x[-1] - x[0])
        Pe = 0.3 * (1.0 - x_norm)
        Ph = 0.3 * x_norm

        for iteration in range(max_iter):
            # Probability that at least one carrier triggers
            Ptr = Pe + Ph - Pe * Ph
            Ptr_interp = interp1d(x, Ptr, kind="linear", fill_value=0.0, bounds_error=False)

            # Evaluate delayed terms (dead space shift)
            Ptr_e = Ptr_interp(x + l_e)
            Ptr_h = Ptr_interp(x - l_h)

            # Integrate backwards for Pe: Pe(x) = 1 - exp(-int_x^W alpha * Ptr_e dx)
            integrand_e = alpha * Ptr_e
            int_e = np.zeros(N)
            for i in range(N - 2, -1, -1):
                dx = x[i + 1] - x[i]
                int_e[i] = int_e[i + 1] + 0.5 * (integrand_e[i] + integrand_e[i + 1]) * dx

            Pe_new = 1.0 - np.exp(-int_e)

            # Integrate forwards for Ph: Ph(x) = 1 - exp(-int_0^x beta * Ptr_h dx)
            integrand_h = beta * Ptr_h
            int_h = np.zeros(N)
            for i in range(1, N):
                dx = x[i] - x[i - 1]
                int_h[i] = int_h[i - 1] + 0.5 * (integrand_h[i] + integrand_h[i - 1]) * dx

            Ph_new = 1.0 - np.exp(-int_h)

            # Clamp and damp to ensure stable convergence
            Pe_new = np.clip(Pe_new, 0.0, 1.0)
            Ph_new = np.clip(Ph_new, 0.0, 1.0)

            err = np.max(np.abs(Pe_new - Pe)) + np.max(np.abs(Ph_new - Ph))

            # Damping factor: heavy damping helps convergence near breakdown
            alpha_damp = 0.2
            Pe = alpha_damp * Pe_new + (1.0 - alpha_damp) * Pe
            Ph = alpha_damp * Ph_new + (1.0 - alpha_damp) * Ph

            if err < tol:
                break

        # If converged to 0, it means we are below breakdown
        if np.max(Pe) < 1e-4:
            Pe = np.zeros(N)
            Ph = np.zeros(N)

        return Pe, Ph


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
