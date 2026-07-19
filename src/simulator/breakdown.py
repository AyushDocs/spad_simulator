"""Breakdown voltage detection — physics-based current rise criterion."""
from __future__ import annotations

import numpy as np

from ..core.grid import Grid1D
from ..avalanche.breakdown import (
    BreakdownCriterion,
)
from ..utils._logging import get_logger

log = get_logger("simulator")

# Dead space parameters for InP multiplication layer
_DEAD_SPACE_EG = 1.35  # eV, InP bandgap


class MultiplicationCurrentRise(BreakdownCriterion):
    """Detect breakdown when multiplied dark current rises sharply.

    At each bias:
        1. Compute effective ionization coefficients alpha(E), beta(E)
           with dead space correction
        2. Compute multiplication factor M via McIntyre's formula
        3. Compute total current = J_dark * M * area
        4. Breakdown when current surges > 5× in one bias step

    Dead space reduces the effective ionization coefficients:
        alpha_eff = alpha / (1 + alpha * ld)
    where ld = E_th / F is the distance a carrier must travel
    to gain threshold energy E_th.
    """

    def __init__(self, ionization, grid, dark_current_fn,
                 detector_area: float, V_step: float,
                 Eg: float = _DEAD_SPACE_EG) -> None:
        self.ion = ionization
        self.grid = grid
        self.dark_current_fn = dark_current_fn
        self.detector_area = detector_area
        self.V_step = V_step
        self.Eg = Eg
        self._I_prev: float | None = None

    def _compute_multiplication(self, E: np.ndarray) -> float:
        x = self.grid.x
        alpha = self.ion.effective_alpha_n(np.abs(E), Eg=self.Eg)
        beta = self.ion.effective_alpha_p(np.abs(E), Eg=self.Eg)

        active = np.abs(E) > 1e5
        if not np.any(active):
            return 1.0

        dx = np.diff(x)
        diff = alpha - beta
        cum = np.zeros_like(x)
        for i in range(len(x) - 2, -1, -1):
            cum[i] = cum[i + 1] + diff[i] * dx[i]

        integrand = beta * np.exp(cum)
        denom = 1.0 - float(np.trapezoid(integrand, x))

        if denom <= 0.01:
            return 1e6
        return min(1.0 / denom, 1e6)

    def check(self, V: float, phi: np.ndarray, E: np.ndarray) -> tuple[bool, dict]:
        M = self._compute_multiplication(E)
        J_dark = self.dark_current_fn(E)
        I_total = float(np.trapezoid(J_dark, self.grid.x)) * self.detector_area * M

        triggered = False
        if self._I_prev is not None and self._I_prev > 0:
            if I_total / self._I_prev > 5.0:
                triggered = True
        self._I_prev = I_total

        return triggered, {"V": V, "M": M, "I_total": I_total}


def find_breakdown(
    poisson_solver,
    grid: Grid1D,
    ionization,
    _Vbr: float | None,
    V_start: float = 0.0,
    V_max: float = 100.0,
    V_step: float = 0.5,
    force: bool = False,
    **kwargs,
) -> tuple[float | None, list[dict]]:
    if _Vbr is not None and not force:
        return _Vbr, []

    def dark_current_fn(E: np.ndarray) -> np.ndarray:
        ni = 1e11
        tau = 1e-9
        return np.full_like(E, 1.602e-19 * ni / (2.0 * tau))

    crit = MultiplicationCurrentRise(
        ionization, grid, dark_current_fn,
        detector_area=1e-6, V_step=V_step,
        Eg=_DEAD_SPACE_EG,
    )

    Vbr: float | None = None
    results: list[dict] = []
    V = V_start
    log.info("Breakdown sweep: V_start=%.1f V_max=%.1f V_step=%.2f",
             V_start, V_max, V_step)

    while V <= V_max:
        try:
            phi, info_solver = poisson_solver.solve(V)
            E = grid.gradient(phi)
            triggered, info = crit.check(V, phi, E)
            results.append(info)
            log.debug("  V=%.2f  M=%.1f  I_total=%.2e  triggered=%s",
                      V, info["M"], info["I_total"], triggered)

            if triggered:
                Vbr = V
                log.info("Breakdown detected at V = %.2f V  (I_total=%.2e, M=%.1f)",
                         Vbr, info["I_total"], info["M"])
                break
            V += V_step

        except Exception as e:
            log.warning("  V=%.2f failed: %s", V, e)
            results.append({"V": V, "converged": False, "error": str(e)})
            if Vbr is None:
                Vbr = V
            break

    if Vbr is not None:
        log.info("Breakdown at V = %.2f V  (current rise: >5x jump in I_total)", Vbr)
    else:
        log.warning("No breakdown detected in range")

    return Vbr, results
