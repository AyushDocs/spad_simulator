"""Breakdown voltage detection via multiplication-based current rise."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.grid import Grid1D
from ..utils._logging import get_logger

log = get_logger("avalanche.breakdown")


class BreakdownCriterion(ABC):
    """Abstract criterion for detecting breakdown at a given bias."""

    @abstractmethod
    def check(self, V: float, phi: np.ndarray, E: np.ndarray) -> tuple[bool, dict]:
        ...


class CurrentRiseCriterion(BreakdownCriterion):
    """Detect breakdown when current dI/dV exceeds a slope threshold.

    Uses a moving-window current computation to detect the sharp rise
    characteristic of avalanche multiplication.
    """

    def __init__(self, compute_current, V_step: float,
                 slope_threshold: float = 2.0, window: int = 3) -> None:
        self.compute_current = compute_current
        self.V_step = V_step
        self.slope_threshold = slope_threshold
        self.window = window
        self._prev_currents: list[float] = []

    def check(self, V: float, phi: np.ndarray, E: np.ndarray) -> tuple[bool, dict]:
        I = self.compute_current(V, phi, E)
        self._prev_currents.append(I)

        info = {"V": V, "I": I}

        if len(self._prev_currents) < self.window + 1:
            return False, info

        recent = self._prev_currents[-(self.window + 1):]
        dI = recent[-1] - recent[0]
        dV = self.V_step * self.window
        slope = dI / (dV * abs(recent[0]) + 1e-30)

        info["slope"] = slope
        return slope > self.slope_threshold, info


class BreakdownVoltage:
    """Sweep voltage to find breakdown using a BreakdownCriterion.

    Iterates from V_start upward, solving Poisson at each step,
    until the criterion triggers.
    """

    def __init__(self, poisson_solver, grid: Grid1D,
                 criterion: BreakdownCriterion, V_step: float = 0.5) -> None:
        self.poisson = poisson_solver
        self.grid = grid
        self.criterion = criterion
        self.V_step = V_step

    def find(self, V_start: float = 0.0,
             V_max: float = 150.0) -> tuple[float | None, list[dict]]:
        results: list[dict] = []
        V = V_start

        while V <= V_max:
            try:
                phi, info_solver = self.poisson.solve(V)
                E = self.grid.gradient(phi)
                triggered, info = self.criterion.check(V, phi, E)
                info["converged"] = True
                results.append(info)

                if triggered:
                    log.info(f"Breakdown criterion triggered at V = {V:.2f} V")
                    return V, results

                V += self.V_step

            except Exception as e:
                results.append({"V": V, "converged": False, "error": str(e)})
                log.debug(f"Poisson failed at V = {V:.2f}: {e}")
                V += self.V_step

        return None, results
