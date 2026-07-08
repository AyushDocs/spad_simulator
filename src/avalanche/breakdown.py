from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from ..core.grid import Grid1D
from .ionization import IonizationCoefficients
from .trigger import TriggerSolver
from ..utils._logging import get_logger

log = get_logger("breakdown")


class BreakdownCriterion(ABC):
    """Strategy for detecting breakdown at a given bias."""

    @abstractmethod
    def check(self, V: float, phi: np.ndarray, E: np.ndarray) -> tuple[bool, dict]:
        ...


class TriggerCriterion(BreakdownCriterion):
    """Detect breakdown when max trigger probability exceeds a threshold."""

    def __init__(self, ionization: IonizationCoefficients,
                 trigger: TriggerSolver,
                 grid: Grid1D,
                 threshold: float = 0.99) -> None:
        self.ion = ionization
        self.trigger = trigger
        self.grid = grid
        self.threshold = threshold

    def check(self, V: float, phi: np.ndarray,
              E: np.ndarray) -> tuple[bool, dict]:
        alpha = self.ion.alpha(E)
        beta = self.ion.beta(E)
        Pe, Ph = self.trigger.solve(E, alpha, beta, self.grid.x)
        BrP_max = float(np.max(Pe))
        return BrP_max > self.threshold, {"V": V, "BrP_max": BrP_max}


class CurrentCriterion(BreakdownCriterion):
    """Detect breakdown when total current exceeds a threshold."""

    def __init__(self, compute_current, I_threshold: float = 1e-6) -> None:
        self._fn = compute_current
        self._I_th = I_threshold

    def check(self, V: float, phi: np.ndarray,
              E: np.ndarray) -> tuple[bool, dict]:
        I_total = self._fn(V, phi, E)
        return I_total > self._I_th, {"V": V, "I_total": I_total}


class BreakdownVoltage:
    """
    Sweep bias, solve Poisson -> E, then detect breakdown using a
    pluggable ``BreakdownCriterion`` strategy.
    """

    def __init__(self, poisson_solver, grid: Grid1D,
                 criterion: BreakdownCriterion,
                 V_step: float = 0.1) -> None:
        self.poisson = poisson_solver
        self.grid = grid
        self.criterion = criterion
        self.V_step = V_step

    def find(self, V_start: float = 0.0, V_max: float = 100.0,
             phi_n: float | None = None, phi_p: float = 0.0
             ) -> Tuple[float | None, List[dict]]:
        V = V_start
        results: List[dict] = []
        Vbr: float | None = None
        prev_phi = None

        while V <= V_max:
            phi, info = self.poisson.solve(V, phi_n, phi_p, guess=prev_phi)
            prev_phi = phi
            E = self.grid.gradient(phi)

            detected, rec = self.criterion.check(V, phi, E)
            rec["converged"] = info["converged"]
            results.append(rec)
            if detected and Vbr is None:
                Vbr = V
                log.info("Breakdown at V = %.2f V", V)

            V += self.V_step

        return Vbr, results
