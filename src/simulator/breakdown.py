"""Breakdown voltage detection."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..avalanche.breakdown import TriggerCriterion, CurrentCriterion
from ..utils._logging import get_logger

log = get_logger("simulator")


def find_breakdown(
    poisson_service,
    ionization,
    trigger,
    dark_current,
    grid,
    device,
    detector_area: float,
    _Vbr: float | None,
    V_start: float = 0.0,
    V_max: float = 100.0,
    V_step: float = 0.1,
    force: bool = False,
    criterion: str = "current",
    I_threshold: float = 1e-6,
) -> Tuple[float | None, List[dict]]:
    """Run breakdown search. Returns (Vbr, sweep_results)."""
    if _Vbr is not None and not force:
        return _Vbr, []

    if criterion == "current":
        def _compute_current(V: float, phi: np.ndarray, E: np.ndarray) -> float:
            alpha = ionization.alpha(E)
            beta = ionization.beta(E)
            Pe, Ph = trigger.solve(E, alpha, beta, grid.x)
            Ptr = Pe + Ph - Pe * Ph
            Ptr_max = float(np.max(Ptr))
            M = min(1.0 / (1.0 - Ptr_max + 1e-15), 10000.0)
            J_total = dark_current.total_dark_current_density(
                grid.x, E, device.material.ni, device.material.Eg,
                device.material.mc, device.material.mh,
            )
            I_primary = float(np.trapezoid(J_total, grid.x) * detector_area)
            return I_primary * M

        crit = CurrentCriterion(_compute_current, I_threshold=I_threshold)
    else:
        crit = TriggerCriterion(ionization, trigger, grid)

    Vbr, results = poisson_service.find_breakdown(V_start, V_max, crit, V_step)
    if Vbr is None:
        raise ValueError("No breakdown detected")
    log.info(f"Vbr = {Vbr:.1f} V")
    return Vbr, results
