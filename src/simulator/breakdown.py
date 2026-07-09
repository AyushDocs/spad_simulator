"""Breakdown voltage detection."""
from __future__ import annotations

import numpy as np

from ..core.physics_helpers import combined_trigger_probability, avalanche_gain
from ..avalanche.breakdown import TriggerCriterion, CurrentCriterion, BreakdownVoltage
from ..utils._logging import get_logger
from ..utils._exceptions import PhysicsError

log = get_logger("simulator")


def find_breakdown(
    poisson_solver,
    grid,
    ionization,
    trigger,
    dark_current,
    device,
    detector_area: float,
    _Vbr: float | None,
    V_start: float = 0.0,
    V_max: float = 100.0,
    V_step: float = 0.1,
    force: bool = False,
    criterion: str = "current",
    I_threshold: float = 1e-6,
) -> tuple[float | None, list[dict]]:
    """Run breakdown search. Returns (Vbr, sweep_results)."""
    if _Vbr is not None and not force:
        return _Vbr, []

    crit: CurrentCriterion | TriggerCriterion
    if criterion == "current":
        def _compute_current(V: float, phi: np.ndarray, E: np.ndarray) -> float:
            alpha = ionization.alpha(E)
            beta = ionization.beta(E)
            Pe, Ph = trigger.solve(E, alpha, beta, grid.x)
            Ptr = combined_trigger_probability(Pe, Ph)
            M = avalanche_gain(float(np.max(Ptr)))
            J_total = dark_current.total_dark_current_density(
                grid.x, E, device.material.ni, device.material.Eg,
                device.material.mc, device.material.mh,
            )
            I_primary = float(np.trapezoid(J_total, grid.x) * detector_area)
            return I_primary * M

        crit = CurrentCriterion(_compute_current, I_threshold=I_threshold)
    else:
        crit = TriggerCriterion(ionization, trigger, grid)

    bv = BreakdownVoltage(poisson_solver, grid, crit, V_step)
    Vbr, results = bv.find(V_start, V_max)
    if Vbr is None:
        raise PhysicsError("No breakdown detected")
    log.info(f"Vbr = {Vbr:.1f} V")
    return Vbr, results
