from __future__ import annotations

import numpy as np


def PDE(BrP: np.ndarray) -> float:
    """Photon Detection Efficiency (proportional to mean BrP)."""
    return float(np.mean(BrP))


def DCR(BrP: np.ndarray, dark_gen_rate: float = 1e3,
        area: float = 1e-6) -> float:
    """Dark Count Rate (simplified)."""
    return dark_gen_rate * area * float(np.mean(BrP))


class CostFunction:
    """
    Multi-objective SPAD cost.

        J = w1*PDE - w2*DCR - w3*jitter - w4*|BV - BV0|
    """

    def __init__(self, weights: dict | None = None,
                 BV_target: float = 20.0) -> None:
        self.w = weights or dict(w_pde=1.0, w_dcr=0.5,
                                 w_jitter=0.3, w_bv=1.0)
        self.BV_target = BV_target

    def evaluate(self, simulator, doping_params: dict) -> tuple:
        simulator.set_doping(doping_params)
        Vbr, _ = simulator.find_breakdown()
        Pe, Ph, _ = simulator.solve_trigger(Vbr + 1.0 if Vbr else 20.0)
        brp = np.maximum(Pe, Ph)
        pde = PDE(brp)
        dcr = DCR(brp)
        bv_err = abs((Vbr or 0) - self.BV_target)
        J = (self.w["w_pde"] * pde - self.w["w_dcr"] * dcr
             - self.w["w_jitter"] * 0.0 - self.w["w_bv"] * bv_err)
        return J, dict(PDE=pde, DCR=dcr, BV=Vbr, J=J)
