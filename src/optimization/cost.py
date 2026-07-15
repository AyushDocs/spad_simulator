from __future__ import annotations

from copy import deepcopy

import numpy as np

from ..core.layer import Layer
from ..utils._logging import get_logger

log = get_logger("optimization.cost")


def PDE(BrP: np.ndarray) -> float:
    """Photon Detection Efficiency (proportional to mean BrP)."""
    return float(np.mean(BrP))


def DCR(BrP: np.ndarray, dark_gen_rate: float = 1e3,
        area: float = 1e-6) -> float:
    """Dark Count Rate (simplified)."""
    return dark_gen_rate * area * float(np.mean(BrP))


class CostFunction:
    """Multi-objective SPAD cost.

    J = w1*PDE − w2*DCR − w3*jitter − w4*|BV − BV_target|

    Parameters
    ----------
    weights : dict, optional
        Keys: ``w_pde``, ``w_dcr``, ``w_jitter``, ``w_bv``.
    BV_target : float
        Target breakdown voltage (V).
    layer_doping_keys : dict[int, str], optional
        Mapping from layer index to the key the PSO vector provides.
        Example: ``{2: "charge_doping", 4: "absorber_doping"}``.
    """

    def __init__(self, weights: dict | None = None,
                 BV_target: float = 20.0,
                 layer_doping_keys: dict[int, str] | None = None) -> None:
        self.w = weights or dict(w_pde=1.0, w_dcr=0.5,
                                 w_jitter=0.3, w_bv=1.0)
        self.BV_target = BV_target
        self.layer_doping_keys = layer_doping_keys or {}

    def evaluate(self, simulator, doping_params: dict) -> tuple[float, dict]:
        """Evaluate cost for the given doping parameters.

        Parameters
        ----------
        simulator : SPADSimulator
        doping_params : dict
            Keys are layer index strings (e.g. ``"2"``), values are
            doping densities (cm⁻³).  Converted to integers internally.

        Returns
        -------
        (J, info) where J is the scalar cost and info is a diagnostics dict.
        """
        # Build modified layers with updated doping
        layers = [deepcopy(lyr) for lyr in simulator.device.layers]
        for layer_idx_str, doping_val in doping_params.items():
            idx = int(layer_idx_str)
            if 0 <= idx < len(layers):
                object.__setattr__(layers[idx], "doping_A", float(doping_val))
        simulator.set_layers(layers)

        # Find breakdown and compute trigger probability
        Vbr, _ = simulator.find_breakdown(V_start=10, V_max=150, V_step=0.5)
        if Vbr is None:
            return -1e6, dict(PDE=0.0, DCR=0.0, BV=None, J=-1e6)

        Vex = max(1.0, self.BV_target * 0.05)
        Pe, Ph, E = simulator.solve_trigger(Vbr + Vex)
        Ptr = Pe + Ph - Pe * Ph

        # Absorption-weighted average over multiplication region
        x = simulator.grid.x
        x_mult_max = 4e-4
        mult_mask = (np.abs(E) > 1e4) & (x < x_mult_max)
        if np.any(mult_mask):
            alpha_opt = simulator.materials["InGaAs"].absorption_coefficient(1550e-9)
            w = alpha_opt * np.exp(-alpha_opt * x[mult_mask])
            w_sum = float(np.sum(w))
            ptr_mean = float(np.sum(Ptr[mult_mask] * w) / w_sum) if w_sum > 0 else 0.0
        else:
            ptr_mean = 0.0

        pde = PDE(np.array([ptr_mean]))
        dcr = DCR(np.array([ptr_mean]))
        bv_err = abs(Vbr - self.BV_target)

        J = (self.w["w_pde"] * pde
             - self.w["w_dcr"] * dcr
             - self.w["w_jitter"] * 0.0
             - self.w["w_bv"] * bv_err)

        return J, dict(PDE=pde, DCR=dcr, BV=Vbr, J=J)
