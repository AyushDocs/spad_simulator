"""Validate McIntyre trigger probability: spatial profiles + absorption-weighted Ptr(Vex)."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ..utils.loaders import PlotConfig
from . import _config as _cfg
from ._config import PLOT_DIR

log = get_logger()


def run_trigger_back_calculate(sim: SPADSimulator, Vbr: float,
                               plot_cfg: PlotConfig | None = None) -> None:
    """Validate trigger probability by showing spatial profiles and Ptr(Vex).

    Two panels:
      1. Spatial Pe(x), Ph(x), Ptr(x) at Vex = 1, 3, 5 V
      2. Absorption-weighted Pe_mean, Ph_mean, Ptr_mean vs Vex
    """
    if plot_cfg and not plot_cfg.is_enabled("trigger_back_calculate"):
        return
    x = sim.grid.x * 1e4  # convert to µm
    alpha_opt = sim.materials["InGaAs"].absorption_coefficient(1550e-9)

    # --- Spatial profiles at a few Vex values ---
    vex_spatial = [1.0, 3.0, 5.0]
    Pe_spatial = []
    Ph_spatial = []
    Ptr_spatial = []
    E_spatial = []
    labels = []

    for Vex in vex_spatial:
        try:
            _, E, Pe, Ph, _, _ = sim.get_fields(Vbr + Vex)
            Ptr = Pe + Ph - Pe * Ph
            Pe_spatial.append(Pe)
            Ph_spatial.append(Ph)
            Ptr_spatial.append(Ptr)
            E_spatial.append(np.abs(E))
            labels.append(f"Vex={Vex:.0f}V")
            log.info("  Spatial Vex=%.0fV: Pe_max=%.4f, Ph_max=%.4f, Ptr_max=%.4f",
                     Vex, float(Pe.max()), float(Ph.max()), float(Ptr.max()))
        except Exception as e:
            log.info("  Spatial Vex=%.0fV failed: %s", Vex, e)

    # --- Absorption-weighted Ptr vs Vex ---
    vex_arr = np.linspace(-10, 10, 41)
    Pe_mean = np.full(len(vex_arr), np.nan)
    Ph_mean = np.full(len(vex_arr), np.nan)
    Ptr_mean = np.full(len(vex_arr), np.nan)

    for j, Vex in enumerate(vex_arr):
        Vbias = Vbr + Vex
        if Vbias <= 0:
            continue
        try:
            _, E, Pe, Ph, _, _ = sim.get_fields(Vbias)
            Ptr = Pe + Ph - Pe * Ph
            mult_mask = (np.abs(E) > _cfg.FIELD_THRESHOLD) & (sim.grid.x < _cfg.X_MULT_MAX)
            if np.any(mult_mask):
                w = alpha_opt * np.exp(-alpha_opt * sim.grid.x[mult_mask])
                w_sum = float(np.sum(w))
                Pe_mean[j] = float(np.sum(Pe[mult_mask] * w) / w_sum)
                Ph_mean[j] = float(np.sum(Ph[mult_mask] * w) / w_sum)
                Ptr_mean[j] = float(np.sum(Ptr[mult_mask] * w) / w_sum)
        except Exception:
            pass

    log.info("  Ptr_mean range: %.4f – %.4f",
             float(np.nanmin(Ptr_mean)), float(np.nanmax(Ptr_mean)))

    get_plotter("trigger_back_calculate", plot_dir=PLOT_DIR).plot(
        x, Pe_spatial, Ph_spatial, Ptr_spatial, E_spatial,
        labels, vex_arr, Pe_mean, Ph_mean, Ptr_mean, Vbr=Vbr)
