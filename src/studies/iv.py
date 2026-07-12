"""I-V characteristic studies."""
from __future__ import annotations

import numpy as np

from ..simulator.photocurrent import compute_photocurrent
from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR, OPTICAL_POWER

log = get_logger()

R_Q = 1e5  # quenching resistor (Ohms)


def run_iv_characteristic(sim: SPADSimulator, Vbr: float) -> None:
    V_sweep = np.arange(0, Vbr + 30, 1.0)
    I_dark, I_light = [], []
    for V in V_sweep:
        try:
            _, E, Pe, Ph, _, xr = sim.get_fields(float(V))
            dc = sim.compute_dark_current(float(V), E=E)
            I_dark.append(dc["I_dark"])
            I_photo = compute_photocurrent(
                grid_x=sim.grid.x,
                layers=sim.device.layers,
                materials=sim.materials,
                pdp_model=sim.pdp_model,
                detector_area=sim.detector_area,
                wavelength=1550e-9,
                power=OPTICAL_POWER,
                E=E,
                Pe=Pe,
                Ph=Ph,
                xr=xr,
                multiply=True,
                V_bias=float(V),
                V_br=Vbr,
            )
            I_light.append(dc["I_dark"] + I_photo)
        except Exception:
            I_dark.append(np.nan)
            I_light.append(np.nan)

    n = min(len(V_sweep), len(I_dark))
    V_sweep = V_sweep[:n]
    I_dark_arr = np.array(I_dark[:n])
    I_light_arr = np.array(I_light[:n])
    mask = np.isfinite(I_dark_arr)
    if np.any(mask):
        get_plotter("iv_characteristic", plot_dir=PLOT_DIR).plot(
            V_sweep[mask], I_dark_arr[mask], I_light=I_light_arr[mask],
            optical_power=OPTICAL_POWER, Vbr=Vbr)


def run_comprehensive_iv(sim: SPADSimulator, Vbr: float) -> None:
    V_sweep = np.arange(0, Vbr + 30, 1.0)
    I_dark = []
    for V in V_sweep:
        try:
            _, E, Pe, Ph, _, _ = sim.get_fields(float(V))
            dc = sim.compute_dark_current(float(V), E=E)
            I_dark.append(dc["I_dark"])
        except Exception:
            I_dark.append(np.nan)

    n = min(len(V_sweep), len(I_dark))
    V_sweep = V_sweep[:n]
    I_dark_arr = np.array(I_dark[:n])
    mask = np.isfinite(I_dark_arr)
    if np.any(mask):
        get_plotter("comprehensive_iv", plot_dir=PLOT_DIR).plot(
            V_sweep[mask], I_dark_arr[mask], Vbr=Vbr)
