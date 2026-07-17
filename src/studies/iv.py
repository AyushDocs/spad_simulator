"""I-V characteristic studies."""
from __future__ import annotations

import numpy as np

from ..simulator.photocurrent import compute_photocurrent
from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ..utils.loaders import PlotConfig
from ._config import PLOT_DIR, OPTICAL_POWER, R_Q

log = get_logger()


def run_iv_characteristic(sim: SPADSimulator, Vbr: float,
                         plot_cfg: PlotConfig | None = None) -> None:
    if plot_cfg and not plot_cfg.is_enabled("iv_characteristic"):
        return
    V_sweep = np.arange(0, Vbr + 30, 1.0)
    I_dark, I_light = [], []
    I_light_floor = 0.0  # last pre-breakdown light current (ensures dI/dV ≥ 0)
    for V in V_sweep:
        try:
            _, E, Pe, Ph, _, xr = sim.get_fields(float(V))
            dc = sim.compute_dark_current(float(V), E=E)

            if float(V) >= Vbr:
                V_ex = float(V) - Vbr
                I_geiger = V_ex / R_Q
                # Geiger-mode contribution adds on top of the pre-breakdown
                # currents, keeping the curve monotonic (dI/dV ≥ 0).
                # The McIntyre dark current and the photocurrent floor
                # provide a baseline; the Geiger term dominates at modest V_ex.
                I_dark_val = dc["I_dark"] + I_geiger
                I_light_val = I_light_floor + I_geiger
            else:
                I_dark_val = dc["I_dark"]
                M_photo = min(dc["M"], 2000.0)
                I_photo = compute_photocurrent(
                    grid_x=sim.grid.x,
                    layers=sim.device.layers,
                    materials=sim.materials,
                    pde_model=sim.pde_model,
                    detector_area=sim.detector_area,
                    wavelength=1550e-9,
                    power=OPTICAL_POWER,
                    E=E,
                    Pe=Pe,
                    Ph=Ph,
                    xr=xr,
                    multiply=True,
                    M=M_photo,
                )
                I_light_val = dc["I_dark"] + I_photo
                I_light_floor = I_light_val
        except Exception:
            I_dark_val = np.nan
            I_light_val = np.nan
        I_dark.append(I_dark_val)
        I_light.append(I_light_val)

    n = min(len(V_sweep), len(I_dark))
    V_sweep = V_sweep[:n]
    I_dark_arr = np.array(I_dark[:n])
    I_light_arr = np.array(I_light[:n])
    mask = np.isfinite(I_dark_arr)
    if np.any(mask):
        get_plotter("iv_characteristic", plot_dir=PLOT_DIR).plot(
            V_sweep[mask], I_dark_arr[mask], I_light=I_light_arr[mask],
            optical_power=OPTICAL_POWER, Vbr=Vbr)


def run_comprehensive_iv(sim: SPADSimulator, Vbr: float,
                        plot_cfg: PlotConfig | None = None) -> None:
    if plot_cfg and not plot_cfg.is_enabled("comprehensive_iv"):
        return
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
