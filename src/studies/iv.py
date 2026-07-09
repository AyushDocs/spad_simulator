"""I-V characteristic studies."""
from __future__ import annotations

import numpy as np

from ..core.constants import h, c
from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR, OPTICAL_POWER

log = get_logger()


def run_iv_characteristic(sim: SPADSimulator, Vbr: float) -> None:
    V_sweep = np.linspace(0, Vbr + 10, 61)
    I_dark, I_light = [], []
    for V in V_sweep:
        try:
            _, E, Pe, Ph, _, xr = sim.get_fields(float(V))
            dc = sim.compute_dark_current(float(V), E=E)
            I_dark.append(dc["I_dark"])
            photo = sim.compute_photocurrent(float(V), power=OPTICAL_POWER,
                                             E=E, Pe=Pe, Ph=Ph, xr=xr)
            I_light.append(photo + dc["I_dark"])
        except Exception:
            I_dark.append(np.nan)
            I_light.append(np.nan)

    I_dark, I_light = np.array(I_dark), np.array(I_light)
    mask = np.isfinite(I_dark)
    if np.any(mask):
        get_plotter("iv_characteristic", plot_dir=PLOT_DIR).plot(
            V_sweep[mask], I_dark[mask], I_light=I_light[mask],
            optical_power=OPTICAL_POWER)


def run_comprehensive_iv(sim: SPADSimulator, Vbr: float) -> None:
    dead_zone_layers, absorber = sim.pdp_model.find_absorber(
        sim.device.layers, "InGaAs")
    dead_zone = sum(l.thickness for l in dead_zone_layers)

    alpha_arr = np.array([
        sim.materials[lyr.material].absorption_coefficient(1310e-9)
        for lyr in sim.device.layers
    ])
    alpha_grid = np.zeros_like(sim.grid.x)
    xs = 0.0
    for lyr, alpha_val in zip(sim.device.layers, alpha_arr):
        xe = xs + lyr.thickness
        mask = (sim.grid.x >= xs - 1e-16) & (sim.grid.x <= xe + 1e-16)
        alpha_grid[mask] = alpha_val
        xs = xe

    Eph = h * c / 1310e-9
    phi_photon = OPTICAL_POWER / (Eph * sim.detector_area)
    absorber_start = dead_zone
    absorber_end = dead_zone + absorber.thickness

    V_sweep = np.linspace(Vbr - 5, Vbr + 10, 11)
    I_dark, I_photo_prim, I_total, M_vals = [], [], [], []
    for V in V_sweep:
        try:
            _, E, Pe, Ph, _, xr = sim.get_fields(float(V))
            dc = sim.compute_dark_current(float(V), E=E)
            Ptr = Pe + Ph - Pe * Ph
            M = min(1.0 / (1.0 - float(np.max(Ptr)) + 1e-15), 10000.0)
            M_vals.append(M)
            I_dark.append(dc["I_dark"])

            J_pp = sim.pdp_model.photocurrent_density(
                sim.grid.x, alpha_grid, phi_photon, absorber_start, absorber_end)
            # Trapezoidal rule: O(h²), handles non-uniform grid near heterojunctions
            I_pp = float(np.trapezoid(J_pp, sim.grid.x) * sim.detector_area)
            I_photo_prim.append(I_pp)
            I_total.append(dc["I_dark"] + I_pp * M)
        except Exception:
            I_dark.append(np.nan)
            I_photo_prim.append(np.nan)
            I_total.append(np.nan)
            M_vals.append(np.nan)

    I_dark, I_pp, I_total, M_vals = [np.array(a) for a in
                                      (I_dark, I_photo_prim, I_total, M_vals)]
    mask = np.isfinite(I_dark)
    if np.any(mask):
        get_plotter("comprehensive_iv", plot_dir=PLOT_DIR).plot(
            V_sweep[mask], I_dark[mask],
            I_photo_primary=I_pp[mask], I_total_illuminated=I_total[mask],
            gain=M_vals[mask], Vbr=Vbr)
