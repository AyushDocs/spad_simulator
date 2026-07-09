"""Dark current and DCR studies."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..utils.ingestion import DataIngestionService
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR

log = get_logger()


def run_dark_current_sweep(sim: SPADSimulator, Vbr: float) -> None:
    Vex_range = np.linspace(0, 10, 11)
    I_dark, dcr = [], []
    for Vex in Vex_range:
        try:
            dc = sim.compute_dark_current(float(Vbr + Vex))
            I_dark.append(dc["I_dark"])
            dcr.append(dc["DCR"])
        except Exception:
            I_dark.append(np.nan)
            dcr.append(np.nan)

    I_dark, dcr = np.array(I_dark), np.array(dcr)
    mask = np.isfinite(I_dark)
    if np.any(mask):
        log.info(f"I_dark: {np.nanmin(I_dark):.2e} - {np.nanmax(I_dark):.2e} A")
        log.info(f"DCR:    {np.nanmin(dcr):.2e} - {np.nanmax(dcr):.2e} cps")
        get_plotter("dark_current", plot_dir=PLOT_DIR).plot(Vex_range[mask], I_dark[mask])
        get_plotter("dcr", plot_dir=PLOT_DIR).plot(Vex_range[mask], dcr[mask])


def run_dcr_vs_temp(svc: DataIngestionService, Vbr: float) -> dict:
    temps = np.array([285, 315])
    Vex = 3.0
    DCR_vals = []

    for T in temps:
        try:
            sim_T, Vbr_T = svc.build_simulator_at_temp(T)
            dc = sim_T.compute_dark_current(Vbr_T + Vex)
            DCR_vals.append(dc["DCR"])
            log.info(f"  T={T}K  Vbr={Vbr_T:.1f}V  DCR={dc['DCR']:.2e} cps")
        except Exception as e:
            log.info(f"  T={T}K failed: {e}")
            DCR_vals.append(np.nan)

    DCR_arr = np.array(DCR_vals)
    mask = np.isfinite(DCR_arr)
    if np.any(mask):
        get_plotter("dcr_vs_temp", plot_dir=PLOT_DIR).plot(
            temps[mask], DCR_arr[mask], Vex=Vex)

    return {"temperatures_K": temps.tolist(), "DCR_cps": DCR_arr.tolist(),
            "Vex": Vex}
