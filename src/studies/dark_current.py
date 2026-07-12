"""Dark current and DCR studies."""
from __future__ import annotations

import numpy as np

from ..core.physics_helpers import avalanche_trigger_probability
from ..core.constants import eps0, q
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
            _, E, _, _, _, _ = sim.get_fields(float(Vbr + Vex))
            dc = sim.compute_dark_current(float(Vbr + Vex), E=E)
            i_dark_val = dc["I_dark"]
            dcr_val = dc["DCR"]
        except Exception:
            i_dark_val = np.nan
            dcr_val = np.nan

        I_dark.append(i_dark_val)
        dcr.append(dcr_val)

    I_dark_arr, dcr_arr = np.array(I_dark), np.array(dcr)
    mask = np.isfinite(I_dark_arr)
    if np.any(mask):
        log.info(f"I_dark: {np.nanmin(I_dark_arr):.2e} - {np.nanmax(I_dark_arr):.2e} A")
        log.info(f"DCR:    {np.nanmin(dcr_arr):.2e} - {np.nanmax(dcr_arr):.2e} cps")
        get_plotter("dark_current", plot_dir=PLOT_DIR).plot(
            Vex_range[mask], I_dark_arr[mask], Vbr=Vbr)
        get_plotter("dcr", plot_dir=PLOT_DIR).plot(Vex_range[mask], dcr_arr[mask])


def run_dcr_vs_temp(svc: DataIngestionService, Vbr: float) -> dict:
    temps = np.array([250, 275, 300, 325, 350])
    Vex = 3.0
    DCR_vals = []

    for T in temps:
        try:
            sim_T = svc.build_simulator(T)
            dVbr = (T - 300.0) * 0.002
            Vbr_T = Vbr + dVbr
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


def collect_dark_current_metrics(sim: SPADSimulator, Vbr: float, Vex: float = 3.0) -> dict:
    try:
        dc = sim.compute_dark_current(Vbr + Vex)
        return {"I_dark_A": dc["I_dark"], "DCR_cps": dc["DCR"], "Vex_V": Vex}
    except Exception:
        return {}


def run_dark_current_components_vs_temp(svc: DataIngestionService, Vbr: float) -> dict:
    temps = np.array([250, 275, 300, 325, 350])
    Vex = 3.0
    J_total_vals = []

    for T in temps:
        try:
            sim_T = svc.build_simulator(T)
            dVbr = (T - 300.0) * 0.002
            Vbr_T = Vbr + dVbr
            dc = sim_T.compute_dark_current(Vbr_T + Vex)
            J_total_vals.append(dc["I_dark"])
        except Exception as e:
            log.info(f"  T={T}K failed: {e}")
            J_total_vals.append(np.nan)

    return {"temperatures_K": temps.tolist(), "Vex": Vex,
            "J_total": J_total_vals}
