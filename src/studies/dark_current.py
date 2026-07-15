"""Dark current and DCR studies."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..utils.ingestion import DataIngestionService
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from . import _config as _cfg
from ._config import PLOT_DIR

log = get_logger()

Q_E = 1.602e-19  # elementary charge (C)


def run_dark_current_sweep(sim: SPADSimulator, Vbr: float) -> None:
    Vex_range = np.linspace(0, 10, 11)
    I_dark, dcr = [], []

    for Vex in Vex_range:
        try:
            _, E, _, _, _, _ = sim.get_fields(float(Vbr + Vex))
            dc = sim.compute_dark_current(float(Vbr + Vex), E=E)
            i_dark_val = dc["I_dark"]
            # DCR = primary generation rate × trigger probability
            I_primary = dc["I_dark"] / dc["M"] if dc["M"] > 0 else dc["I_dark"]
            P_trig = _cfg.absorption_weighted_trigger(sim, E)
            dcr_val = abs(I_primary / Q_E) * P_trig  # cps
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
        get_plotter("dcr", plot_dir=PLOT_DIR).plot(
            Vex_range[mask], dcr_arr[mask], Vbr=Vbr)


def run_dcr_vs_temp(svc: DataIngestionService, Vbr: float) -> dict:
    temps = np.array([250, 275, 300, 325, 350])
    Vex = 3.0
    DCR_vals = []
    Vbr_vals = []

    for T in temps:
        try:
            sim_T, Vbr_T = svc.build_simulator_at_temp(T)
            dc = sim_T.compute_dark_current(Vbr_T + Vex)
            I_primary = dc["I_dark"] / dc["M"] if dc["M"] > 0 else dc["I_dark"]
            P_trig = _cfg.absorption_weighted_trigger(sim_T, dc["E"])
            dcr = abs(I_primary / Q_E) * P_trig  # cps
            DCR_vals.append(dcr)
            Vbr_vals.append(Vbr_T)
            log.info(f"  T={T}K  Vbr={Vbr_T:.1f}V  DCR={dcr:.2e} cps")
        except Exception as e:
            log.info(f"  T={T}K failed: {e}")
            DCR_vals.append(np.nan)
            Vbr_vals.append(np.nan)

    DCR_arr = np.array(DCR_vals)
    mask = np.isfinite(DCR_arr)
    if np.any(mask):
        get_plotter("dcr_vs_temp", plot_dir=PLOT_DIR).plot(
            temps[mask], DCR_arr[mask], Vex=Vex)

    return {"temperatures_K": temps.tolist(), "DCR_cps": DCR_arr.tolist(),
            "Vex": Vex}


def run_dark_current_component_sweep(sim: SPADSimulator, Vbr: float) -> None:
    """Sweep excess bias and plot each dark current component separately."""
    Vex_range = np.linspace(-15, 50, 131)
    I_srh, I_btbt, I_tat = [], [], []
    x = sim.grid.x

    for Vex in Vex_range:
        Vb = Vbr + Vex
        if Vb <= 0:
            I_srh.append(np.nan)
            I_btbt.append(np.nan)
            I_tat.append(np.nan)
            continue
        try:
            _, E, _, _, _, _ = sim.get_fields(float(Vb))
            F = np.abs(E)
            comps = sim.current.components
            J_srh = comps[0].compute(x, F)
            J_btbt = comps[1].compute(x, F)
            J_tat = comps[2].compute(x, F)
            I_srh.append(float(np.trapezoid(J_srh, x) * sim.detector_area))
            I_btbt.append(float(np.trapezoid(J_btbt, x) * sim.detector_area))
            I_tat.append(float(np.trapezoid(J_tat, x) * sim.detector_area))
        except Exception:
            I_srh.append(np.nan)
            I_btbt.append(np.nan)
            I_tat.append(np.nan)

    arr_srh, arr_btbt, arr_tat = map(np.array, (I_srh, I_btbt, I_tat))
    mask = np.isfinite(arr_srh)
    if np.any(mask):
        get_plotter("dark_current_components", plot_dir=PLOT_DIR).plot(
            Vex_range[mask], arr_srh[mask], arr_btbt[mask], arr_tat[mask], Vbr=Vbr)


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
            sim_T, Vbr_T = svc.build_simulator_at_temp(T)
            dc = sim_T.compute_dark_current(Vbr_T + Vex)
            J_total_vals.append(dc["I_dark"])
        except Exception as e:
            log.info(f"  T={T}K failed: {e}")
            J_total_vals.append(np.nan)

    return {"temperatures_K": temps.tolist(), "Vex": Vex,
            "J_total": J_total_vals}
