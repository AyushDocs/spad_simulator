"""Dark current and DCR studies."""
from __future__ import annotations

import numpy as np

from scipy.constants import e as Q_e

from ..simulator import SPADSimulator
from ..utils.ingestion import DataIngestionService
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ..utils.loaders import PlotConfig
from ..simulator.photocurrent import compute_pde_spectrum
from . import _config as _cfg
from ._config import PLOT_DIR

log = get_logger()

Q_E = 1.602e-19  # elementary charge (C)


def run_dark_current_sweep(sim: SPADSimulator, Vbr: float,
                           plot_cfg: PlotConfig | None = None) -> None:
    if plot_cfg and not plot_cfg.is_enabled("dark_current_sweep"):
        return
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


def run_dcr_vs_temp(svc: DataIngestionService, Vbr: float,
                    plot_cfg: PlotConfig | None = None) -> dict:
    if plot_cfg and not plot_cfg.is_enabled("dcr_vs_temp"):
        return {}
    temps = np.array([180, 200, 225, 250, 275, 300])
    Vex = 2.0
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


def run_dark_current_component_sweep(sim: SPADSimulator, Vbr: float,
                                     plot_cfg: PlotConfig | None = None) -> None:
    """Sweep excess bias and plot each dark current component separately."""
    if plot_cfg and not plot_cfg.is_enabled("dark_current_component_sweep"):
        return
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


def run_dark_current_components_vs_temp(svc: DataIngestionService, Vbr: float,
                                        plot_cfg: PlotConfig | None = None) -> dict:
    if plot_cfg and not plot_cfg.is_enabled("dark_current_components_vs_temp"):
        return {}
    temps = np.array([180, 200, 225, 250, 275, 300])
    Vex = 3.0
    J_total_vals = []
    J_srh_vals = []
    J_tat_vals = []
    J_btbt_vals = []

    for T in temps:
        try:
            sim_T, Vbr_T = svc.build_simulator_at_temp(T)
            Vbias = Vbr_T + Vex

            # Compute total DCR and fields
            _, E, _, _, _, _ = sim_T.get_fields(Vbias)

            # Get individual components
            comps = sim_T.current.compute_individual(sim_T.grid.x, np.abs(E))
            x = sim_T.grid.x

            # Integrate components over area to get Amperes, then divide by Area in cm^2 for A/cm^2
            area_cm2 = sim_T.detector_area * 1e4

            def int_comp(J_comp):
                I_comp = float(np.trapezoid(J_comp, x)) * sim_T.detector_area
                return abs(I_comp) / area_cm2

            J_srh_vals.append(int_comp(comps.get("SRH", np.zeros_like(x))))
            J_tat_vals.append(int_comp(comps.get("TAT", np.zeros_like(x))))
            J_btbt_vals.append(int_comp(comps.get("BTBT", np.zeros_like(x))))

            dc = sim_T.compute_dark_current(Vbias)
            J_total_vals.append(abs(dc["I_dark"] / dc["M"]) / area_cm2)

        except Exception as e:
            log.info(f"  T={T}K failed: {e}")
            J_total_vals.append(np.nan)
            J_srh_vals.append(np.nan)
            J_tat_vals.append(np.nan)
            J_btbt_vals.append(np.nan)

    from ..utils.plotter import get_plotter
    get_plotter("dark_current_vs_temp_components", plot_dir="plots/spad").plot(
        temps, np.array(J_srh_vals), np.array(J_btbt_vals), np.array(J_tat_vals),
        J_total=np.array(J_total_vals), Vex=Vex
    )

    return {"temperatures_K": temps.tolist(), "Vex": Vex,
            "J_total": J_total_vals, "J_srh": J_srh_vals,
            "J_tat": J_tat_vals, "J_btbt": J_btbt_vals}


def run_dcr_pde_vs_vex(sim: SPADSimulator, Vbr: float,
                       plot_cfg: PlotConfig | None = None) -> None:
    """Sweep excess voltage and plot DCR + PDE (1550 nm) on same graph."""
    if plot_cfg and not plot_cfg.is_enabled("dcr_pde_vs_vex"):
        return

    vex_pts = np.linspace(0, 10, 21)
    wl = 1550e-9
    dcr_arr = np.full(len(vex_pts), np.nan)
    pde_arr = np.full(len(vex_pts), np.nan)

    for j, Vex in enumerate(vex_pts):
        Vbias = Vbr + Vex
        if Vbias <= 0:
            continue
        try:
            _, E, Pe, Ph, xl, xr = sim.get_fields(Vbias)
            dc = sim.compute_dark_current(Vbias, E=E)
            I_primary = dc["I_dark"] / dc["M"] if dc["M"] > 0 else dc["I_dark"]
            P_trig = _cfg.absorption_weighted_trigger(sim, E)
            dcr_arr[j] = abs(I_primary / Q_E) * P_trig

            pde_val = compute_pde_spectrum(
                grid_x=sim.grid.x,
                dx=sim.grid.dx,
                layers=sim.device.layers,
                pde_model=sim.pde_model,
                wavelengths=np.array([wl]),
                Vex=Vex,
                xr=xr,
                Pe=Pe,
                Ph=Ph,
                material_name="InGaAs",
            )
            pde_arr[j] = float(pde_val[0])
        except Exception:
            pass

    valid = np.isfinite(dcr_arr) & np.isfinite(pde_arr)
    if np.any(valid):
        log.info(f"DCR range: {np.nanmin(dcr_arr[valid]):.2e} – "
                 f"{np.nanmax(dcr_arr[valid]):.2e} cps")
        log.info(f"PDE range: {np.nanmin(pde_arr[valid])*100:.4f}% – "
                 f"{np.nanmax(pde_arr[valid])*100:.4f}%")
        get_plotter("dcr_pde_vs_vex", plot_dir=PLOT_DIR).plot(
            vex_pts[valid], dcr_arr[valid], pde_arr[valid], wavelength_nm=1550)


def run_generation_rate_profile(sim: SPADSimulator, Vbr: float,
                                plot_cfg: PlotConfig | None = None) -> None:
    """Plot SRH, BTBT, and TAT generation rates vs device depth at a fixed bias."""
    if plot_cfg and not plot_cfg.is_enabled("generation_rate_profile"):
        return
    Vex = 3.0
    Vb = Vbr + Vex
    x = sim.grid.x
    x_um = x * 1e4

    try:
        _, E, _, _, _, _ = sim.get_fields(float(Vb))
        F = np.abs(E)
        comps = sim.current.components
        J_srh = comps[0].compute(x, F)
        J_btbt = comps[1].compute(x, F)
        J_tat = comps[2].compute(x, F)

        G_srh = np.abs(J_srh) / Q_e
        G_btbt = np.abs(J_btbt) / Q_e
        G_tat = np.abs(J_tat) / Q_e

        get_plotter("generation_rate_profile", plot_dir=PLOT_DIR).plot(
            x_um, G_srh, G_btbt, G_tat, Vex=Vex)
        log.info(f"  Generation rate profile at Vex={Vex}V: "
                 f"SRH max={G_srh.max():.2e}  BTBT max={G_btbt.max():.2e}  TAT max={G_tat.max():.2e} cm⁻³·s⁻¹")
    except Exception as e:
        log.info(f"  Generation rate profile failed: {e}")
