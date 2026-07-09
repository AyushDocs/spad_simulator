"""PDP and PDE studies."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..utils.ingestion import DataIngestionService
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR

log = get_logger()


def run_pdp_spectrum(sim: SPADSimulator, Vbr: float) -> None:
    pdp_wavelengths = np.linspace(900, 1700, 41) * 1e-9
    pdp_spectra, vex_list_pdp = [], []
    for Vex in [1, 3, 5, 8]:
        try:
            _, E, _ = sim.solve_poisson(Vbr + Vex)
            Pe, Ph = sim._trigger_for_pdp(E)
            _, xr, _ = sim.depletion_width(Vbr + Vex)
            pdp = sim.compute_pdp_spectrum(
                pdp_wavelengths, float(Vex), material_name="InGaAs",
                E=E, Pe=Pe, Ph=Ph, xr=xr)
            pdp = np.clip(pdp, 0, 1)
            pdp_spectra.append(pdp)
            vex_list_pdp.append(Vex)
            log.info(f"  Vex = {Vex} V: PDP max = {np.max(pdp) * 100:.4f}%")
        except Exception as e:
            log.info(f"  Vex = {Vex} V: {e}")

    if pdp_spectra:
        get_plotter("pdp", plot_dir=PLOT_DIR).plot(
            pdp_wavelengths, np.array(pdp_spectra), vex_list_pdp)


def run_pdp_vs_vex(sim: SPADSimulator, Vbr: float) -> None:
    dead_zone_layers, absorber = sim.pdp_model.find_absorber(
        sim.device.layers, "InGaAs")
    dead_zone = sum(l.thickness for l in dead_zone_layers)

    wl_apt = np.array([1100, 1310, 1550, 1610])
    vex_pts = np.linspace(0, 10, 11)
    pdp_dict = {lam: [] for lam in wl_apt}
    for Vex in vex_pts:
        try:
            _, E, _ = sim.solve_poisson(Vbr + Vex)
            Pe, Ph = sim._trigger_for_pdp(E)
            _, xr, _ = sim.depletion_width(Vbr + Vex)
            Ptr = Pe + Ph - Pe * Ph
            x_end = min(xr, dead_zone + absorber.thickness)
            mask = (sim.grid.x >= dead_zone) & (sim.grid.x <= x_end)
            xx = sim.grid.x[mask] - dead_zone
            for lam in wl_apt:
                trans = sim.pdp_model.dead_zone_transmission(lam * 1e-9, dead_zone_layers)
                pdp = sim.pdp_model.pdp_integral(
                    lam * 1e-9, xx, Ptr[mask], trans, sim.grid.dx,
                    material_name="InGaAs")
                pdp_dict[lam].append(pdp)
        except Exception:
            for lam in wl_apt:
                pdp_dict[lam].append(0.0)

    get_plotter("pdp_vs_vex", plot_dir=PLOT_DIR).plot(
        vex_pts, {lam: np.array(v) for lam, v in pdp_dict.items()},
        wavelengths_nm=wl_apt)


def run_pdp_vs_temp(svc: DataIngestionService, Vbr: float) -> dict:
    temps = np.array([285, 315])
    Vex = 3.0
    wavelengths = [1310, 1550]
    pdp_dict = {wl: [] for wl in wavelengths}

    for T in temps:
        try:
            sim_T, Vbr_T = svc.build_simulator_at_temp(T)
            for wl in wavelengths:
                pdp_spectrum = sim_T.compute_pdp_spectrum(
                    np.array([wl * 1e-9]), float(Vex), material_name="InGaAs")
                pdp_dict[wl].append(float(pdp_spectrum[0]))
            log.info(f"  T={T}K  Vbr={Vbr_T:.1f}V  "
                     f"PDP1310={pdp_dict[1310][-1]*100:.1f}%  "
                     f"PDP1550={pdp_dict[1550][-1]*100:.1f}%")
        except Exception as e:
            log.info(f"  T={T}K failed: {e}")
            for wl in wavelengths:
                pdp_dict[wl].append(0.0)

    pdp_plot = {wl: np.array(vals) for wl, vals in pdp_dict.items()}
    get_plotter("pdp_vs_temp", plot_dir=PLOT_DIR).plot(
        temps, pdp_plot, wavelengths_nm=np.array(wavelengths))

    return {"temperatures_K": temps.tolist(), "pdp": pdp_dict, "Vex": Vex}


def run_pde_vs_bias(sim: SPADSimulator, Vbr: float) -> dict:
    Vex_range = np.linspace(0, 10, 21)
    wavelength = 1310e-9
    PDE_vals = []

    for Vex in Vex_range:
        try:
            pdp_spectrum = sim.compute_pdp_spectrum(
                np.array([wavelength]), float(Vex),
                material_name="InGaAs")
            PDE_vals.append(float(pdp_spectrum[0]))
        except Exception:
            PDE_vals.append(0.0)

    PDE_arr = np.array(PDE_vals)
    get_plotter("pde", plot_dir=PLOT_DIR).plot(Vex_range, PDE_arr)

    pde_max = float(np.max(PDE_arr))
    log.info(f"  PDE(1310nm): max={pde_max*100:.1f}%")
    return {"pde_max": pde_max, "wavelength_nm": 1310}


def collect_pdp_max_metrics(sim: SPADSimulator, wavelengths_nm: list, Vex: float = 3.0) -> dict:
    """Collect PDP at key wavelengths for artifact output."""
    metrics: dict = {}
    for wl_nm in wavelengths_nm:
        try:
            pdp_spectrum = sim.compute_pdp_spectrum(
                np.array([wl_nm * 1e-9]), Vex, material_name="InGaAs")
            metrics[f"{wl_nm}nm"] = float(np.max(pdp_spectrum))
        except Exception:
            metrics[f"{wl_nm}nm"] = 0.0
    return metrics
