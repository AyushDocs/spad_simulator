"""PDE studies."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..simulator.photocurrent import compute_pde_spectrum
from ..utils.ingestion import DataIngestionService
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ..utils.loaders import PlotConfig
from ._config import PLOT_DIR

log = get_logger()


def run_pde_spectrum(sim: SPADSimulator, Vbr: float,
                     plot_cfg: PlotConfig | None = None) -> None:
    if plot_cfg and not plot_cfg.is_enabled("pde_spectrum"):
        return
    pde_wavelengths = np.linspace(900, 1700, 41) * 1e-9
    pde_spectra, vex_list_pde = [], []
    for Vex in [-15, -10, -5, -3, -1, 1, 3, 5]:
        try:
            _, E, Pe, Ph, xl, xr = sim.get_fields(Vbr + Vex)
            pde_vals = compute_pde_spectrum(
                grid_x=sim.grid.x,
                dx=sim.grid.dx,
                layers=sim.device.layers,
                pde_model=sim.pde_model,
                wavelengths=pde_wavelengths,
                Vex=Vex,
                xr=xr,
                Pe=Pe,
                Ph=Ph,
                material_name="InGaAs",
            )
            pde_spectra.append(pde_vals)
            vex_list_pde.append(Vex)
            log.info(f"  Vex = {Vex} V: PDE(1550nm) = {pde_vals[np.argmin(np.abs(pde_wavelengths - 1550e-9))] * 100:.4f}%")
        except Exception as e:
            log.info(f"  Vex = {Vex} V: {e}")

    if pde_spectra:
        get_plotter("pde", plot_dir=PLOT_DIR).plot(
            pde_wavelengths * 1e9, np.array(pde_spectra), vex_list_pde)


def run_pde_vs_vex(sim: SPADSimulator, Vbr: float,
                   plot_cfg: PlotConfig | None = None) -> None:
    if plot_cfg and not plot_cfg.is_enabled("pde_vs_vex"):
        return
    vex_pts = np.linspace(-20, 25, 46)
    wavelengths = [905, 1310, 1550, 1610]
    pde_dict: dict[int, np.ndarray] = {wl: np.zeros(len(vex_pts)) for wl in wavelengths}

    for j, Vex in enumerate(vex_pts):
        Vbias = Vbr + Vex
        if Vbias <= 0:
            continue
        try:
            _, E, Pe, Ph, xl, xr = sim.get_fields(Vbias)
            pde_vals = compute_pde_spectrum(
                grid_x=sim.grid.x,
                dx=sim.grid.dx,
                layers=sim.device.layers,
                pde_model=sim.pde_model,
                wavelengths=np.array([wl * 1e-9 for wl in wavelengths]),
                Vex=Vex,
                xr=xr,
                Pe=Pe,
                Ph=Ph,
                material_name="InGaAs",
            )
            for i, wl in enumerate(wavelengths):
                pde_dict[wl][j] = float(pde_vals[i])
        except Exception:
            pass

    get_plotter("pde_vs_vex", plot_dir=PLOT_DIR).plot(
        vex_pts, pde_dict, wavelengths_nm=np.array(wavelengths))


def run_pde_vs_temp(svc: DataIngestionService, Vbr: float,
                    plot_cfg: PlotConfig | None = None) -> dict:
    if plot_cfg and not plot_cfg.is_enabled("pde_vs_temp"):
        return {}
    temps = np.array([255, 275, 295, 315, 335])
    Vex = 3.0
    wavelengths = [1310, 1550]
    pde_dict: dict[int, list[float]] = {wl: [] for wl in wavelengths}

    for T in temps:
        try:
            sim_T, Vbr_T = svc.build_simulator_at_temp(T)
            Vbias = Vbr + Vex  # fixed bias across all temperatures
            _, E_T, Pe_T, Ph_T, xl_T, xr_T = sim_T.get_fields(Vbias)
            pde_vals = compute_pde_spectrum(
                grid_x=sim_T.grid.x,
                dx=sim_T.grid.dx,
                layers=sim_T.device.layers,
                pde_model=sim_T.pde_model,
                wavelengths=np.array([wl * 1e-9 for wl in wavelengths]),
                Vex=Vex,
                xr=xr_T,
                Pe=Pe_T,
                Ph=Ph_T,
                material_name="InGaAs",
            )
            for i, wl in enumerate(wavelengths):
                pde_dict[wl].append(float(pde_vals[i]))
            log.info(f"  T={T}K  Vbr={Vbr_T:.1f}V  Vbias={Vbias:.1f}V  PDE(1550nm)={pde_vals[wavelengths.index(1550)]*100:.2f}%")
        except Exception as e:
            log.info(f"  T={T}K failed: {e}")
            for wl in wavelengths:
                pde_dict[wl].append(0.0)

    pde_arr_dict = {wl: np.array(vals) for wl, vals in pde_dict.items()}
    mask = np.all(np.isfinite(list(pde_arr_dict.values())), axis=0)
    if np.any(mask):
        get_plotter("pde_vs_temp", plot_dir=PLOT_DIR).plot(
            temps[mask], {wl: vals[mask] for wl, vals in pde_arr_dict.items()})

    return {"temperatures_K": temps.tolist(), "pde": pde_dict, "Vex": Vex}


def collect_pde_max_metrics(sim: SPADSimulator, wavelengths_nm: list, Vex: float = 3.0) -> dict:
    try:
        # Find breakdown first
        Vbr = sim._Vbr
        if Vbr is None:
            Vbr, _ = sim.find_breakdown(V_start=0, V_max=150, V_step=1.0)
        _, E, Pe, Ph, xl, xr = sim.get_fields(Vbr + Vex)
        pde_vals = compute_pde_spectrum(
            grid_x=sim.grid.x,
            dx=sim.grid.dx,
            layers=sim.device.layers,
            pde_model=sim.pde_model,
            wavelengths=np.array([wl * 1e-9 for wl in wavelengths_nm]),
            Vex=Vex,
            xr=xr,
            Pe=Pe,
            Ph=Ph,
            material_name="InGaAs",
        )
        return {f"{wl}nm": float(pde_vals[i]) for i, wl in enumerate(wavelengths_nm)}
    except Exception as e:
        log.info(f"collect_pde_max_metrics failed: {e}")
        return {f"{wl}nm": 0.0 for wl in wavelengths_nm}



def run_absorption_profile(sim: SPADSimulator, Vbr: float,
                           plot_cfg: PlotConfig | None = None) -> None:
    """Plot Beer-Lambert absorption."""
    if plot_cfg and not plot_cfg.is_enabled("absorption_profile"):
        return
    x_abs = np.linspace(0, 1e-4, 200)
    x_um = x_abs * 1e4

    wavelengths_nm = [905, 1310, 1550, 1610]
    G_dict: dict[str, np.ndarray] = {}

    mat = sim.materials.get("InGaAs")
    fallbacks = {905: 1.5e4, 1310: 1.0e4, 1550: 7.0e3, 1610: 5.0e3}

    for lam_nm in wavelengths_nm:
        lam = lam_nm * 1e-9
        if mat is not None:
            alpha = float(mat.absorption_coefficient(lam))
        else:
            alpha = fallbacks.get(lam_nm, 1e4)
        G = alpha * np.exp(-alpha * x_abs)
        G_dict[str(lam_nm)] = G

    get_plotter("absorption_profile", plot_dir=PLOT_DIR).plot(
        x_um, G_dict, material_name="InGaAs")


def run_pde_3d(sim: SPADSimulator, Vbr: float) -> None:
    """3D PDE surface."""
    log.info("  PDE 3D: skipped (requires full PDE model)")
