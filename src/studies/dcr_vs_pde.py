"""DCR vs PDE study for absorption layer width sweep."""
from __future__ import annotations

import numpy as np

from ..core.constants import q
from ..simulator import SPADSimulator
from ..simulator.photocurrent import compute_pde_spectrum
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from . import _config as _cfg
from ._config import PLOT_DIR

log = get_logger()


def run_dcr_vs_pde(sim: SPADSimulator, Vbr: float) -> None:
    """Sweep absorption layer width (constant multiplication) and plot DCR vs PDE.

    For each absorption width:
      1. Rebuild device with modified InGaAs thickness
      2. Find new Vbr
      3. Sweep excess voltage → compute DCR and PDE at 1550 nm
      4. Plot DCR (y, log, Hz/cm²) vs PDE (x, fraction) with one curve per width
    """
    idx = _cfg.layer_index_by_material(list(sim.device.layers), "InGaAs")
    if idx is None:
        log.warning("No InGaAs absorption layer found")
        return

    widths_um = np.linspace(0.5, 5.0, 10)
    widths_cm = widths_um * 1e-4

    original_layers = list(sim.device.layers)

    vex_pts = np.linspace(0.5, 10, 20)
    wl = 1550e-9
    curves: dict[str, dict] = {}

    for w_um, w_cm in zip(widths_um, widths_cm):
        label = f"{w_um:.1f} µm"
        log.info("  Absorption width = %s", label)

        try:
            layers = list(original_layers)
            _cfg.mutate_thickness(layers, idx, w_cm)
            sim.set_layers(layers)
            vbr_new, _ = sim.find_breakdown(
                V_start=40, V_max=150, V_step=0.5, force=True)
            if vbr_new is None:
                log.info("    Vbr not found, skipping")
                continue
        except Exception as e:
            log.info("    Vbr search failed: %s", e)
            continue

        dcr_arr = np.full(len(vex_pts), np.nan)
        pde_arr = np.full(len(vex_pts), np.nan)

        for j, Vex in enumerate(vex_pts):
            Vbias = vbr_new + Vex
            try:
                _, E, Pe, Ph, xl, xr = sim.get_fields(Vbias)

                # DCR
                dc = sim.compute_dark_current(Vbias, E=E)
                I_primary = dc["I_dark"] / dc["M"] if dc["M"] > 0 else dc["I_dark"]
                P_trig = _cfg.absorption_weighted_trigger(sim, E)
                dcr_arr[j] = abs(I_primary / float(q.to("C").magnitude)) * P_trig

                # PDE at 1550 nm
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

        curves[label] = {"PDE": pde_arr, "DCR": dcr_arr, "Vex": vex_pts}
        valid = np.isfinite(dcr_arr) & np.isfinite(pde_arr) & (pde_arr > 0) & (dcr_arr > 0)
        if np.any(valid):
            log.info("    PDE range: %.2f%% – %.2f%%, DCR range: %.2e – %.2e cps",
                     float(np.nanmin(pde_arr[valid]) * 100),
                     float(np.nanmax(pde_arr[valid]) * 100),
                     float(np.nanmin(dcr_arr[valid])),
                     float(np.nanmax(dcr_arr[valid])))

    sim.set_layers(original_layers)

    if curves:
        get_plotter("dcr_vs_pde", plot_dir=PLOT_DIR).plot(
            curves, Vex=3.0, detector_area_cm2=sim.detector_area)
