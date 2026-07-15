"""Trap-density sweep: I-V components for different N_T values."""
from __future__ import annotations

import numpy as np

from ..simulator.photocurrent import compute_photocurrent
from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR, OPTICAL_POWER

log = get_logger()


def run_trap_density_iv(sim: SPADSimulator, Vbr: float) -> None:
    """Sweep reverse bias for 4 trap densities and plot components.

    For each N_T in [0, 1e15, 5e15, 1e16] the total dark current,
    photocurrent, SRH, BTBT, and avalanche contribution are recorded
    vs reverse bias and rendered as a 2×2 subplot.
    """
    nt_values = [0.0, 1e15, 5e15, 1e16]
    V_sweep = np.arange(0.0, Vbr + 35.0, 1.0)
    subplots_data = []

    for nt in nt_values:
        sim.set_nt(nt)
        dark, optical, srh, btbt, avalanche = [], [], [], [], []
        n_ok = 0

        for V in V_sweep:
            try:
                _, E, Pe, Ph, _, xr = sim.get_fields(float(V))
                x = sim.grid.x
                F = np.abs(E)
                area = sim.detector_area

                comps = sim.current.compute_individual(x, F)
                I_srh = float(np.trapezoid(comps["SRH"], x) * area)
                I_btbt = float(np.trapezoid(comps["BTBT"], x) * area)
                I_tat = float(np.trapezoid(comps["TAT"], x) * area)
                I_primary = I_srh + I_btbt + I_tat

                dc = sim.compute_dark_current(float(V), E=E)
                M = dc["M"]
                I_dark = I_primary * M

                try:
                    I_photo = compute_photocurrent(
                        grid_x=x, layers=sim.device.layers,
                        materials=sim.materials, pde_model=sim.pde_model,
                        detector_area=area,
                        wavelength=1550e-9, power=OPTICAL_POWER,
                        E=E, Pe=Pe, Ph=Ph, xr=xr,
                        multiply=True, M=dc["M"],
                    )
                except Exception as exc:
                    log.warning("  photocurrent failed at V=%.1f: %s", V, exc)
                    I_photo = 0.0

                I_av = I_primary * (M - 1.0) if M > 1.0 else 0.0

                dark.append(I_dark)
                optical.append(I_photo)
                srh.append(I_srh)
                btbt.append(I_btbt)
                avalanche.append(I_av)
                n_ok += 1
            except Exception as exc:
                log.warning("  components failed at V=%.1f: %s", V, exc)
                dark.append(np.nan)
                optical.append(np.nan)
                srh.append(np.nan)
                btbt.append(np.nan)
                avalanche.append(np.nan)

        subplots_data.append({
            "label": f"N$_\\mathrm{{T}}$ = {nt:.0e} cm$^{{-3}}$",
            "V": V_sweep.copy(),
            "dark": np.array(dark),
            "optical": np.array(optical),
            "srh": np.array(srh),
            "btbt": np.array(btbt),
            "avalanche": np.array(avalanche),
        })
        log.info("  Trap density N_T=%.0e: %d/%d bias points ok",
                 nt, n_ok, len(V_sweep))

    get_plotter("trap_density_iv", plot_dir=PLOT_DIR).plot(
        subplots_data, Vbr=Vbr)
