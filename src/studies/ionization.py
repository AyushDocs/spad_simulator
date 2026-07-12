"""Ionization coefficient and multiplication studies."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..avalanche.ionization import IonizationCoefficients
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR

log = get_logger()


def run_ionization_vs_field(sim: SPADSimulator, Vbr: float) -> None:
    """Sweep E, compute alpha(E) and beta(E)."""
    E_arr = np.logspace(5, 7, 200)  # V/cm

    ion = sim.ionization
    alpha_e = ion.alpha_n(E_arr)
    beta_h = ion.alpha_p(E_arr)

    log.info(f"  alpha_max={alpha_e.max():.2e} cm^-1  "
             f"beta_max={beta_h.max():.2e} cm^-1")

    get_plotter("ionization_vs_field", plot_dir=PLOT_DIR).plot(
        E_arr, {"Default": alpha_e}, {"Default": beta_h},
        material_name="InP")


def run_multiplication_vs_vex(sim: SPADSimulator, Vbr: float) -> None:
    """Compute multiplication factor M vs excess bias."""
    Vex_range = np.linspace(0.1, 10, 20)
    M_vals = []

    for Vex in Vex_range:
        try:
            _, E, _, _, _, _ = sim.get_fields(float(Vbr + Vex))
            alpha = sim.ionization.alpha_n(np.abs(E))
            active = np.abs(E) > 1e5
            if np.any(active):
                alpha_active = alpha[active]
                M = float(np.exp(np.mean(alpha_active) * 1e-5))
            else:
                M = 1.0
            M_vals.append(M)
        except Exception as e:
            log.info(f"  Vex={Vex:.1f}V failed: {e}")
            M_vals.append(np.nan)

    M_arr = np.array(M_vals)
    mask = np.isfinite(M_arr) & (M_arr > 0)
    if np.any(mask):
        get_plotter("multiplication_vs_vex", plot_dir=PLOT_DIR).plot(
            Vex_range[mask], M_arr[mask], Vbr=Vbr)
        log.info(f"  M: {np.nanmin(M_arr[mask]):.1f} - {np.nanmax(M_arr[mask]):.1f}")
