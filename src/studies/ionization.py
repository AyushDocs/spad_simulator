"""Ionization coefficient and multiplication studies."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..avalanche.ionization import VanOverstraetenDeManCoefficients
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ..utils.loaders import PlotConfig
from ._config import PLOT_DIR, EG_INP

log = get_logger()


def run_ionization_vs_field(sim: SPADSimulator, Vbr: float,
                            plot_cfg: PlotConfig | None = None) -> None:
    """Sweep E, compute raw and effective (dead-space) alpha(E) and beta(E).

    Compares the simulator's default model with the Van Overstraeten–de Man
    model and reports the ionization ratio with the operating field marked.
    """
    if plot_cfg and not plot_cfg.is_enabled("ionization_vs_field"):
        return
    E_arr = np.logspace(5, 7, 200)  # V/cm

    # Default model (Okuto-Crowell or Chynoweth from simulator config)
    ion = sim.ionization
    alpha_oc_raw = ion.alpha_n(E_arr)
    beta_oc_raw = ion.alpha_p(E_arr)
    alpha_oc_eff = ion.effective_alpha_n(E_arr, Eg=EG_INP)
    beta_oc_eff = ion.effective_alpha_p(E_arr, Eg=EG_INP)

    # Van Overstraeten–de Man model for comparison
    mat = sim.materials.get("InP", next(iter(sim.materials.values())))
    vodm = VanOverstraetenDeManCoefficients(mat, T=sim.T)
    alpha_vodm = vodm.alpha_n(E_arr)
    beta_vodm = vodm.alpha_p(E_arr)

    log.info(f"  OC alpha_max={alpha_oc_raw.max():.2e} cm^-1  "
             f"OC beta_max={beta_oc_raw.max():.2e} cm^-1")
    log.info(f"  VODM alpha_max={alpha_vodm.max():.2e} cm^-1  "
             f"VODM beta_max={beta_vodm.max():.2e} cm^-1")

    get_plotter("ionization_vs_field", plot_dir=PLOT_DIR).plot(
        E_arr,
        {"Okuto-Crowell": alpha_oc_raw,
         "OC (dead-space)": alpha_oc_eff,
         "VODM": alpha_vodm},
        {"Okuto-Crowell": beta_oc_raw,
         "OC (dead-space)": beta_oc_eff,
         "VODM": beta_vodm},
        material_name="InP")

    # Ionization ratio k = beta/alpha with peak operating field
    k_oc = np.where(alpha_oc_raw > 0, beta_oc_raw / alpha_oc_raw, np.nan)
    k_vodm = np.where(alpha_vodm > 0, beta_vodm / alpha_vodm, np.nan)

    try:
        _, E_peak, _, _, _, _ = sim.get_fields(Vbr)
        peak_field = float(np.max(np.abs(E_peak)))
    except Exception:
        peak_field = None

    get_plotter("ionization_ratio", plot_dir=PLOT_DIR).plot(
        E_arr,
        {"Okuto-Crowell": k_oc, "VODM": k_vodm},
        material_name="InP",
        peak_field=peak_field)


def run_multiplication_vs_vex(sim: SPADSimulator, Vbr: float,
                              plot_cfg: PlotConfig | None = None) -> None:
    """Compute APD multiplication factor M vs excess voltage below breakdown.

    Uses the coupled McIntyre first-order ODEs for multiplication:
        dMn/dx = −α(x)·(Mn + Mp),  Mn(W) = 1
        dMp/dx = +β(x)·(Mn + Mp),  Mp(0) = 1
    where M = Mn(0) for electron injection.  This is the standard
    below-breakdown APD gain, *not* the Geiger-mode trigger probability.
    """
    if plot_cfg and not plot_cfg.is_enabled("multiplication_vs_vex"):
        return
    # Sweep below breakdown: M diverges as V → Vbr⁻
    delta_range = np.linspace(0.2, 8, 30)
    V_arr = Vbr - delta_range  # bias voltage (below breakdown)
    M_vals = []

    for Vb in V_arr:
        try:
            _, E, _, _, _, _ = sim.get_fields(float(Vb))
            M = sim._compute_multiplication(E)
            M_vals.append(M)
        except Exception as e:
            log.info(f"  V={Vb:.1f}V failed: {e}")
            M_vals.append(np.nan)

    M_arr = np.array(M_vals)
    mask = np.isfinite(M_arr) & (M_arr > 0)
    if np.any(mask):
        get_plotter("multiplication_vs_vex", plot_dir=PLOT_DIR).plot(
            V_arr[mask], M_arr[mask], Vbr=Vbr)
        log.info(f"  M: {np.nanmin(M_arr[mask]):.1f} - {np.nanmax(M_arr[mask]):.1f}")
