"""Ionization coefficient and multiplication studies."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR, EG_INP

log = get_logger()


def run_ionization_vs_field(sim: SPADSimulator, Vbr: float) -> None:
    """Sweep E, compute raw and effective (dead-space) alpha(E) and beta(E).

    Also produces the ionization ratio plot (k = beta/alpha) with the
    device operating field marked.
    """
    E_arr = np.logspace(5, 7, 200)  # V/cm

    ion = sim.ionization
    alpha_raw = ion.alpha_n(E_arr)
    beta_raw = ion.alpha_p(E_arr)
    alpha_eff = ion.effective_alpha_n(E_arr, Eg=EG_INP)
    beta_eff = ion.effective_alpha_p(E_arr, Eg=EG_INP)

    log.info(f"  alpha_max={alpha_raw.max():.2e} cm^-1  "
             f"beta_max={beta_raw.max():.2e} cm^-1")

    get_plotter("ionization_vs_field", plot_dir=PLOT_DIR).plot(
        E_arr,
        {"Raw": alpha_raw, "Effective (dead-space)": alpha_eff},
        {"Raw": beta_raw, "Effective (dead-space)": beta_eff},
        material_name="InP")

    # Ionization ratio k = beta/alpha with peak operating field
    k_raw = np.where(alpha_raw > 0, beta_raw / alpha_raw, np.nan)
    k_eff = np.where(alpha_eff > 0, beta_eff / alpha_eff, np.nan)

    # Find peak field at breakdown
    try:
        _, E_peak, _, _, _, _ = sim.get_fields(Vbr)
        peak_field = float(np.max(np.abs(E_peak)))
    except Exception:
        peak_field = None

    get_plotter("ionization_ratio", plot_dir=PLOT_DIR).plot(
        E_arr,
        {"Raw": k_raw, "Effective (dead-space)": k_eff},
        material_name="InP",
        peak_field=peak_field)


def run_multiplication_vs_vex(sim: SPADSimulator, Vbr: float) -> None:
    """Compute APD multiplication factor M vs excess voltage below breakdown.

    Uses the coupled McIntyre first-order ODEs for multiplication:
        dMn/dx = −α(x)·(Mn + Mp),  Mn(W) = 1
        dMp/dx = +β(x)·(Mn + Mp),  Mp(0) = 1
    where M = Mn(0) for electron injection.  This is the standard
    below-breakdown APD gain, *not* the Geiger-mode trigger probability.
    """
    # Sweep below breakdown: M diverges as V → Vbr⁻
    delta_range = np.linspace(0.2, 8, 30)
    Vex_arr = -delta_range  # negative excess voltage = below breakdown
    M_vals = []

    for delta in delta_range:
        Vb = Vbr - delta
        try:
            _, E, _, _, _, _ = sim.get_fields(float(Vb))
            M = sim._compute_multiplication(E)
            M_vals.append(M)
        except Exception as e:
            log.info(f"  Vbr-{delta:.1f}V failed: {e}")
            M_vals.append(np.nan)

    M_arr = np.array(M_vals)
    mask = np.isfinite(M_arr) & (M_arr > 0)
    if np.any(mask):
        get_plotter("multiplication_vs_vex", plot_dir=PLOT_DIR).plot(
            Vex_arr[mask], M_arr[mask], Vbr=Vbr)
        log.info(f"  M: {np.nanmin(M_arr[mask]):.1f} - {np.nanmax(M_arr[mask]):.1f}")
