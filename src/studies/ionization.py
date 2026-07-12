"""Ionization coefficient and multiplication studies."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR

log = get_logger()


def run_ionization_vs_field(sim: SPADSimulator, Vbr: float) -> None:
    """Sweep E, compute raw and effective (dead-space) alpha(E) and beta(E)."""
    E_arr = np.logspace(5, 7, 200)  # V/cm

    ion = sim.ionization
    alpha_raw = ion.alpha_n(E_arr)
    beta_raw = ion.alpha_p(E_arr)
    alpha_eff = ion.effective_alpha_n(E_arr, Eg=1.35)
    beta_eff = ion.effective_alpha_p(E_arr, Eg=1.35)

    log.info(f"  alpha_max={alpha_raw.max():.2e} cm^-1  "
             f"beta_max={beta_raw.max():.2e} cm^-1")

    get_plotter("ionization_vs_field", plot_dir=PLOT_DIR).plot(
        E_arr,
        {"Raw": alpha_raw, "Effective (dead-space)": alpha_eff},
        {"Raw": beta_raw, "Effective (dead-space)": beta_eff},
        material_name="InP")


def run_multiplication_vs_vex(sim: SPADSimulator, Vbr: float) -> None:
    """Compute multiplication factor M vs (Vbr - V) via McIntyre integral."""
    # Sweep below breakdown: M diverges as V → Vbr⁻
    # Sweep from just below Vbr down to Vbr - 5V
    delta_range = np.linspace(0.2, 5, 24)
    Vex_arr = -delta_range  # negative excess voltage = below breakdown
    M_vals = []

    for delta in delta_range:
        Vb = Vbr - delta
        try:
            _, E, Pe, Ph, _, _ = sim.get_fields(float(Vb))
            # Electron injection: M_n = 1 / (1 - Pe)
            # Use Pe at the start of the multiplication layer
            M = float(1.0 / np.clip(1.0 - Pe[int(len(Pe) * 0.1)], 1e-10, None))
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
