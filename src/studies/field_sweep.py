"""Electric-field profile studies: sweep absorption and multiplication layer widths."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR

log = get_logger()


def run_efield_vs_absorption_width(sim: SPADSimulator, Vbr: float) -> None:
    """Plot E-field profiles for different absorption-layer (InGaAs) widths."""
    widths_um = [1.0, 2.5, 4.5]
    results: list[tuple[float, np.ndarray, np.ndarray]] = []

    original_layers = list(sim.device.layers)

    for w_um in widths_um:
        try:
            layers = [
                l.model_copy(update={"thickness": w_um * 1e-4
                                     if l.material == "InGaAs"
                                     else l.thickness})
                for l in original_layers
            ]
            sim.set_layers(layers)
            Vbr_i, _ = sim.find_breakdown(V_start=40, V_max=150, V_step=1.0,
                                          force=True)
            if Vbr_i is None:
                log.info("  Absorption width %.1f µm: no breakdown found", w_um)
                continue
            phi, E, _ = sim.solve_poisson(float(Vbr_i))
            E_arr = np.asarray(E)
            results.append((w_um, sim.grid.x.copy(), E_arr))
            log.info("  Absorption width %.1f µm: Vbr=%.1f V", w_um, Vbr_i)
        except Exception as e:
            log.info("  Absorption width %.1f µm failed: %s", w_um, e)

    if results:
        get_plotter("efield_vs_absorption", plot_dir=PLOT_DIR).plot(results)

    sim.set_layers(original_layers)


def run_efield_vs_multiplication_width(sim: SPADSimulator, Vbr: float) -> None:
    """Plot E-field profiles for different multiplication-layer (InP cap) widths."""
    widths_um = [0.2, 0.5, 1.0]
    results: list[tuple[float, np.ndarray, np.ndarray]] = []

    original_layers = list(sim.device.layers)

    for w_um in widths_um:
        try:
            layers = [
                l.model_copy(update={"thickness": w_um * 1e-4
                                     if l.material == "InP"
                                        and i == 1
                                     else l.thickness})
                for i, l in enumerate(original_layers)
            ]
            sim.set_layers(layers)
            Vbr_i, _ = sim.find_breakdown(V_start=40, V_max=150, V_step=1.0,
                                          force=True)
            if Vbr_i is None:
                log.info("  Multiplication width %.1f µm: no breakdown found",
                         w_um)
                continue
            phi, E, _ = sim.solve_poisson(float(Vbr_i))
            E_arr = np.asarray(E)
            results.append((w_um, sim.grid.x.copy(), E_arr))
            log.info("  Multiplication width %.1f µm: Vbr=%.1f V", w_um, Vbr_i)
        except Exception as e:
            log.info("  Multiplication width %.1f µm failed: %s", w_um, e)

    if results:
        get_plotter("efield_vs_multiplication", plot_dir=PLOT_DIR).plot(results)

    sim.set_layers(original_layers)
