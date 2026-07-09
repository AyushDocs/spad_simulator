"""Field and structure studies: breakdown, device plot, potential/E-field sweep, trigger profiles."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR

log = get_logger()


def find_breakdown(sim: SPADSimulator) -> float:
    Vbr, _ = sim.find_breakdown(V_start=0, V_max=80, V_step=1.0)
    if Vbr is None:
        raise ValueError("No breakdown detected")
    log.info(f"Vbr = {Vbr:.1f} V")
    return Vbr


def plot_device_structure(sim: SPADSimulator) -> None:
    for i, lyr in enumerate(sim.device.layers):
        label = "i"
        if lyr.doping_A > 1e14:
            label = "n+" if lyr.is_donor else "p+"
        log.info(f"Layer {i}:  {label:3s}  {lyr.thickness * 1e4:.2f} um  "
                 f"{lyr.material:8s}  A={lyr.doping_A:.1e}  ({lyr.doping_type})")

    p = get_plotter("device_structure", plot_dir=PLOT_DIR)
    p.plot(sim.grid.x, sim.device.material.mat_name,
           sim.device.net_doping_on_grid, sim.device.net_doping_on_grid)


def run_field_sweep(sim: SPADSimulator, Vbr: float) -> None:
    Vex_list = [0, 1, 2, 3, 4, 5]
    V_list = [Vbr + vex for vex in Vex_list]
    phi_list, E_list = [], []
    for V in V_list:
        phi, E, _ = sim.solve_poisson(float(V))
        phi_list.append(phi)
        E_list.append(E)
        log.info(f"Vex = {V - Vbr:.0f} V  phi_max = {phi.max():.1f}  "
                 f"|E|_max = {np.max(np.abs(E)):.2e}")

    get_plotter("potential_profile", plot_dir=PLOT_DIR).plot(
        sim.grid.x, np.array(phi_list), V_list)
    get_plotter("electric_field", plot_dir=PLOT_DIR).plot(
        sim.grid.x, np.array(E_list), V_list, Vbr=Vbr)


def run_trigger_profiles(sim: SPADSimulator, Vbr: float) -> None:
    Vex_list = [1, 3, 5]
    Pe_list, Ph_list, V_list = [], [], []
    for Vex in Vex_list:
        try:
            Pe, Ph, E = sim.solve_trigger(Vbr + Vex)
            Pe_list.append(Pe)
            Ph_list.append(Ph)
            V_list.append(Vbr + Vex)
            log.info(f"  Trigger Vex={Vex}V  Pe_max={np.max(Pe):.4f}  "
                     f"Ph_max={np.max(Ph):.4f}")
        except Exception as e:
            log.info(f"  Trigger Vex={Vex}V failed: {e}")

    if Pe_list:
        get_plotter("trigger_probability", plot_dir=PLOT_DIR).plot(
            sim.grid.x, np.array(Pe_list), np.array(Ph_list), V_list)
