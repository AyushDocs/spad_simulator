"""Field and structure studies: breakdown, structure, sweep, trigger, peak field, avalanche map."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR

log = get_logger()


def find_breakdown(sim: SPADSimulator) -> float:
    Vbr, _ = sim.find_breakdown(V_start=0, V_max=150, V_step=1.0)
    if Vbr is None:
        raise ValueError("No breakdown detected")
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
    Vex_list = [-10, -5, -2, 0.5, 1, 2, 3, 5]
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
        Vex_succeeded = [V - Vbr for V in V_list]
        sub_br_idx = [i for i, vex in enumerate(Vex_succeeded) if vex < 0]
        geiger_low_idx = [i for i, vex in enumerate(Vex_succeeded) if 0 <= vex <= 2]
        geiger_high_idx = [i for i, vex in enumerate(Vex_succeeded) if vex > 2]

        plotter = get_plotter("trigger_probability", plot_dir=PLOT_DIR)

        if sub_br_idx:
            pe_sub = np.array([Pe_list[i] for i in sub_br_idx])
            ph_sub = np.array([Ph_list[i] for i in sub_br_idx])
            v_sub = [V_list[i] for i in sub_br_idx]
            plotter.plot(sim.grid.x, pe_sub, ph_sub, v_sub, Vbr=Vbr,
                         doping=sim.device.net_doping_on_grid,
                         filename="trigger_probability_sub_breakdown.png")

        if geiger_low_idx:
            pe_low = np.array([Pe_list[i] for i in geiger_low_idx])
            ph_low = np.array([Ph_list[i] for i in geiger_low_idx])
            v_low = [V_list[i] for i in geiger_low_idx]
            plotter.plot(sim.grid.x, pe_low, ph_low, v_low, Vbr=Vbr,
                         doping=sim.device.net_doping_on_grid,
                         filename="trigger_probability_geiger_low.png")

        if geiger_high_idx:
            pe_high = np.array([Pe_list[i] for i in geiger_high_idx])
            ph_high = np.array([Ph_list[i] for i in geiger_high_idx])
            v_high = [V_list[i] for i in geiger_high_idx]
            plotter.plot(sim.grid.x, pe_high, ph_high, v_high, Vbr=Vbr,
                         doping=sim.device.net_doping_on_grid,
                         filename="trigger_probability_geiger_high.png")

        # Calculate succeeded Vex and ATP list, then plot
        succeeded_vex = [V - Vbr for V in V_list]
        atp_list = [pe + ph - pe * ph for pe, ph in zip(Pe_list, Ph_list)]
        get_plotter("atp", plot_dir=PLOT_DIR).plot(
            sim.grid.x * 1e4, succeeded_vex, atp_list, Vbr=Vbr)


def run_peak_field_vs_bias(sim: SPADSimulator, Vbr: float) -> None:
    """Sweep bias voltage, extract peak electric field at each."""
    Vex_range = np.linspace(0, 10, 21)
    Vbias_arr = Vbr + Vex_range
    E_peak = []

    for Vbias in Vbias_arr:
        try:
            _, E, _ = sim.solve_poisson(float(Vbias))
            E_peak.append(float(np.max(np.abs(E))))
        except Exception as e:
            log.info(f"  Vbias={Vbias:.1f}V failed: {e}")
            E_peak.append(np.nan)

    E_peak_arr = np.array(E_peak)
    mask = np.isfinite(E_peak_arr)
    if np.any(mask):
        get_plotter("peak_field_vs_bias", plot_dir=PLOT_DIR).plot(
            Vbias_arr[mask], E_peak_arr[mask], Vbr=Vbr)
        log.info(f"  Peak E: {np.nanmin(E_peak_arr):.2e} – "
                 f"{np.nanmax(E_peak_arr):.2e} V/cm")


def run_avalanche_probability_map(sim: SPADSimulator, Vbr: float) -> None:
    """2D heatmap: trigger probability vs depth and excess voltage."""
    Vex_arr = np.linspace(0, 8, 20)
    Pe_list = []

    for Vex in Vex_arr:
        try:
            Pe, Ph, E = sim.solve_trigger(Vbr + Vex)
            Pe_list.append(Pe)
        except Exception as e:
            log.info(f"  Vex={Vex:.1f}V failed: {e}")
            Pe_list.append(np.zeros_like(sim.grid.x))

    Pe_2d = np.array(Pe_list)
    x_um = sim.grid.x * 1e4
    get_plotter("avalanche_map", plot_dir=PLOT_DIR).plot(
        x_um, Vex_arr, Pe_2d, Vbr=Vbr)
    log.info(f"  Avalanche map: {Pe_2d.shape[0]} Vex × {Pe_2d.shape[1]} x points")


def run_trigger_vs_vex(sim: SPADSimulator, Vbr: float,
                       Vex_min: float = -20.0, Vex_max: float = 10.0,
                       n_pts: int = 61) -> None:
    """Plot spatially-averaged Pe and Ph vs excess voltage.

    Sweeps Vex from Vex_min (sub-breakdown) to Vex_max (Geiger regime).
    Averages over the InP multiplication layer only (x < 4 µm) to exclude
    the heterojunction field spike at the absorber/buffer interface.
    """
    Vex_arr = np.linspace(Vex_min, Vex_max, n_pts)
    Pe_mean_arr = np.full(n_pts, float("nan"))
    Ph_mean_arr = np.full(n_pts, float("nan"))
    Ptr_mean_arr = np.full(n_pts, float("nan"))

    # InP multiplication layer ends at ~3.5–4 µm from top surface
    x_mult_max = 4e-4  # cm  (4 µm)
    x = sim.grid.x

    for j, Vex in enumerate(Vex_arr):
        Vbias = Vbr + Vex
        if Vbias <= 0:
            continue
        try:
            Pe, Ph, E = sim.solve_trigger(Vbias)
            Ptr = Pe + Ph - Pe * Ph
            # Restrict to multiplication layer, exclude heterojunction spike
            mult_mask = (np.abs(E) > 1e4) & (x < x_mult_max)
            if np.any(mult_mask):
                Pe_mean_arr[j] = float(np.mean(Pe[mult_mask]))
                Ph_mean_arr[j] = float(np.mean(Ph[mult_mask]))
                Ptr_mean_arr[j] = float(np.mean(Ptr[mult_mask]))
            else:
                Pe_mean_arr[j] = 0.0
                Ph_mean_arr[j] = 0.0
                Ptr_mean_arr[j] = 0.0
        except Exception as e:
            log.info(f"  trigger Vex={Vex:.1f}V failed: {e}")

    log.info(f"  Pe_mean range: {np.nanmin(Pe_mean_arr):.4f} – {np.nanmax(Pe_mean_arr):.4f}")
    log.info(f"  Ph_mean range: {np.nanmin(Ph_mean_arr):.4f} – {np.nanmax(Ph_mean_arr):.4f}")
    log.info(f"  Ptr_mean range: {np.nanmin(Ptr_mean_arr):.4f} – {np.nanmax(Ptr_mean_arr):.4f}")

    get_plotter("trigger_vs_vex", plot_dir=PLOT_DIR).plot(
        Vex_arr, Pe_mean_arr, Ph_mean_arr, Vbr=Vbr, Ptr_max=Ptr_mean_arr)


def run_breakdown_vs_temp(svc, Vbr_room: float) -> dict:
    """Sweep temperature, find breakdown voltage at each."""
    temps = np.array([250, 275, 300, 325, 350])
    Vbr_arr = []

    for T in temps:
        try:
            sim_T, Vbr_T = svc.build_simulator_at_temp(T)
            Vbr_arr.append(Vbr_T)
            log.info(f"  T={T}K  Vbr={Vbr_T:.1f} V")
        except Exception as e:
            log.info(f"  T={T}K failed: {e}")
            Vbr_arr.append(np.nan)

    Vbr_np = np.array(Vbr_arr)
    mask = np.isfinite(Vbr_np)
    if np.any(mask):
        get_plotter("breakdown_voltage_vs_temp", plot_dir=PLOT_DIR).plot(
            temps[mask], Vbr_np[mask])

    return {"temperatures_K": temps.tolist(), "Vbr_V": Vbr_np.tolist()}
