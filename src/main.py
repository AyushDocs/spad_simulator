#!/usr/bin/env python3
"""InGaAs/InP SAGCM SPAD Simulator (1D center-axis)."""

import logging
import os

import numpy as np

from .core.constants import h, c
from .core.material import Material
from .core.absorption import InterpolatedAbsorption
from .core.layer import Layer
from .core.device import Device
from .simulator import SPADSimulator
from .utils._logging import get_logger, set_log_level
from .utils.loaders import load_materials, load_absorption, load_device
from .utils.plotter import get_plotter

_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
_plot_dir = os.path.normpath(os.path.join(_data_dir, "..", "plots", "spad"))
_optical_power = 1e-6
log = get_logger()


def build_sagcm_spad() -> Device:
    cfg = load_device(os.path.join(_data_dir, "device_sagcm.xml"))
    mat_data = load_materials(os.path.join(_data_dir, "materials.xml"))
    abs_data = load_absorption(os.path.join(_data_dir, "absorption.xml"))

    materials = {
        name: Material(data, absorption=InterpolatedAbsorption(abs_data.get(name)),
                       T=cfg.temperature)
        for name, data in mat_data.items()
    }

    layers = [
        Layer(
            thickness=lyr["thickness_cm"],
            doping_type=lyr["doping_type"],
            doping_A=lyr.get("doping_A", 0.0),
            doping_m=lyr.get("doping_m", 0.0),
            material=lyr["material"],
        )
        for lyr in cfg.layers
    ]
    return Device(layers, materials, no_of_nodes=cfg.nx)


def _find_breakdown(sim: SPADSimulator) -> float:
    Vbr, _ = sim.find_breakdown(V_start=0, V_max=80, V_step=1.0)
    if Vbr is None:
        raise ValueError("No breakdown detected")
    log.info(f"Vbr = {Vbr:.1f} V")
    return Vbr


def _plot_device_structure(sim: SPADSimulator) -> None:
    for i, lyr in enumerate(sim.device.layers):
        label = "i"
        if lyr.doping_A > 1e14:
            label = "n+" if lyr.is_donor else "p+"
        log.info(f"Layer {i}:  {label:3s}  {lyr.thickness * 1e4:.2f} um  "
                 f"{lyr.material:8s}  A={lyr.doping_A:.1e}  ({lyr.doping_type})")

    p = get_plotter("device_structure", plot_dir=_plot_dir)
    p.plot(sim.grid.x, sim.device.material.mat_name,
           sim.device.net_doping_on_grid, sim.device.net_doping_on_grid)


def _run_field_sweep(sim: SPADSimulator, Vbr: float) -> None:
    Vex_list = [0, 1, 2, 3, 4, 5]
    V_list = [Vbr + vex for vex in Vex_list]
    phi_list, E_list = [], []
    for V in V_list:
        phi, E, _ = sim.solve_poisson(float(V))
        phi_list.append(phi)
        E_list.append(E)
        log.info(f"Vex = {V - Vbr:.0f} V  phi_max = {phi.max():.1f}  "
                 f"|E|_max = {np.max(np.abs(E)):.2e}")

    get_plotter("potential_profile", plot_dir=_plot_dir).plot(
        sim.grid.x, np.array(phi_list), V_list)
    get_plotter("electric_field", plot_dir=_plot_dir).plot(
        sim.grid.x, np.array(E_list), V_list, Vbr=Vbr)


def _run_dark_current_sweep(sim: SPADSimulator, Vbr: float) -> None:
    Vex_range = np.linspace(0, 10, 11)
    I_dark, dcr = [], []
    for Vex in Vex_range:
        try:
            dc = sim.compute_dark_current(float(Vbr + Vex))
            I_dark.append(dc["I_dark"])
            dcr.append(dc["DCR"])
        except Exception:
            I_dark.append(np.nan)
            dcr.append(np.nan)

    I_dark, dcr = np.array(I_dark), np.array(dcr)
    mask = np.isfinite(I_dark)
    if np.any(mask):
        log.info(f"I_dark: {np.nanmin(I_dark):.2e} - {np.nanmax(I_dark):.2e} A")
        log.info(f"DCR:    {np.nanmin(dcr):.2e} - {np.nanmax(dcr):.2e} cps")
        get_plotter("dark_current", plot_dir=_plot_dir).plot(Vex_range[mask], I_dark[mask])
        get_plotter("dcr", plot_dir=_plot_dir).plot(Vex_range[mask], dcr[mask])


def _run_iv_characteristic(sim: SPADSimulator, Vbr: float) -> None:
    V_sweep = np.linspace(0, Vbr + 10, 61)
    I_dark, I_light = [], []
    for V in V_sweep:
        try:
            _, E, Pe, Ph, _, xr = sim.get_fields(float(V))
            dc = sim.compute_dark_current(float(V), E=E)
            I_dark.append(dc["I_dark"])
            photo = sim.compute_photocurrent(float(V), power=_optical_power,
                                             E=E, Pe=Pe, Ph=Ph, xr=xr)
            I_light.append(photo + dc["I_dark"])
        except Exception:
            I_dark.append(np.nan)
            I_light.append(np.nan)

    I_dark, I_light = np.array(I_dark), np.array(I_light)
    mask = np.isfinite(I_dark)
    if np.any(mask):
        get_plotter("iv_characteristic", plot_dir=_plot_dir).plot(
            V_sweep[mask], I_dark[mask], I_light=I_light[mask],
            optical_power=_optical_power)


def _run_pdp_spectrum(sim: SPADSimulator, Vbr: float) -> None:
    pdp_wavelengths = np.linspace(900, 1700, 41) * 1e-9
    pdp_spectra, vex_list_pdp = [], []
    for Vex in [1, 3, 5, 8]:
        try:
            _, E, _ = sim.solve_poisson(Vbr + Vex)
            Pe, Ph = sim._trigger_for_pdp(E)
            _, xr, _ = sim.depletion_width(Vbr + Vex)
            pdp = sim.compute_pdp_spectrum(
                pdp_wavelengths, float(Vex), material_name="InGaAs",
                E=E, Pe=Pe, Ph=Ph, xr=xr)
            pdp = np.clip(pdp, 0, 1)
            pdp_spectra.append(pdp)
            vex_list_pdp.append(Vex)
            log.info(f"  Vex = {Vex} V: PDP max = {np.max(pdp) * 100:.4f}%")
        except Exception as e:
            log.info(f"  Vex = {Vex} V: {e}")

    if pdp_spectra:
        get_plotter("pdp", plot_dir=_plot_dir).plot(
            pdp_wavelengths, np.array(pdp_spectra), vex_list_pdp)


def _run_pdp_vs_vex(sim: SPADSimulator, Vbr: float) -> None:
    dead_zone_layers, absorber = sim.pdp_model.find_absorber(
        sim.device.layers, "InGaAs")
    dead_zone = sum(l.thickness for l in dead_zone_layers)

    wl_apt = np.array([1100, 1310, 1550, 1610])
    vex_pts = np.linspace(0, 10, 11)
    pdp_dict = {lam: [] for lam in wl_apt}
    for Vex in vex_pts:
        try:
            _, E, _ = sim.solve_poisson(Vbr + Vex)
            Pe, Ph = sim._trigger_for_pdp(E)
            _, xr, _ = sim.depletion_width(Vbr + Vex)
            Ptr = Pe + Ph - Pe * Ph
            x_end = min(xr, dead_zone + absorber.thickness)
            mask = (sim.grid.x >= dead_zone) & (sim.grid.x <= x_end)
            xx = sim.grid.x[mask] - dead_zone
            for lam in wl_apt:
                trans = sim.pdp_model.dead_zone_transmission(lam * 1e-9, dead_zone_layers)
                pdp = sim.pdp_model.pdp_integral(
                    lam * 1e-9, xx, Ptr[mask], trans, sim.grid.dx,
                    material_name="InGaAs")
                pdp_dict[lam].append(pdp)
        except Exception:
            for lam in wl_apt:
                pdp_dict[lam].append(0.0)

    get_plotter("pdp_vs_vex", plot_dir=_plot_dir).plot(
        vex_pts, {lam: np.array(v) for lam, v in pdp_dict.items()},
        wavelengths_nm=wl_apt)


def _run_comprehensive_iv(sim: SPADSimulator, Vbr: float) -> None:
    dead_zone_layers, absorber = sim.pdp_model.find_absorber(
        sim.device.layers, "InGaAs")
    dead_zone = sum(l.thickness for l in dead_zone_layers)

    alpha_arr = np.array([
        sim.materials[lyr.material].absorption_coefficient(1310e-9)
        for lyr in sim.device.layers
    ])
    alpha_grid = np.zeros_like(sim.grid.x)
    xs = 0.0
    for lyr, alpha_val in zip(sim.device.layers, alpha_arr):
        xe = xs + lyr.thickness
        mask = (sim.grid.x >= xs - 1e-16) & (sim.grid.x <= xe + 1e-16)
        alpha_grid[mask] = alpha_val
        xs = xe

    Eph = h * c / 1310e-9
    phi_photon = _optical_power / (Eph * sim.detector_area)
    absorber_start = dead_zone
    absorber_end = dead_zone + absorber.thickness

    V_sweep = np.linspace(Vbr - 5, Vbr + 10, 11)
    I_dark, I_photo_prim, I_total, M_vals = [], [], [], []
    for V in V_sweep:
        try:
            _, E, Pe, Ph, _, xr = sim.get_fields(float(V))
            dc = sim.compute_dark_current(float(V), E=E)
            Ptr = Pe + Ph - Pe * Ph
            M = min(1.0 / (1.0 - float(np.max(Ptr)) + 1e-15), 10000.0)
            M_vals.append(M)
            I_dark.append(dc["I_dark"])

            J_pp = sim.pdp_model.photocurrent_density(
                sim.grid.x, alpha_grid, phi_photon, absorber_start, absorber_end)
            I_pp = float(np.trapezoid(J_pp, sim.grid.x) * sim.detector_area)
            I_photo_prim.append(I_pp)
            I_total.append(dc["I_dark"] + I_pp * M)
        except Exception:
            I_dark.append(np.nan)
            I_photo_prim.append(np.nan)
            I_total.append(np.nan)
            M_vals.append(np.nan)

    I_dark, I_pp, I_total, M_vals = [np.array(a) for a in
                                      (I_dark, I_photo_prim, I_total, M_vals)]
    mask = np.isfinite(I_dark)
    if np.any(mask):
        get_plotter("comprehensive_iv", plot_dir=_plot_dir).plot(
            V_sweep[mask], I_dark[mask],
            I_photo_primary=I_pp[mask], I_total_illuminated=I_total[mask],
            gain=M_vals[mask], Vbr=Vbr)


def main() -> None:
    set_log_level(logging.INFO)
    sim = SPADSimulator(build_sagcm_spad())

    _plot_device_structure(sim)
    Vbr = _find_breakdown(sim)
    _run_field_sweep(sim, Vbr)
    _run_dark_current_sweep(sim, Vbr)
    _run_iv_characteristic(sim, Vbr)
    _run_pdp_spectrum(sim, Vbr)
    _run_pdp_vs_vex(sim, Vbr)
    _run_comprehensive_iv(sim, Vbr)

    log.info("\n  Plots saved to %s/", _plot_dir)


if __name__ == "__main__":
    main()
