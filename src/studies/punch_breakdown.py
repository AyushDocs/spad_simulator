"""Punch-through and breakdown voltage vs device parameter sweeps."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ..utils.loaders import PlotConfig
from . import _config as _cfg
from ._config import PLOT_DIR

log = get_logger()


# ---------------------------------------------------------------------------
#  Helpers — delegate to shared utilities in _config
# ---------------------------------------------------------------------------

def _absorber_start(layers: list) -> float:
    """Cumulative thickness at the start of the InGaAs absorber (cm)."""
    pos = 0.0
    for lyr in layers:
        if lyr.material == "InGaAs":
            return pos
        pos += lyr.thickness
    return pos


def _find_punchthrough(sim: SPADSimulator, V_max: float = 150.0,
                       V_step: float = 1.0) -> float:
    """Find punch-through voltage (depletion first reaches absorber)."""
    x_target = _absorber_start(sim.device.layers)
    for V in np.arange(0.0, V_max + V_step, V_step):
        try:
            xl, xr, _ = sim.depletion_width(float(V))
            if xr >= x_target:
                return float(V)
        except Exception:
            pass
    return V_max


def _find_breakdown(sim: SPADSimulator,
                    V_start: float = 40.0, V_max: float = 150.0) -> float:
    """Find breakdown voltage for the current device state."""
    try:
        vbr, _ = sim.find_breakdown(V_start=V_start, V_max=V_max,
                                    V_step=1.0, force=True)
        return float(vbr) if vbr is not None else V_max
    except Exception:
        return V_max


# ---------------------------------------------------------------------------
#  Mutators (same pattern as param_sweep.py)
# ---------------------------------------------------------------------------

def _set_thickness(layers: list, idx: int, val: float) -> None:
    _cfg.mutate_thickness(layers, idx, val)


def _set_doping(layers: list, idx: int, val: float) -> None:
    _cfg.mutate_doping(layers, idx, val)


def _set_charge_sheet(layers: list, idx: int, sigma: float,
                      thin_t: float = 0.05e-4) -> None:
    """Set charge sheet to thickness *thin_t* with doping = sigma/thin_t."""
    _cfg.mutate_thickness(layers, idx, thin_t)
    _cfg.mutate_doping(layers, idx, sigma / thin_t)


# ---------------------------------------------------------------------------
#  Parameter lookup helpers
# ---------------------------------------------------------------------------

def _idx_by_material(layers: list, mat: str) -> int | None:
    return _cfg.layer_index_by_material(layers, mat)


def _idx_by_material_and_doping(layers: list, mat: str, dtype: str,
                                 min_d: float = 0.0,
                                 max_d: float = float("inf")) -> int | None:
    return _cfg.layer_index_by_material_and_doping(layers, mat, dtype, min_d, max_d)


def _idx_by_doping_type(layers: list, dtype: str) -> int | None:
    return _cfg.layer_index_by_doping_type(layers, dtype)


# ---------------------------------------------------------------------------
#  Single-parameter sweep
# ---------------------------------------------------------------------------

def _sweep_param(sim: SPADSimulator, param_values: np.ndarray,
                 mutate_fn, layer_idx: int | None = None,
                 use_nt: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Sweep a device parameter, return (V_pt, V_br) arrays."""
    orig_layers = list(sim.device.layers)
    V_pt_list: list[float] = []
    V_br_list: list[float] = []

    for val in param_values:
        try:
            if use_nt:
                sim.set_nt(float(val))
            else:
                layers = list(orig_layers)
                mutate_fn(layers, layer_idx, val)
                sim.set_layers(layers)
            vpt = _find_punchthrough(sim)
            vbr = _find_breakdown(sim)
            V_pt_list.append(vpt)
            V_br_list.append(vbr)
        except Exception:
            V_pt_list.append(np.nan)
            V_br_list.append(np.nan)

    if not use_nt:
        sim.set_layers(orig_layers)

    return np.array(V_pt_list), np.array(V_br_list)


# ---------------------------------------------------------------------------
#  Panels
# ---------------------------------------------------------------------------

def _panel_absorption(sim: SPADSimulator) -> dict:
    idx = _idx_by_material(sim.device.layers, "InGaAs")
    vals = np.linspace(1.0, 5.0, 9) * 1e-4
    V_pt, V_br = _sweep_param(sim, vals, _set_thickness, layer_idx=idx)
    return {"values": vals * 1e4, "V_pt": V_pt, "V_br": V_br,
            "xlabel": "D$_{\\mathrm{abs}}$ (µm)",
            "label": "(a) Absorption thickness"}


def _panel_multiplication(sim: SPADSimulator) -> dict:
    idx = _idx_by_material_and_doping(sim.device.layers, "InP", "donor",
                                       max_d=1e16)
    vals = np.linspace(0.2, 2.0, 10) * 1e-4
    V_pt, V_br = _sweep_param(sim, vals, _set_thickness, layer_idx=idx)
    return {"values": vals * 1e4, "V_pt": V_pt, "V_br": V_br,
            "xlabel": "D$_{\\mathrm{mul}}$ (µm)",
            "label": "(b) Multiplication thickness"}


def _panel_trap(sim: SPADSimulator) -> dict:
    vals = np.array([0.0, 1e15, 5e15, 1e16])
    V_pt, V_br = _sweep_param(sim, vals, mutate_fn=None, use_nt=True)
    return {"values": vals, "V_pt": V_pt, "V_br": V_br,
            "xlabel": "N$_{\\mathrm{trap}}$ (cm$^{-3}$)",
            "label": "(c) Trap concentration"}


def _panel_pdop(sim: SPADSimulator) -> dict:
    idx = _idx_by_doping_type(sim.device.layers, "acceptor")
    vals = np.logspace(17, 19, 9)
    V_pt, V_br = _sweep_param(sim, vals, _set_doping, layer_idx=idx)
    return {"values": vals, "V_pt": V_pt, "V_br": V_br,
            "xlabel": "P-InP doping (cm$^{-3}$)",
            "label": "(d) P-InP doping"}


def _panel_charge(sim: SPADSimulator) -> dict:
    idx = _idx_by_material_and_doping(sim.device.layers, "InP", "donor",
                                       min_d=1e16)
    charge_layer = sim.device.layers[idx]
    t_charge = charge_layer.thickness

    # Sweep σ in a range centred near the nominal σ₀ = 2.5e12 cm⁻².
    # The low-σ edge (σ < 2.0e12) produces unrealistically high V_br because
    # the 0.25 µm charge sheet is too thick to behave as a δ-function sheet
    # charge at very low bulk doping, so we restrict to σ ≥ 2.0e12.
    sigma_vals = np.linspace(2.0, 3.5, 9) * 1e12
    n_vals = sigma_vals / t_charge
    V_pt, V_br = _sweep_param(sim, n_vals, _set_doping, layer_idx=idx)
    return {"values": sigma_vals, "V_pt": V_pt, "V_br": V_br,
            "xlabel": "Sheet charge density (cm$^{-2}$)",
            "label": "(e) Surface charge density"}


def _panel_muldop(sim: SPADSimulator) -> dict:
    idx = _idx_by_material_and_doping(sim.device.layers, "InP", "donor",
                                       max_d=1e16)
    vals = np.logspace(14, 16, 9)
    V_pt, V_br = _sweep_param(sim, vals, _set_doping, layer_idx=idx)
    return {"values": vals, "V_pt": V_pt, "V_br": V_br,
            "xlabel": "Mul. doping (cm$^{-3}$)",
            "label": "(f) Multiplication doping"}


# ---------------------------------------------------------------------------
#  Orchestrator
# ---------------------------------------------------------------------------

def run_punch_breakdown_sweep(sim: SPADSimulator, Vbr: float,
                              plot_cfg: PlotConfig | None = None) -> None:
    """Run all 6 parameter sweeps and plot V_pt / V_br together."""
    if plot_cfg and not plot_cfg.is_enabled("punch_breakdown_sweep"):
        return
    log.info("Punch-through / breakdown voltage sweeps")

    panels = [
        _panel_absorption(sim),
        _panel_multiplication(sim),
        _panel_trap(sim),
        _panel_pdop(sim),
        _panel_charge(sim),
        _panel_muldop(sim),
    ]

    for p in panels:
        label = p.get("label", "")
        n_ok = int(np.sum(np.isfinite(p["V_pt"]) & np.isfinite(p["V_br"])))
        log.info("  %s: %d/%d ok", label, n_ok, len(p["values"]))

    get_plotter("punch_breakdown_sweep", plot_dir=PLOT_DIR).plot(panels)
