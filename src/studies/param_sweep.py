"""Parameter-sweep and voltage-sweep automation framework.

Provides reusable routines for:
- Voltage (I-V) sweeps with current decomposition
- Single-parameter sweeps (layer thickness, doping) vs breakdown voltage
- Result export to CSV
"""

from __future__ import annotations

import csv
import os
from typing import Callable

import numpy as np

from ..simulator import SPADSimulator
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ..utils.loaders import PlotConfig
from . import _config as _cfg
from ._config import PLOT_DIR, R_Q

log = get_logger("param_sweep")


# ---------------------------------------------------------------------------
#  I-V sweep
# ---------------------------------------------------------------------------

def run_iv_sweep(
    sim: SPADSimulator,
    Vbr: float,
    V_start: float = 0.0,
    V_stop: float | None = None,
    V_step: float = 1.0,
    decompose: bool = True,
    plot: bool = True,
    plot_cfg: PlotConfig | None = None,
) -> dict:
    """Sweep voltage from ``V_start`` to ``V_stop`` and compute dark current.

    Parameters
    ----------
    sim : SPADSimulator
    Vbr : float
        Breakdown voltage (V).
    V_start, V_stop : float
        Bias range.  If ``V_stop`` is *None* it defaults to ``Vbr + 30``.
    V_step : float
        Voltage increment (V).
    decompose : bool
        If True, decompose current into SRH / BTBT / TAT contributions.
    plot : bool
        If True, generate an I-V plot with components.

    Returns
    -------
    dict with keys ``V``, ``I_dark``, ``M``, and per-mechanism arrays.
    """
    if plot_cfg and not plot_cfg.is_enabled("iv_sweep"):
        return {}
    if V_stop is None:
        V_stop = Vbr + 30.0

    V_range = np.arange(V_start, V_stop, V_step)
    I_dark: list[float] = []
    M_vals: list[float] = []
    I_srh: list[float] = []
    I_btbt: list[float] = []
    I_tat: list[float] = []

    for V in V_range:
        try:
            dc = sim.compute_dark_current(float(V))
            I_dark_val = dc.get("I_dark", np.nan)
            if float(V) >= Vbr:
                I_dark_val += (float(V) - Vbr) / R_Q
            I_dark.append(I_dark_val)
            M_vals.append(dc.get("M", 1.0))
            I_srh.append(np.nan)
            I_btbt.append(np.nan)
            I_tat.append(np.nan)
        except Exception:
            I_dark.append(np.nan)
            M_vals.append(np.nan)
            I_srh.append(np.nan)
            I_btbt.append(np.nan)
            I_tat.append(np.nan)

    arr = lambda v: np.array(v)  # noqa: E731
    result = {
        "V": V_range,
        "I_dark": arr(I_dark),
        "M": arr(M_vals),
        "I_srh": arr(I_srh),
        "I_btbt": arr(I_btbt),
        "I_tat": arr(I_tat),
    }

    if plot:
        mask = np.isfinite(arr(I_dark))
        if np.any(mask):
            get_plotter("iv_sweep", plot_dir=PLOT_DIR).plot(
                V_range[mask], arr(I_dark)[mask],
                I_srh=arr(I_srh)[mask] if decompose else None,
                I_btbt=arr(I_btbt)[mask] if decompose else None,
                I_tat=arr(I_tat)[mask] if decompose else None,
                Vbr=Vbr,
            )

    return result


# ---------------------------------------------------------------------------
#  Device parameter sweep helpers
# ---------------------------------------------------------------------------

_LayerMutator = Callable[[list, int, float], None]


def _mutate_thickness(layers: list, idx: int, value: float) -> None:
    """Set layer ``idx`` thickness to ``value`` (cm)."""
    _cfg.mutate_thickness(layers, idx, value)


def _mutate_doping(layers: list, idx: int, value: float) -> None:
    """Set layer ``idx`` ``doping_A`` to ``value`` (cm⁻³)."""
    _cfg.mutate_doping(layers, idx, value)


def _mutate_doping_type(layers: list, idx: int, value: str) -> None:
    """Set layer ``idx`` ``doping_type`` to ``value`` (donor/acceptor)."""
    from copy import deepcopy
    old = layers[idx]
    layers[idx] = deepcopy(old)
    object.__setattr__(layers[idx], "doping_type", str(value))


_PARAM_MUTATORS: dict[str, _LayerMutator] = {
    "thickness": _mutate_thickness,
    "doping": _mutate_doping,
    "doping_type": _mutate_doping_type,
}


def _describe_layer(sim: SPADSimulator, idx: int) -> str:
    lyr = sim.device.layers[idx]
    return f"{lyr.material} {lyr.thickness*1e4:.2f}um  {lyr.doping_A:.1e}cm⁻³ {lyr.doping_type}"


def sweep_parameter(
    sim: SPADSimulator,
    param: str,
    layer_idx: int,
    values: np.ndarray,
    *,
    find_Vbr_kw: dict | None = None,
    label: str | None = None,
    plot: bool = True,
    export_csv: str | None = None,
) -> dict[str, np.ndarray]:
    """Vary a single device parameter and observe the effect on Vbr.

    Parameters
    ----------
    sim : SPADSimulator
    param : str
        One of ``"thickness"``, ``"doping"``, ``"doping_type"``.
    layer_idx : int
        Index into ``sim.device.layers``.
    values : np.ndarray
        Parameter values to sweep over.
    find_Vbr_kw : dict, optional
        Keyword arguments forwarded to ``sim.find_breakdown()``.
    label : str, optional
        Label for the x-axis (default: auto-generated).
    plot : bool
        If True, generate a plot.
    export_csv : str, optional
        If given, save results as CSV at this path.

    Returns
    -------
    dict with keys ``values``, ``Vbr`` (float or NaN for each value).
    """
    mutator = _PARAM_MUTATORS.get(param)
    if mutator is None:
        raise ValueError(f"Unknown parameter '{param}'. Choose from {list(_PARAM_MUTATORS)}")

    if find_Vbr_kw is None:
        find_Vbr_kw = {"V_start": 40, "V_max": 150, "V_step": 1.0}

    original_layers = list(sim.device.layers)
    original_Vbr = None
    try:
        original_Vbr, _ = sim.find_breakdown(**find_Vbr_kw, force=True)
    except Exception:
        pass

    Vbr_vals: list[float] = []
    for val in values:
        try:
            layers = list(original_layers)
            mutator(layers, layer_idx, val)
            sim.set_layers(layers)
            vbr, _ = sim.find_breakdown(**find_Vbr_kw, force=True)
            Vbr_vals.append(float(vbr) if vbr is not None else np.nan)
        except Exception:
            Vbr_vals.append(np.nan)

    sim.set_layers(original_layers)

    result = {"values": np.asarray(values), "Vbr": np.array(Vbr_vals)}

    desc = label or f"{param} of layer {layer_idx} ({_describe_layer(sim, layer_idx).replace('  ', ', ')})"

    if plot:
        mask = np.isfinite(result["Vbr"])
        if np.any(mask):
            get_plotter("param_sweep", plot_dir=PLOT_DIR).plot(
                result["values"][mask], result["Vbr"][mask],
                xlabel=desc, ylabel="Breakdown Voltage (V)",
                title=f"Vbr vs {desc}",
                fname=f"Vbr_vs_{param}_layer{layer_idx}.png",
            )

    if export_csv:
        _save_csv(export_csv,
                  {"values": result["values"], "Vbr": result["Vbr"]},
                  ["values", "Vbr"])

    return result


def sweep_voltage_and_parameter(
    sim: SPADSimulator,
    Vbr: float,
    param: str,
    layer_idx: int,
    param_values: np.ndarray,
    V_start: float = 0.0,
    V_stop: float | None = None,
    V_step: float = 2.0,
    decompose: bool = True,
    plot: bool = True,
    export_csv: str | None = None,
) -> dict:
    """Sweep a device parameter and at each value run a full I-V sweep.

    Useful for e.g. ``[I-V curves for different absorption thicknesses]``.

    Returns a dict with keys ``param_values`` (list), ``V_range`` (list),
    and ``I_dark`` (2-D list, shape ``(N_param, N_V)``).
    """
    if V_stop is None:
        V_stop = Vbr + 30.0
    V_range = np.arange(V_start, V_stop, V_step)

    original_layers = list(sim.device.layers)
    mutator = _PARAM_MUTATORS[param]

    I_dark_2d: list[list[float]] = []
    param_vals_used: list[float] = []

    for val in param_values:
        try:
            layers = list(original_layers)
            mutator(layers, layer_idx, val)
            sim.set_layers(layers)
            iv = run_iv_sweep(sim, Vbr, V_start, V_stop, V_step,
                              decompose=decompose, plot=False)
            I_dark_2d.append(iv["I_dark"].tolist())
            param_vals_used.append(float(val))
        except Exception:
            param_vals_used.append(float(val))
            I_dark_2d.append([np.nan] * len(V_range))

    sim.set_layers(original_layers)

    result = {
        "param_values": np.array(param_vals_used),
        "V_range": V_range,
        "I_dark": np.array(I_dark_2d),
    }

    if plot:
        get_plotter("param_sweep_iv", plot_dir=PLOT_DIR).plot(
            result["param_values"], result["V_range"], result["I_dark"],
            param_label=param,
        )

    if export_csv:
        rows = [["param_value"] + list(V_range)]
        for pv, irow in zip(param_vals_used, I_dark_2d):
            rows.append([pv] + irow)
        _write_csv_rows(export_csv, rows)

    return result


# ---------------------------------------------------------------------------
#  Parameter sweeps matching the paper (Table 4)
# ---------------------------------------------------------------------------

def sweep_absorption_thickness(sim: SPADSimulator, Vbr: float,
                               plot_cfg: PlotConfig | None = None) -> dict:
    """Vary InGaAs absorber thickness and report Vbr vs thickness."""
    if plot_cfg and not plot_cfg.is_enabled("sweep_absorption_thickness"):
        return {}
    idx = _layer_index_by_material(sim, "InGaAs")
    if idx is None:
        log.warning("No InGaAs layer found in device stack")
        return {"values": np.array([]), "Vbr": np.array([])}
    widths = np.linspace(1.0, 5.0, 9) * 1e-4
    return sweep_parameter(sim, "thickness", idx, widths,
                           label="Absorption thickness (µm)")


def sweep_multiplication_thickness(sim: SPADSimulator, Vbr: float,
                                   plot_cfg: PlotConfig | None = None) -> dict:
    """Vary intrinsic InP multiplication-layer thickness.

    Uses finer voltage stepping (0.1 V) to reduce quantization noise in
    the breakdown voltage search.  Residual oscillations from uniform-grid
    boundary snapping are an inherent limitation of the piecewise-constant
    material assignment on a fixed mesh.
    """
    if plot_cfg and not plot_cfg.is_enabled("sweep_multiplication_thickness"):
        return {}
    idx = _layer_index_by_material_and_doping(sim, "InP", "donor", max_doping=1e16)
    if idx is None:
        log.warning("No intrinsic InP multiplication layer found")
        return {"values": np.array([]), "Vbr": np.array([])}
    widths = np.linspace(0.2, 2.0, 10) * 1e-4
    return sweep_parameter(
        sim, "thickness", idx, widths,
        find_Vbr_kw={"V_start": 40, "V_max": 150, "V_step": 0.1},
        label="Multiplication thickness (µm)",
    )


def sweep_charge_density(sim: SPADSimulator, Vbr: float,
                         plot_cfg: PlotConfig | None = None) -> dict:
    """Vary charge-sheet doping."""
    if plot_cfg and not plot_cfg.is_enabled("sweep_charge_density"):
        return {}
    idx = _layer_index_by_material_and_doping(sim, "InP", "donor", min_doping=1e16)
    if idx is None:
        log.warning("No charge layer found")
        return {"values": np.array([]), "Vbr": np.array([])}
    densities = np.logspace(16, 18, 9)
    return sweep_parameter(sim, "doping", idx, densities,
                           label="Charge-layer doping (cm⁻³)")


def sweep_p_layer_doping(sim: SPADSimulator, Vbr: float,
                         plot_cfg: PlotConfig | None = None) -> dict:
    """Vary p+ contact layer doping."""
    if plot_cfg and not plot_cfg.is_enabled("sweep_p_layer_doping"):
        return {}
    idx = _layer_index_by_doping_type(sim, "acceptor")
    if idx is None:
        log.warning("No p-type layer found")
        return {"values": np.array([]), "Vbr": np.array([])}
    densities = np.logspace(17, 19, 9)
    return sweep_parameter(sim, "doping", idx, densities,
                           label="P-layer doping (cm⁻³)")


# ---------------------------------------------------------------------------
#  Internal helpers — thin wrappers over shared utilities in _config
# ---------------------------------------------------------------------------

def _layer_index_by_material(sim: SPADSimulator, mat: str) -> int | None:
    return _cfg.layer_index_by_material(list(sim.device.layers), mat)


def _layer_index_by_material_and_doping(
    sim: SPADSimulator, mat: str, dtype: str,
    min_doping: float = 0.0, max_doping: float = float("inf"),
) -> int | None:
    return _cfg.layer_index_by_material_and_doping(
        list(sim.device.layers), mat, dtype, min_doping, max_doping)


def _layer_index_by_doping_type(sim: SPADSimulator, dtype: str) -> int | None:
    return _cfg.layer_index_by_doping_type(list(sim.device.layers), dtype)


def _save_csv(path: str, data: dict[str, np.ndarray], cols: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in zip(*[data[c] for c in cols]):
            w.writerow(row)
    log.info("saved  %s", path)


def _write_csv_rows(path: str, rows: list[list]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    log.info("saved  %s", path)
