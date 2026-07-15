"""Shared constants and helpers for study runners."""
from __future__ import annotations

import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PLOT_DIR = os.path.normpath(os.path.join(DATA_DIR, "..", "plots", "spad"))
OPTICAL_POWER = 1e-6
R_Q = 2e5  # quenching resistor (Ohms)

# Device physics constants (CGS units consistent with grid)
EG_INP = 1.35       # InP bandgap at 300 K (eV)
FIELD_THRESHOLD = 1e4  # V/cm — minimum |E| for ionization activity
X_MULT_MAX = 4e-4   # cm — end of InP multiplication layer (~4 μm)
M_MAX = 5000.0      # smooth cap on avalanche multiplication factor


# ---------------------------------------------------------------------------
#  Layer helpers (used by param_sweep and punch_breakdown)
# ---------------------------------------------------------------------------

def layer_index_by_material(layers: list, mat: str) -> int | None:
    """Return index of first layer matching *mat*, or None."""
    for i, lyr in enumerate(layers):
        if lyr.material == mat:
            return i
    return None


def layer_index_by_material_and_doping(
    layers: list, mat: str, dtype: str,
    min_doping: float = 0.0, max_doping: float = float("inf"),
) -> int | None:
    """Return index of first layer matching material, doping type and range."""
    for i, lyr in enumerate(layers):
        if (lyr.material == mat and lyr.doping_type == dtype
                and min_doping <= lyr.doping_A <= max_doping):
            return i
    return None


def layer_index_by_doping_type(layers: list, dtype: str) -> int | None:
    """Return index of first layer with the given doping type."""
    for i, lyr in enumerate(layers):
        if lyr.doping_type == dtype:
            return i
    return None


def mutate_thickness(layers: list, idx: int, value: float) -> None:
    """Set layer ``idx`` thickness to ``value`` (cm) without mutating neighbours."""
    from copy import deepcopy
    old = layers[idx]
    layers[idx] = deepcopy(old)
    object.__setattr__(layers[idx], "thickness", float(value))


def mutate_doping(layers: list, idx: int, value: float) -> None:
    """Set layer ``idx`` ``doping_A`` to ``value`` (cm⁻³)."""
    from copy import deepcopy
    old = layers[idx]
    layers[idx] = deepcopy(old)
    object.__setattr__(layers[idx], "doping_A", float(value))


# ---------------------------------------------------------------------------
#  Absorption-weighted trigger probability
# ---------------------------------------------------------------------------

def absorption_weighted_trigger(sim, E: np.ndarray) -> float:
    """Absorption-weighted spatial average of trigger probability.

    Uses the InGaAs absorption coefficient at 1550 nm to weight the
    trigger probability over the multiplication region.  This is the
    standard weighting used across all study modules.
    """
    x = sim.grid.x
    mult_mask = (np.abs(E) > FIELD_THRESHOLD) & (x < X_MULT_MAX)
    if not np.any(mult_mask):
        return 0.0
    try:
        alpha_opt = sim.materials["InGaAs"].absorption_coefficient(1550e-9)
    except (KeyError, AttributeError):
        alpha_opt = 7500.0  # cm⁻¹ fallback for InGaAs at 1550 nm
    w = alpha_opt * np.exp(-alpha_opt * x[mult_mask])
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return 0.0
    alpha = sim.ionization.effective_alpha_n(np.abs(E), Eg=EG_INP)
    beta = sim.ionization.effective_alpha_p(np.abs(E), Eg=EG_INP)
    Pe, Ph = sim.trigger.solve(E, alpha, beta, x)
    Ptr = Pe + Ph - Pe * Ph
    return float(np.sum(Ptr[mult_mask] * w) / w_sum)
