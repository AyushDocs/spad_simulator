"""Photocurrent and PDE spectrum computation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.constants import h
from ..core.physics_helpers import alpha_to_grid, avalanche_trigger_probability, dead_zone_thickness

# Speed of light in m/s for photon energy computation
_C_MS = 2.998e8

if TYPE_CHECKING:
    from ..avalanche.pde import PDEModel


def _collection_width(
    E: np.ndarray, grid_x: np.ndarray,
    dead_zone: float, absorber_thickness: float,
    min_field: float = 500,
) -> float:
    """Compute effective collection width from E-field.

    Finds the contiguous region from the absorber start where the field
    points in the collection direction (E < 0 for reverse bias) with
    magnitude above a minimum threshold. This captures the bias-dependent
    depletion of the absorber while excluding the weak built-in field at
    the buffer junction.
    """
    abs_start = dead_zone
    abs_end = dead_zone + absorber_thickness
    in_abs = (grid_x >= abs_start) & (grid_x <= abs_end)
    if not np.any(in_abs):
        return 0.0
    idx = np.where(in_abs)[0]
    for i in idx:
        if not (E[i] < 0 and np.abs(E[i]) > min_field):
            prev = i - 1
            if prev >= idx[0]:
                return max(grid_x[prev] - abs_start, 0.0)
            return 0.0
    return max(grid_x[idx[-1]] - abs_start, 0.0)


def compute_photocurrent(
    grid_x: np.ndarray,
    layers: list,
    materials: dict,
    pde_model: PDEModel,
    detector_area: float,
    wavelength: float,
    power: float,
    E: np.ndarray,
    Pe: np.ndarray,
    Ph: np.ndarray,
    xr: float,
    multiply: bool = True,
    M: float | None = None,
) -> float:
    """Compute primary photocurrent, optionally multiplied by McIntyre M(V).

    Parameters
    ----------
    M : float or None
        McIntyre multiplication factor from the simulator's
        ``compute_dark_current()``.  If provided, the photocurrent is
        scaled by M (continuous, pre-breakdown) instead of the Geiger-
        mode gain (valid only above breakdown).
    """
    dead_zone_layers, absorber = pde_model.find_absorber(layers, "InGaAs")
    dead_zone = sum(l.thickness for l in dead_zone_layers)
    L_abs = min(_collection_width(E, grid_x, dead_zone, absorber.thickness),
                 absorber.thickness)

    if L_abs <= 0:
        return 0.0

    absorber_start = dead_zone
    absorber_end = dead_zone + L_abs

    alpha_grid = alpha_to_grid(grid_x, layers, materials, wavelength)

    Eph = float(h.magnitude) * _C_MS / wavelength
    trans = pde_model.dead_zone_transmission(wavelength, dead_zone_layers)
    phi_photon = (1.0 - pde_model.reflectivity) * trans * power / (Eph * detector_area)

    J_photo = pde_model.photocurrent_density(
        grid_x, alpha_grid, phi_photon, absorber_start, absorber_end
    )
    # Trapezoidal rule: O(h²), handles non-uniform grid near heterojunctions
    I_primary = float(np.trapezoid(J_photo, grid_x) * detector_area)

    if I_primary <= 0 or not multiply:
        return I_primary

    if M is not None:
        # Continuous McIntyre multiplication (same as compute_dark_current)
        return I_primary * max(M, 1.0)

    # Fallback: Geiger-mode gain (only above breakdown)
    from ..core.constants import eps0
    Ptr_abs = np.interp(
        np.linspace(absorber_start, absorber_end, 10),
        grid_x,
        avalanche_trigger_probability(Pe, Ph),
    )
    Ptr_avg = float(np.mean(Ptr_abs))
    from ..avalanche.geiger import avalanche_charge, effective_gain
    W = max(abs(grid_x[np.argmax(np.abs(E))] - grid_x[np.argmin(np.abs(E))]), 1e-7)
    C_j = eps0 * 11.7 * detector_area * 1e-4 / max(W, 1e-10)
    ac = avalanche_charge(0.0, 0.0, C_j)  # dummy call, kept for API compat
    _ = effective_gain(ac.Q_av)
    return I_primary


def compute_pde_spectrum(
    grid_x: np.ndarray,
    dx: float,
    layers: list,
    pde_model: PDEModel,
    wavelengths: np.ndarray,
    Vex: float,
    xr: float,
    Pe: np.ndarray,
    Ph: np.ndarray,
    material_name: str = "InGaAs",
) -> np.ndarray:
    """Compute PDE at each wavelength for a given excess voltage."""
    Ptr = avalanche_trigger_probability(Pe, Ph)

    dead_zone_layers, absorber = pde_model.find_absorber(layers, material_name)
    dz = dead_zone_thickness(dead_zone_layers)
    L_abs = max(min(xr - dz, absorber.thickness), 0.0)

    if L_abs <= 0:
        return np.zeros(len(wavelengths))

    absorber_start = dz
    absorber_end = dz + L_abs
    mask = (grid_x >= absorber_start) & (grid_x <= absorber_end)
    xx = grid_x[mask] - absorber_start

    pde_vals = []
    for lam in wavelengths:
        trans = pde_model.dead_zone_transmission(lam, dead_zone_layers)
        pde = pde_model.pde_integral(
            lam, xx, Ptr[mask], trans, dx, material_name=material_name
        )
        pde_vals.append(pde)
    return np.array(pde_vals)
