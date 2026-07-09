"""Photocurrent and PDP spectrum computation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.constants import h, c
from ..core.physics_helpers import alpha_to_grid, combined_trigger_probability, dead_zone_thickness

if TYPE_CHECKING:
    from ..core.device import Device
    from ..core.grid import Grid1D
    from ..avalanche.pdp import PDPModel


def compute_photocurrent(
    grid_x: np.ndarray,
    layers: list,
    materials: dict,
    pdp_model: PDPModel,
    detector_area: float,
    wavelength: float,
    power: float,
    E: np.ndarray,
    Pe: np.ndarray,
    Ph: np.ndarray,
    xr: float,
) -> float:
    """Compute primary photocurrent multiplied by avalanche gain."""
    dead_zone_layers, absorber = pdp_model.find_absorber(layers, "InGaAs")
    dead_zone = sum(l.thickness for l in dead_zone_layers)
    L_abs = max(min(xr - dead_zone, absorber.thickness), 0.0)

    if L_abs <= 0:
        return 0.0

    absorber_start = dead_zone
    absorber_end = dead_zone + L_abs

    alpha_grid = alpha_to_grid(grid_x, layers, materials, wavelength)

    Eph = h * c / wavelength
    phi_photon = power / (Eph * detector_area)

    J_photo = pdp_model.photocurrent_density(
        grid_x, alpha_grid, phi_photon, absorber_start, absorber_end
    )
    # Trapezoidal rule: O(h²), handles non-uniform grid near heterojunctions
    I_primary = float(np.trapezoid(J_photo, grid_x) * detector_area)

    if I_primary <= 0:
        return 0.0

    Ptr_abs = np.interp(
        np.linspace(absorber_start, absorber_end, 10),
        grid_x,
        combined_trigger_probability(Pe, Ph),
    )
    Ptr_avg = float(np.mean(Ptr_abs))
    M_raw = 1.0 / (1.0 - Ptr_avg + 1e-15)
    M = min(M_raw, 10000.0)
    return I_primary * M


def compute_pdp_spectrum(
    grid_x: np.ndarray,
    dx: float,
    layers: list,
    pdp_model: PDPModel,
    wavelengths: np.ndarray,
    Vex: float,
    xr: float,
    Pe: np.ndarray,
    Ph: np.ndarray,
    material_name: str = "InGaAs",
) -> np.ndarray:
    """Compute PDP at each wavelength for a given excess voltage."""
    Ptr = combined_trigger_probability(Pe, Ph)

    dead_zone_layers, absorber = pdp_model.find_absorber(layers, material_name)
    dz = dead_zone_thickness(dead_zone_layers)
    L_abs = max(min(xr - dz, absorber.thickness), 0.0)

    if L_abs <= 0:
        return np.zeros(len(wavelengths))

    absorber_start = dz
    absorber_end = dz + L_abs
    mask = (grid_x >= absorber_start) & (grid_x <= absorber_end)
    xx = grid_x[mask] - absorber_start

    pdp_vals = []
    for lam in wavelengths:
        trans = pdp_model.dead_zone_transmission(lam, dead_zone_layers)
        pdp = pdp_model.pdp_integral(
            lam, xx, Ptr[mask], trans, dx, material_name=material_name
        )
        pdp_vals.append(pdp)
    return np.array(pdp_vals)
