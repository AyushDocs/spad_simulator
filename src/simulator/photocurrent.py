"""Photocurrent and PDP spectrum computation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.constants import h, c
from ..core.physics_helpers import alpha_to_grid, avalanche_trigger_probability, dead_zone_thickness

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
    multiply: bool = True,
    V_bias: float = 80.0,
    V_br: float = 78.0,
) -> float:
    """Compute primary photocurrent (optionally multiplied by Geiger-mode gain)."""
    dead_zone_layers, absorber = pdp_model.find_absorber(layers, "InGaAs")
    dead_zone = sum(l.thickness for l in dead_zone_layers)
    L_abs = max(min(xr - dead_zone, absorber.thickness), 0.0)

    if L_abs <= 0:
        return 0.0

    absorber_start = dead_zone
    absorber_end = dead_zone + L_abs

    alpha_grid = alpha_to_grid(grid_x, layers, materials, wavelength)

    Eph = h * c / wavelength
    trans = pdp_model.dead_zone_transmission(wavelength, dead_zone_layers)
    phi_photon = (1.0 - pdp_model.reflectivity) * trans * power / (Eph * detector_area)

    J_photo = pdp_model.photocurrent_density(
        grid_x, alpha_grid, phi_photon, absorber_start, absorber_end
    )
    # Trapezoidal rule: O(h²), handles non-uniform grid near heterojunctions
    I_primary = float(np.trapezoid(J_photo, grid_x) * detector_area)

    if I_primary <= 0 or not multiply:
        return I_primary

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
    ac = avalanche_charge(V_bias, V_br, C_j)
    M = effective_gain(ac.Q_av) if ac.triggered else 1.0
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
    Ptr = avalanche_trigger_probability(Pe, Ph)

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
