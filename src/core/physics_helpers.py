"""Shared physics helpers — eliminate formula duplication."""
from __future__ import annotations

import numpy as np


def avalanche_trigger_probability(Pe: np.ndarray, Ph: np.ndarray) -> np.ndarray:
    """Combined trigger probability: Pe ∪ Ph (independent events)."""
    return Pe + Ph - Pe * Ph


def dead_zone_thickness(dead_zone_layers: list) -> float:
    """Total dead-zone thickness from a list of layers."""
    return sum(l.thickness for l in dead_zone_layers)


def alpha_to_grid(
    grid_x: np.ndarray,
    layers: list,
    materials: dict,
    wavelength: float,
) -> np.ndarray:
    """Map per-layer absorption coefficients onto the spatial grid."""
    alpha_arr = np.array([
        materials[lyr.material].absorption_coefficient(wavelength)
        for lyr in layers
    ])
    alpha_grid = np.zeros_like(grid_x)
    xs = 0.0
    for lyr, alpha_val in zip(layers, alpha_arr):
        xe = xs + lyr.thickness
        mask = (grid_x >= xs - 1e-16) & (grid_x <= xe + 1e-16)
        alpha_grid[mask] = alpha_val
        xs = xe
    return alpha_grid
