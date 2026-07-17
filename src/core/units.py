"""Pint unit registry for the SPAD simulator.

All internal calculations use a mixed SI-CGS system:
  cm, V, A, eV, s, K, F, ohm

This module provides the shared registry, physical constants,
and helper functions for creating and converting quantities.
"""
from __future__ import annotations

import pint

ureg: pint.UnitRegistry = pint.UnitRegistry(cache_folder=":auto:")

# Custom unit: electron rest mass (ratio to kg)
ureg.define("electron_rest_mass = 9.1093837015e-31 kilogram = m0")

Q_ = ureg.Quantity

# -- Unit aliases for XML strings that pint may not parse directly ----------
_UNIT_ALIASES: dict[str, str] = {
    "cm^2/V·s": "cm**2/(V*s)",
    "cm^2/V*s": "cm**2/(V*s)",
    "cm2/(V*s)": "cm**2/(V*s)",
    "cm2/V*s": "cm**2/(V*s)",
    "cm2/V*S": "cm**2/(V*s)",
    "cm^-3": "cm**-3",
    "cm-3": "cm**-3",
    "m^-3": "m**-3",
    "m-3": "m**-3",
    "eV/K": "eV/K",
    "m0": "electron_rest_mass",
    "1": "dimensionless",
}


def parse_unit(unit_str: str) -> pint.Unit:
    """Parse a unit string, applying aliases for XML-style notation."""
    normalized = _UNIT_ALIASES.get(unit_str.strip(), unit_str.strip())
    return ureg.Unit(normalized)


def Q(value, unit_str: str) -> pint.Quantity:
    """Create a pint Quantity from a value and unit string.

    Handles XML-style unit strings via alias normalization.
    """
    unit = parse_unit(unit_str)
    return Q_(value, unit)


def ensure_quantity(value, unit_str: str) -> pint.Quantity:
    """If value is already a Quantity, validate; otherwise wrap it."""
    if isinstance(value, pint.Quantity):
        return value.to(unit_str)
    return Q_(value, unit_str)
