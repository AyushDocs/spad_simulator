"""Physical constants with pint units.

All constants use the project's mixed SI-CGS unit system.
"""
from __future__ import annotations

from .units import Q_

# Fundamental constants
q = Q_(1.602176634e-19, "C")                    # elementary charge
kB = Q_(1.380649e-23, "J/K")                    # Boltzmann constant
eps0 = Q_(8.854187817e-14, "F/cm")              # vacuum permittivity
hbar = Q_(1.054571817e-34, "J*s")               # reduced Planck constant
m0 = Q_(9.10938356e-31, "kg")                   # electron rest mass
pi = 3.141592653589793                           # dimensionless
c = Q_(2.998e10, "cm/s")                        # speed of light
h = Q_(6.62607015e-34, "J*s")                   # Planck constant


def VT(T: float) -> float:
    """Thermal voltage at temperature T (V)."""
    return float(kB.magnitude) * T / float(q.magnitude)


def thermal_energy(T: float) -> float:
    """Thermal energy at temperature T (J)."""
    return float(kB.magnitude) * T
