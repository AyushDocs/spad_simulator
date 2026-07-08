from __future__ import annotations

# Physical constants (SI units unless noted)
q = 1.602176634e-19         # C (elementary charge)
kB = 1.380649e-23            # J/K (Boltzmann constant)
eps0 = 8.854187817e-14       # F/cm (vacuum permittivity)
hbar = 1.054571817e-34       # J·s (reduced Planck constant)
m0 = 9.10938356e-31          # kg (electron rest mass)
pi = 3.141592653589793
c = 2.998e10                 # cm/s (speed of light)
h = 6.62607015e-34           # J·s (Planck constant)


def VT(T: float) -> float:
    """Thermal voltage at temperature T (V)."""
    return kB * T / q


def thermal_energy(T: float) -> float:
    """Thermal energy at temperature T (J)."""
    return kB * T
