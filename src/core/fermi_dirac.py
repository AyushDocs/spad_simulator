"""Fermi-Dirac statistics and bandgap narrowing models."""
from __future__ import annotations

import numpy as np

from ..core.constants import q, VT


class FermiDiracStatistics:
    """
    Fermi-Dirac carrier statistics with degeneracy corrections.

    For non-degenerate semiconductors (Boltzmann approximation):
        n = Nc * exp((E_F - E_c) / kT)
        p = Nv * exp((E_v - E_F) / kT)

    For degenerate semiconductors (Fermi-Dirac):
        n = Nc * F_{1/2}(eta_n)   where eta_n = (E_F - E_c)/kT
        p = Nv * F_{1/2}(eta_p)   where eta_p = (E_v - E_F)/kT

    F_{1/2}(eta) = (2/sqrt(pi)) * integral_0^inf x^{1/2}/(1+exp(x-eta)) dx

    The Sommerfeld expansion gives:
        F_{1/2}(eta) ≈ (4/(3*sqrt(pi))) * eta^{3/2}  for eta >> 0
    """

    @staticmethod
    def fermi_dirac_half(eta: float | np.ndarray) -> float | np.ndarray:
        """
        Fermi-Dirac integral of order 1/2.

        F_{1/2}(eta) = (2/sqrt(pi)) * integral_0^inf t^{1/2}/(1+exp(t-eta)) dt

        Uses rational approximation for numerical evaluation.
        """
        eta = np.asarray(eta, dtype=float)
        result = np.zeros_like(eta)
        neg = eta < 0
        pos = ~neg

        if np.any(neg):
            exp_neg = np.exp(-eta[neg])
            result[neg] = exp_neg / (1.0 + 0.25 * exp_neg)

        if np.any(pos):
            eta_p = eta[pos]
            result[pos] = (4.0 / (3.0 * np.sqrt(np.pi))) * eta_p ** 1.5

        return float(result) if result.ndim == 0 else result

    @staticmethod
    def carrier_density(ni: float, Eg: float, E_F: float,
                        Nc: float, Nv: float, T: float) -> tuple[float, float]:
        """
        Electron and hole densities from Fermi level position.

        Returns (n, p) in cm⁻³.
        """
        vth = VT(T)
        eta_n = (E_F) / (q * vth)
        eta_p = -(Eg / q + E_F / (q * vth))

        n = Nc * FermiDiracStatistics.fermi_dirac_half(eta_n)
        p = Nv * FermiDiracStatistics.fermi_dirac_half(eta_p)
        return float(n), float(p)


class BandgapNarrowing:
    """
    Bandgap narrowing (BGN) due to heavy doping.

    Model: Delta_Eg = A * (N / N_ref)^{1/3}

    where A is a material constant and N_ref is a reference concentration.

    For InP: A ≈ 2.4e-8 eV·cm, N_ref ≈ 1e17 cm⁻³
    For InGaAs: A ≈ 1.2e-8 eV·cm, N_ref ≈ 1e17 cm⁻³

    References
    ----------
    Slotboom & de Graaff (1976). "Measurements of bandgap narrowing in
    Si bipolar transistors." Solid-State Electronics, 19(10), 857-862.
    """

    def __init__(self, A: float = 2.4e-8, N_ref: float = 1e17) -> None:
        self.A = A
        self.N_ref = N_ref

    def delta_eg(self, N_doping: float) -> float:
        """
        Bandgap narrowing energy (eV).

        Parameters
        ----------
        N_doping : float
            Net doping concentration (cm⁻³).
        """
        N_abs = abs(N_doping)
        if N_abs < self.N_ref:
            return 0.0
        return float(self.A * (N_abs / self.N_ref) ** (1.0 / 3.0))

    def effective_eg(self, Eg0: float, N_doping: float) -> float:
        """Effective bandgap with narrowing (eV)."""
        return Eg0 - self.delta_eg(N_doping)

    @classmethod
    def for_inp(cls) -> BandgapNarrowing:
        return cls(A=2.4e-8, N_ref=1e17)

    @classmethod
    def for_ingaas(cls) -> BandgapNarrowing:
        return cls(A=1.2e-8, N_ref=1e17)


class CaugheyThomasMobility:
    """
    Doping- and temperature-dependent mobility (Caughey-Thomas model).

        mu(N, T) = mu_min + (mu_max - mu_min) / (1 + (N/N_ref)^gamma)

    with temperature scaling:
        mu(T) = mu(300K) * (T/300K)^{-alpha_T}

    For InP electrons: mu_max=5400, mu_min=200, N_ref=1e17, gamma=0.45, alpha_T=2.1
    For InP holes: mu_max=2000, mu_min=100, N_ref=1e17, gamma=0.45, alpha_T=2.2

    References
    ----------
    Caughey, D.M. & Thomas, R.E. (1967). "Carrier mobilities in silicon
    empirically related to doping and field." Proc. IEEE, 55(12), 2192-2193.
    """

    def __init__(self, mu_max: float, mu_min: float, N_ref: float,
                 gamma: float = 0.45, alpha_T: float = 2.1) -> None:
        self.mu_max = mu_max
        self.mu_min = mu_min
        self.N_ref = N_ref
        self.gamma = gamma
        self.alpha_T = alpha_T

    def mobility(self, N: float, T: float = 300.0) -> float:
        """
        Mobility at doping N and temperature T.

        Returns mobility in cm²/(V·s).
        """
        N_abs = abs(N)
        mu_300 = self.mu_min + (self.mu_max - self.mu_min) / (
            1.0 + (N_abs / self.N_ref) ** self.gamma
        )
        return float(mu_300 * (T / 300.0) ** (-self.alpha_T))

    @classmethod
    def for_inp_electron(cls) -> CaugheyThomasMobility:
        return cls(mu_max=5400, mu_min=200, N_ref=1e17, gamma=0.45, alpha_T=2.1)

    @classmethod
    def for_inp_hole(cls) -> CaugheyThomasMobility:
        return cls(mu_max=2000, mu_min=100, N_ref=1e17, gamma=0.45, alpha_T=2.2)

    @classmethod
    def for_ingaas_electron(cls) -> CaugheyThomasMobility:
        return cls(mu_max=12000, mu_min=500, N_ref=1e17, gamma=0.5, alpha_T=2.0)

    @classmethod
    def for_ingaas_hole(cls) -> CaugheyThomasMobility:
        return cls(mu_max=450, mu_min=50, N_ref=1e17, gamma=0.5, alpha_T=2.1)
