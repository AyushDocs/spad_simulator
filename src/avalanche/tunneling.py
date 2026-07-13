from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator
from scipy.special import erfc

from ..core.constants import q, kB, hbar, m0, pi
from ..utils.pydantic_types import NDArray


class TunnelingModel(BaseModel):
    """
    Band-to-band (BTBT) and trap-assisted (TAT) tunneling currents.

    BTBT (Kane model, derived from first principles):
        J_BTBT = A * F^2 * exp(-B / F)
        where A and B are derived from:
            mR = mc * mh / (mc + mh)
            A  = (q^2 * (2*mR)^(3/2)) / (4*pi*hbar^2 * Eg^(3/2))
            B  = pi * sqrt(mR/2) / (2*q*hbar)

    TAT (Hurkx phonon-assisted tunneling model):
        Computes a field-dependent enhancement of the SRH generation rate.
        G_TAT(F) = G_SRH(0) * (1 + Gamma_n + Gamma_p)
        where Gamma is the Hurkx enhancement factor for each carrier:
            Gamma = (dE/kT) * sqrt(pi/2) * (F/F0) * exp(F^2/(2F0^2)) * erfc(-F/(sqrt2*F0))
        with F0 = 4*sqrt(2m*)*dE^{3/2} / (3*q*hbar)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    T: float = 300.0
    N_T: float = 1e12
    Eg_mulp: float = 1.35
    mc_mulp: float = 0.041
    mh_mulp: float = 0.4
    Nc_grid: NDArray | None = None
    Nv_grid: NDArray | None = None

    _A: float = PrivateAttr()
    _B_btbt: float = PrivateAttr()
    _T_private: float = PrivateAttr()
    _N_T_private: float = PrivateAttr()
    _Nc_grid_private: np.ndarray | None = PrivateAttr()
    _Nv_grid_private: np.ndarray | None = PrivateAttr()

    @field_validator("T")
    @classmethod
    def _T_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Temperature must be positive, got {v}")
        return v

    @field_validator("N_T")
    @classmethod
    def _N_T_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Trap concentration must be non-negative, got {v}")
        return v

    @field_validator("Eg_mulp")
    @classmethod
    def _Eg_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Bandgap must be positive, got {v}")
        return v

    @field_validator("mc_mulp", "mh_mulp")
    @classmethod
    def _mass_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Effective mass must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _compute_coefficients(self):
        self._T_private = self.T
        self._N_T_private = self.N_T
        self._Nc_grid_private = self.Nc_grid
        self._Nv_grid_private = self.Nv_grid

        q_val = float(q.to("C").magnitude)
        hbar_val = float(hbar.to("J*s").magnitude)
        m0_val = float(m0.to("kg").magnitude)

        Eg_J = self.Eg_mulp * q_val
        mc_kg = self.mc_mulp * m0_val
        mh_kg = self.mh_mulp * m0_val
        mR = mc_kg * mh_kg / (mc_kg + mh_kg)

        self._A = q_val**3 * np.sqrt((2.0 * mR) / Eg_J) / (4.0 * pi**3 * hbar_val**2)
        self._B_btbt = pi * np.sqrt(mR / 2.0) * (Eg_J ** 1.5) / (2.0 * q_val * hbar_val)
        return self

    @staticmethod
    def hurkx_gamma(F: np.ndarray, m_eff: float,
                    E_barrier_eV: float, T: float = 300.0) -> np.ndarray:
        r"""Hurkx phonon-assisted tunneling enhancement factor.

        Parameters
        ----------
        F : np.ndarray
            Electric field in V/cm.
        m_eff : float
            Effective mass (in units of m0).
        E_barrier_eV : float
            Barrier height in eV (E_C - E_T for electrons,
            E_T - E_V for holes).
        T : float
            Temperature in K.

        Returns
        -------
        np.ndarray
            Enhancement factor Gamma (dimensionless), >= 0.
            At zero field returns 0.
        """
        if E_barrier_eV <= 0.0 or T <= 0.0:
            return np.zeros_like(F)

        m0_val = float(m0.to("kg").magnitude)
        q_val = float(q.to("C").magnitude)
        hbar_val = float(hbar.to("J*s").magnitude)
        kB_J_K = float(kB.to("J/K").magnitude)

        m_kg = m_eff * m0_val
        E_J = E_barrier_eV * q_val
        kT_J = kB_J_K * T
        dE_over_kT = E_J / kT_J

        F_SI = F * 100.0

        F0_SI = (4.0 * np.sqrt(2.0 * m_kg) * E_J ** 1.5
                 / (3.0 * q_val * hbar_val))

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            x = F_SI / (np.sqrt(2.0) * F0_SI)
            # Use scipy.special.erfc for numpy array support
            exp_x2 = np.exp(x ** 2)
            erfc_neg = erfc(-x)
            prod = exp_x2 * erfc_neg
            overflow = ~np.isfinite(prod)
            if np.any(overflow):
                prod = np.where(overflow, 2.0 * exp_x2, prod)
            Gamma = dE_over_kT * np.sqrt(np.pi / 2.0) * x * prod
            Gamma = np.clip(Gamma, 0.0, 1e20)
            Gamma[F < 100.0] = 0.0

        return Gamma

    def btbt_current(self, F: np.ndarray) -> np.ndarray:
        """BTBT current density. F in V/cm, returns A/cm³."""
        # Convert F from V/cm to V/m
        F_SI = F * 100.0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            J_SI = np.where(F_SI > 1e7,
                            self._A * F_SI ** 2
                            * np.exp(-self._B_btbt / F_SI),
                            0.0)
        return J_SI * 1e-6  # A/m³ -> A/cm³

    def tat_current(self, F: np.ndarray,
                    Eg_mulp: float,
                    mc_mulp: float,
                    mh_mulp: float,
                    ni_val: float,
                    tau_val: float,
                    a_frac: float = 0.75,
                    N_T: float = 5e15) -> np.ndarray:
        """TAT current density (Hurkx model). F in V/cm, returns A/cm³.

        The TAT current is the field-enhanced portion of the SRH generation
        rate using the Hurkx phonon-assisted tunneling model:

            G_TAT(F) = (n_i / (2*tau)) * (Gamma_electron + Gamma_hole)

        where Gamma_e and Gamma_h are the field enhancement factors for
        electron and hole emission from the trap.

        For InGaAs this is the field enhancement only (SRH handles zero-field).
        For InP/InGaAsP this includes the full SRH+TAT generation.
        """
        q_val = float(q.to("C").magnitude)

        E_B_e = (1.0 - a_frac) * Eg_mulp
        E_B_h = a_frac * Eg_mulp

        gamma_e = self.hurkx_gamma(F, m_eff=mc_mulp, E_barrier_eV=E_B_e, T=self._T_private)
        gamma_h = self.hurkx_gamma(F, m_eff=mh_mulp, E_barrier_eV=E_B_h, T=self._T_private)

        Gamma_total = gamma_e + gamma_h

        G_SRH = ni_val / (2.0 * tau_val)
        J = q_val * G_SRH * Gamma_total
        return J
