from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator

from ..core.constants import q, hbar, m0, pi
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

    TAT (Hurkx model, SI units internally):
        F in V/cm at entry, converted to V/m internally.
        Eg in eV at entry, converted to J internally.
        Returns A/cm².
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
                    N_T: float = 1e12,
                    a_frac: float = 0.75) -> np.ndarray:
        """TAT current density. F in V/cm, returns A/cm³."""
        q_val = float(q.to("C").magnitude)
        m0_val = float(m0.to("kg").magnitude)

        Eg_J = Eg_mulp * q_val
        mc_kg = mc_mulp * m0_val
        mh_kg = mh_mulp * m0_val

        hbar_val = float(hbar.to("J*s").magnitude)
        B1 = np.pi * np.sqrt(mh_kg / 2.0) / (2.0 * q_val * hbar_val)
        B2 = np.pi * np.sqrt(mc_kg / 2.0) / (2.0 * q_val * hbar_val)
        E_B1 = a_frac * Eg_J
        E_B2 = (1.0 - a_frac) * Eg_J
        Nc = 1e25  # m⁻³
        Nv = 1e25  # m⁻³
        N_T_SI = N_T * 1e6  # cm⁻³ -> m⁻³

        # Convert F from V/cm to V/m
        F_SI = F * 100.0

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            exp1 = np.exp(-(B1 * (E_B1**1.5)) / F_SI)
            exp2 = np.exp(-(B2 * (E_B2**1.5)) / F_SI)
            num = self._A * F_SI ** 2 * N_T_SI * exp1 * exp2
            den = Nv * exp1 + Nc * exp2
            J_TAT = np.where(F_SI > 1e7,
                             num / (den + 1e-300),
                             0.0)

        J_SI = q_val * J_TAT  # A/m³
        return J_SI * 1e-6  # A/m³ -> A/cm³
