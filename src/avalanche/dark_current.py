"""Dark current density components."""

from __future__ import annotations


import numpy as np
from pydantic.dataclasses import dataclass

from ..core.constants import q, kB
from ..utils.pydantic_types import NDArray
from .current import CurrentDensityComponent


@dataclass(config=dict(arbitrary_types_allowed=True))
class DiffusionCurrentDensity(CurrentDensityComponent):
    """Diffusion current density from minority carriers.

    J_diff ~ q * (Dn/tau_n + Dp/tau_p) * ni² / N_A
    Only active in the InGaAs absorption region.
    """

    T: float
    tau_n_absorber: float
    tau_p_absorber: float
    mat_name_grid: NDArray
    ni_absorber: float

    @property
    def name(self) -> str:
        return "Diffusion"

    def compute(self, x: np.ndarray, F: np.ndarray, **kwargs) -> np.ndarray:
        vth = float((q * kB * T / q**2).magnitude)  # rough thermal velocity
        Dn = 26.0  # cm²/s typical for InGaAs
        Dp = 10.0  # cm²/s typical for InGaAs
        J = float(q.to("C").magnitude) * self.ni_absorber**2 * (
            Dn / self.tau_n_absorber + Dp / self.tau_p_absorber) / 1e16
        in_absorber = self.mat_name_grid == "InGaAs"
        return J * in_absorber


@dataclass(config=dict(arbitrary_types_allowed=True))
class GenerationCurrentDensity(CurrentDensityComponent):
    """Thermal generation current density (SRH).

    J_gen ~ q * ni * W / (2 * tau)
    """

    T: float
    tau_n_absorber: float
    tau_p_absorber: float
    mat_name_grid: NDArray
    ni_absorber: float

    @property
    def name(self) -> str:
        return "Generation"

    def compute(self, x: np.ndarray, F: np.ndarray, **kwargs) -> np.ndarray:
        tau = (self.tau_n_absorber + self.tau_p_absorber) / 2.0
        G = self.ni_absorber / (2.0 * tau)
        J = float(q.to("C").magnitude) * G
        in_absorber = self.mat_name_grid == "InGaAs"
        return J * in_absorber


class DarkCurrentModel:
    """DarkCurrentModel compatibility class for tests."""

    def __init__(self, T, grid_x, N_T, mat_name_grid, ni_absorber, tau_n_absorber, tau_p_absorber, Eg_mulp, mc_mulp, mh_mulp):
        self.T = T
        self.grid_x = grid_x
        self.N_T = N_T
        self.mat_name_grid = mat_name_grid
        self.ni_absorber = ni_absorber
        self.tau_n_absorber = tau_n_absorber
        self.tau_p_absorber = tau_p_absorber
        self.Eg_mulp = Eg_mulp
        self.mc_mulp = mc_mulp
        self.mh_mulp = mh_mulp

        from .current import CompositeCurrentDensity, BTBTCurrentDensity, TATCurrentDensity, SRHCurrentDensity
        self.current = CompositeCurrentDensity()
        self.current.add(SRHCurrentDensity(
            tau_n_absorber=tau_n_absorber,
            tau_p_absorber=tau_p_absorber,
            mat_name_grid=mat_name_grid,
            ni_absorber=ni_absorber,
        ))
        self.current.add(BTBTCurrentDensity(
            Eg_mulp=Eg_mulp, mc_mulp=mc_mulp, mh_mulp=mh_mulp, T=T, N_T=N_T
        ))
        self.current.add(TATCurrentDensity(
            Eg_mulp=Eg_mulp, mc_mulp=mc_mulp, mh_mulp=mh_mulp, T=T, N_T=N_T
        ))

    def total_dark_current_density(self, x, F, ni_arr, Eg_arr, mc_arr, mh_arr):
        return self.current.compute(x, F)

    def generation_rate(self, x, F, ni_arr, Eg_arr, mc_arr, mh_arr):
        J_total = self.total_dark_current_density(x, F, ni_arr, Eg_arr, mc_arr, mh_arr)
        q_val = float(q.to("C").magnitude)
        return J_total / q_val

    def compute_dcr(self, x, F, Pe, ni_arr, Eg_arr, mc_arr, mh_arr, detector_area=1e-6):
        G = self.generation_rate(x, F, ni_arr, Eg_arr, mc_arr, mh_arr)
        return float(np.trapezoid(G * Pe, x) * detector_area)
