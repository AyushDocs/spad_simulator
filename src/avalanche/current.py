from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic.dataclasses import dataclass

from ..core.constants import q
from ..utils.pydantic_types import NDArray
from .tunneling import TunnelingModel


class CurrentDensityComponent(ABC):
    """Interface for current density components returning A/cm³."""

    @abstractmethod
    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


@dataclass(config=dict(arbitrary_types_allowed=True))
class SRHCurrentDensity(CurrentDensityComponent):
    """Shockley-Read-Hall thermal generation current density.

    Only active inside the InGaAs absorption region. Uses the full SRH formula
    with off-center trap level.
    """

    mat_name_grid: NDArray
    materials: dict | None = None  # str -> Material
    T: float = 300.0
    tau_n_absorber: float | None = None
    tau_p_absorber: float | None = None
    ni_absorber: float | None = None

    @property
    def name(self) -> str:
        return "SRH"

    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        mat_name = "InGaAs"
        mask = self.mat_name_grid == mat_name
        J = np.zeros_like(x)
        if not np.any(mask):
            return J

        if self.materials is not None:
            mat = self.materials[mat_name]
            tau_n = mat.tau_n
            tau_p = mat.tau_p

            # trap position below conduction band (Ec - Et) from XML property
            dEc = mat.E_trap
            Eg = mat.Eg(self.T)
            Nc = mat.Nc(self.T)
            Nv = mat.Nv(self.T)
            ni = mat.ni(self.T)

            from ..core.constants import kB
            kB_eV = float(kB.to("eV/K").magnitude)

            n1 = Nc * np.exp(-dEc / (kB_eV * self.T))
            p1 = Nv * np.exp(-(Eg - dEc) / (kB_eV * self.T))

            G = (ni ** 2) / (tau_p * n1 + tau_n * p1)
        else:
            # Fallback for unit tests using mock models
            tau_n = self.tau_n_absorber if self.tau_n_absorber is not None else 1e-6
            tau_p = self.tau_p_absorber if self.tau_p_absorber is not None else 1e-6
            ni = self.ni_absorber if self.ni_absorber is not None else 1e10
            tau = (tau_n + tau_p) / 2.0
            G = ni / (2.0 * tau)

        q_val = float(q.to("C").magnitude)

        J[mask] = q_val * G * (F[mask] > 5e3)
        return J


@dataclass(config=dict(arbitrary_types_allowed=True))
class BTBTCurrentDensity(CurrentDensityComponent):
    """Band-to-band tunneling current density (Kane model).

    A and B are derived from first principles using the multiplication
    layer bandgap and conductivity effective masses.
    """

    Eg_mulp: float
    mc_mulp: float
    mh_mulp: float
    T: float = 300.0
    N_T: float = 1e12

    _tunnel: TunnelingModel | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_tunnel", TunnelingModel(
            T=self.T, N_T=self.N_T,
            Eg_mulp=self.Eg_mulp, mc_mulp=self.mc_mulp, mh_mulp=self.mh_mulp))

    @property
    def name(self) -> str:
        return "BTBT"

    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        J = self._tunnel.btbt_current(F)  # A/cm³
        return J


@dataclass(config=dict(arbitrary_types_allowed=True))
class TATCurrentDensity(CurrentDensityComponent):
    """Trap-assisted tunneling current density (Hurkx model).

    Computes the field-enhanced portion of the SRH generation rate using
    the Hurkx phonon-assisted tunneling model:

        J_TAT = q * G_SRH * (Gamma_e + Gamma_h)

    where Gamma_e, Gamma_h are the Hurkx enhancement factors.

    For InGaAs this is the field enhancement ONLY (zero-field SRH is
    handled by SRHCurrentDensity).  For InP and grading layers this
    provides the full SRH+TAT generation since no other component
    accounts for zero-field SRH in those materials.

    The trap density ``N_T`` (cm⁻³) sets the overall scale of the
    trap-assisted current.  The generation rate G ∝ N_T, so changing
    N_T linearly scales the computed current.  ``N_T = 0`` disables
    all trap-assisted generation.
    """

    mat_name_grid: NDArray
    materials: dict  # str -> Material
    T: float = 300.0
    a_frac: float = 0.75
    N_T: float = 1e12

    @property
    def name(self) -> str:
        return "TAT"

    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        q_val = float(q.to("C").magnitude)
        J = np.zeros_like(x)

        # Scale factor for trap density.  The material's tau corresponds
        # to an implicit reference trap density N_T_ref = 1e12.  When the
        # user sets N_T to a different value, the current scales linearly.
        N_T_ref = 1e12
        nt_scale = self.N_T / N_T_ref if N_T_ref > 0 else 1.0

        from ..core.constants import kB
        kB_eV = float(kB.to("eV/K").magnitude)

        for mat_name in self.materials:
            mat = self.materials[mat_name]
            mask = self.mat_name_grid == mat_name
            if not np.any(mask):
                continue

            ni = mat.ni(self.T)
            tau_n = mat.tau_n
            tau_p = mat.tau_p
            Eg = mat.Eg(self.T)
            Nc = mat.Nc(self.T)
            Nv = mat.Nv(self.T)
            mc = mat.mc
            mh = mat.mh

            # Get dEc (trap position below conduction band) from XML property
            dEc = mat.E_trap

            n1 = Nc * np.exp(-dEc / (kB_eV * self.T))
            p1 = Nv * np.exp(-(Eg - dEc) / (kB_eV * self.T))

            G_SRH = (ni ** 2) / (tau_p * n1 + tau_n * p1)

            gamma_e = TunnelingModel.hurkx_gamma(F[mask], mc, dEc, self.T)
            gamma_h = TunnelingModel.hurkx_gamma(F[mask], mh, Eg - dEc, self.T)

            if mat_name == "InGaAs":
                # SRHCurrentDensity handles zero-field in InGaAs,
                # so only add the field-enhanced portion.
                Gamma_total = gamma_e + gamma_h
            else:
                # Full SRH + TAT generation for materials not covered by SRH
                Gamma_total = 1.0 + gamma_e + gamma_h

            # Apply depletion mask so we don't integrate bulk generation over neutral regions
            J[mask] = q_val * G_SRH * Gamma_total * nt_scale * (F[mask] > 5e3)

        return J


class CompositeCurrentDensity(CurrentDensityComponent):
    """Sum of multiple current components."""

    def __init__(self) -> None:
        self._components: list[CurrentDensityComponent] = []

    def add(self, component: CurrentDensityComponent) -> None:
        self._components.append(component)

    @property
    def components(self) -> list[CurrentDensityComponent]:
        return list(self._components)

    @property
    def name(self) -> str:
        return "Composite"

    def compute(self, x: np.ndarray, F: np.ndarray, **kwargs) -> np.ndarray:
        total = np.zeros_like(x)
        for comp in self._components:
            total += comp.compute(x, F, **kwargs)
        return total


class CurrentDecompositionManager(CurrentDensityComponent):
    """Manages dark-current components with independent enable/disable toggles.

    Each physical mechanism (SRH, BTBT, TAT) can be individually toggled,
    enabling current decomposition: the contribution of each mechanism is
    obtained as *total minus total-without-that-mechanism*.

    The manager plays the same role as ``CompositeCurrentDensity`` in the
    simulator facade — ``.components`` only returns enabled components.
    """

    def __init__(self) -> None:
        self._srh: SRHCurrentDensity | None = None
        self._btbt: BTBTCurrentDensity | None = None
        self._tat: TATCurrentDensity | None = None
        self._enable_srh = True
        self._enable_btbt = True
        self._enable_tat = True

    def add(self, component: CurrentDensityComponent) -> None:
        if isinstance(component, SRHCurrentDensity):
            self._srh = component
        elif isinstance(component, BTBTCurrentDensity):
            self._btbt = component
        elif isinstance(component, TATCurrentDensity):
            self._tat = component
        else:
            raise TypeError(
                f"Unknown current component type: {type(component).__name__}"
            )

    # -- enable/disable toggles ------------------------------------------------

    @property
    def enable_srh(self) -> bool:
        return self._enable_srh

    @enable_srh.setter
    def enable_srh(self, val: bool) -> None:
        self._enable_srh = val

    @property
    def enable_btbt(self) -> bool:
        return self._enable_btbt

    @enable_btbt.setter
    def enable_btbt(self, val: bool) -> None:
        self._enable_btbt = val

    @property
    def enable_tat(self) -> bool:
        return self._enable_tat

    @enable_tat.setter
    def enable_tat(self, val: bool) -> None:
        self._enable_tat = val

    def enable_all(self) -> None:
        self._enable_srh = True
        self._enable_btbt = True
        self._enable_tat = True

    def disable_all(self) -> None:
        self._enable_srh = False
        self._enable_btbt = False
        self._enable_tat = False

    # -- CurrentDensityComponent interface -------------------------------------

    @property
    def components(self) -> list[CurrentDensityComponent]:
        result: list[CurrentDensityComponent] = []
        if self._srh is not None and self._enable_srh:
            result.append(self._srh)
        if self._btbt is not None and self._enable_btbt:
            result.append(self._btbt)
        if self._tat is not None and self._enable_tat:
            result.append(self._tat)
        return result

    @property
    def name(self) -> str:
        return "CurrentDecomposition"

    def compute(self, x: np.ndarray, F: np.ndarray, **kwargs) -> np.ndarray:
        total = np.zeros_like(x)
        for comp in self.components:
            total += comp.compute(x, F, **kwargs)
        return total

    # -- decomposition helpers ------------------------------------------------

    def compute_individual(
        self, x: np.ndarray, F: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Return {name: J(x)} for every registered component (regardless of
        enable flags)."""
        result: dict[str, np.ndarray] = {}
        if self._srh is not None:
            result["SRH"] = self._srh.compute(x, F)
        if self._btbt is not None:
            result["BTBT"] = self._btbt.compute(x, F)
        if self._tat is not None:
            result["TAT"] = self._tat.compute(x, F)
        return result

    def decompose(self, x: np.ndarray, F: np.ndarray) -> dict[str, np.ndarray]:
        """Current decomposition: compute total, then remove each mechanism
        one at a time.  Returns ``{mechanism: contribution}`` where each
        contribution is *total - total_without_that_mechanism* (A/cm³)."""
        saved = (self._enable_srh, self._enable_btbt, self._enable_tat)

        self.enable_all()
        total = self.compute(x, F)

        contrib: dict[str, np.ndarray] = {}

        self._enable_srh = False
        contrib["SRH"] = total - self.compute(x, F)
        self._enable_srh = True

        self._enable_btbt = False
        contrib["BTBT"] = total - self.compute(x, F)
        self._enable_btbt = True

        self._enable_tat = False
        contrib["TAT"] = total - self.compute(x, F)
        self._enable_tat = True

        self._enable_srh, self._enable_btbt, self._enable_tat = saved
        return contrib
