"""SPADSimulator — thin facade that wires subsystems together."""

from __future__ import annotations

import numpy as np

from ..core.device import Device
from ..core.layer import Layer
from ..poisson.service import PoissonService
from ..avalanche.ionization import IonizationCoefficients
from ..avalanche.trigger import TriggerSolver
from ..transport.drift_diffusion import DriftDiffusionSolver, DriftDiffusion
from ..transport.monte_carlo import MonteCarloTransport
from ..self_consistent.particle_mesh import ParticleMesh
from ..self_consistent.circuit import CircuitSolver
from ..self_consistent.loop import SelfConsistentLoop
from ..utils._logging import get_logger

from .builder import build_subsystems
from .field_cache import FieldCache
from .breakdown import find_breakdown as _find_breakdown

log = get_logger("simulator")


class SPADSimulator:
    """Top-level orchestrator for SPAD/APD simulation.

    Constructs all subsystem objects from a ``Device`` and provides
    convenience methods that combine Poisson, avalanche, and current
    computations.  No cached bias state — caller provides guesses.
    """

    def __init__(
        self,
        device: Device,
        detector_area: float = 1e-6,
        transport_material: str = "InP",
        cache_maxlen: int = 200,
    ) -> None:
        self.device = device
        self.grid = self.device.grid
        self.materials = self.device.materials
        self.T = self.device.T
        self.detector_area = detector_area
        self.transport_material = transport_material

        subs = build_subsystems(device, self.grid, self.materials, self.T)
        self.poisson_service: PoissonService = subs["poisson_service"]
        self.ionization: IonizationCoefficients = subs["ionization"]
        self.current = subs["current"]
        self.pde_model = subs["pde_model"]
        self.trigger = TriggerSolver(self.grid)

        self.transport: DriftDiffusion | None = None
        self.mc_sim: MonteCarloTransport | None = None
        self.mesh: ParticleMesh | None = None
        self.circuit: CircuitSolver | None = None
        self.loop: SelfConsistentLoop | None = None

        self._Vbr: float | None = None
        self._dead_space_Eg = 1.35  # InP bandgap (eV) for dead-space correction
        self._field_cache = FieldCache(maxlen=cache_maxlen)

    # -- Reconfiguration -------------------------------------------------------

    def set_layers(self, layers: list[Layer]) -> None:
        self.device = Device(layers=layers, materials=self.materials, no_of_nodes=self.grid.no_of_nodes)
        self.grid = self.device.grid
        self._rebuild()

    def set_doping(self, layer_specs: list) -> None:
        from ..core.doping import DopingProfile, LayerSpec
        specs = [LayerSpec(**s) if isinstance(s, dict) else s for s in layer_specs]
        new_doping = DopingProfile(specs)
        self.device.doping = new_doping
        self._rebuild()

    def set_nt(self, nt: float) -> None:
        for comp in self.current.components:
            if hasattr(comp, "N_T"):
                comp.N_T = nt

    def _rebuild(self) -> None:
        subs = build_subsystems(self.device, self.grid, self.materials, self.T)
        self.poisson_service = subs["poisson_service"]
        self.ionization = subs["ionization"]
        self.current = subs["current"]
        self._Vbr = None
        self._field_cache.clear()

    # -- Field solve ------------------------------------------------------------

    def get_fields(
        self, Vbias: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        cached = self._field_cache.get(Vbias)
        if cached is not None:
            if len(cached) == 6 and isinstance(cached[2], np.ndarray):
                return cached

        guess = self._field_cache.interpolate_guess(Vbias)
        phi, E, _ = self.poisson_service.solve(Vbias, guess=guess)
        xl, xr, _ = self.depletion_width(Vbias, E=E)

        alpha = self.ionization.alpha_n(np.abs(E))
        beta = self.ionization.alpha_p(np.abs(E))

        # We compute the raw dead space lengths to pass to the non-local trigger solver
        l_e = self.ionization.dead_space_length(E, "electron", self._dead_space_Eg)
        l_h = self.ionization.dead_space_length(E, "hole", self._dead_space_Eg)

        Pe, Ph = self.trigger.solve(E, alpha, beta, self.grid.x, l_e=l_e, l_h=l_h)

        result = (phi, E, Pe, Ph, xl, xr)
        self._field_cache.put(Vbias, result)
        return result

    def solve_poisson(
        self,
        Vbias: float,
        phi_n: float | None = None,
        phi_p: float = 0.0,
        guess: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        cached = self._field_cache.get(Vbias)
        if cached is not None:
            return cached[0], self.grid.gradient(cached[0]), {"converged": True, "cached": True}

        if guess is None:
            guess = self._field_cache.interpolate_guess(Vbias)
        phi, E, info = self.poisson_service.solve(Vbias, phi_n, phi_p, guess)

        # Populate field cache for future warm-start
        cached_phi = self._field_cache.get(Vbias)
        if cached_phi is None:
            xl, xr, _ = self.depletion_width(Vbias, E=E)
            alpha = self.ionization.alpha_n(np.abs(E))
            beta = self.ionization.alpha_p(np.abs(E))
            l_e = self.ionization.dead_space_length(E, "electron", self._dead_space_Eg)
            l_h = self.ionization.dead_space_length(E, "hole", self._dead_space_Eg)
            Pe, Ph = self.trigger.solve(E, alpha, beta, self.grid.x, l_e=l_e, l_h=l_h)
            self._field_cache.put(Vbias, (phi, E, Pe, Ph, xl, xr))

        return phi, E, info

    # -- Ionization helpers ------------------------------------------------------

    def solve_trigger(self, Vbias: float, field_threshold: float = 1e4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phi, E, _ = self.solve_poisson(Vbias)
        alpha = self.ionization.alpha_n(np.abs(E))
        beta = self.ionization.alpha_p(np.abs(E))
        l_e = self.ionization.dead_space_length(E, "electron", self._dead_space_Eg)
        l_h = self.ionization.dead_space_length(E, "hole", self._dead_space_Eg)
        Pe, Ph = self.trigger.solve(E, alpha, beta, self.grid.x, l_e=l_e, l_h=l_h, field_threshold=field_threshold)
        return Pe, Ph, E

    def depletion_width(self, Vbias: float, E: np.ndarray | None = None) -> tuple[float, float, float]:
        if E is not None:
            # E is in V/cm, from_field expects eps_field in same units. Let's pass 1e4 V/cm (equivalent to 1e6 V/m)
            return self.poisson_service.depletion.from_field(E, eps_field=1e4)
        return self.poisson_service.depletion_width(Vbias)

    # -- Breakdown voltage -------------------------------------------------------

    def find_breakdown(
        self,
        V_start: float = 0.0,
        V_max: float = 100.0,
        V_step: float = 0.1,
        force: bool = False,
    ) -> tuple[float | None, list[dict]]:
        Vbr, results = _find_breakdown(
            self.poisson_service.poisson, self.grid,
            self.ionization, self._Vbr,
            V_start, V_max, V_step, force,
        )
        if Vbr is not None and self._Vbr is None:
            self._Vbr = Vbr
        return Vbr, results

    # -- Dark current ------------------------------------------------------------

    def _compute_multiplication(self, E: np.ndarray) -> float:
        """Avalanche multiplication factor via the McIntyre integral.

        Electron injection from the absorber (right side, x=W):
            M = 1 / [1 − ∫α(x)·exp(∫_x^W (β−α) dx′) dx]
        """
        alpha = self.ionization.effective_alpha_n(np.abs(E), Eg=self._dead_space_Eg)
        beta = self.ionization.effective_alpha_p(np.abs(E), Eg=self._dead_space_Eg)
        x = self.grid.x
        active = np.abs(E) > 1e5
        if not np.any(active):
            return 1.0
        dx = np.diff(x)
        diff = beta - alpha
        cum = np.zeros_like(x)
        for i in range(len(x) - 2, -1, -1):
            cum[i] = cum[i + 1] + diff[i] * dx[i]
        integrand = alpha * np.exp(cum)
        denom = 1.0 - float(np.trapezoid(integrand, x))
        if denom <= 0.01:
            return 1e6
        return min(1.0 / denom, 1e6)

    def compute_dark_current(self, Vbias: float, E: np.ndarray | None = None) -> dict:
        if E is None:
            _, E, Pe, Ph, _, _ = self.get_fields(Vbias)
        else:
            alpha = self.ionization.alpha_n(np.abs(E))
            beta = self.ionization.alpha_p(np.abs(E))
            l_e = self.ionization.dead_space_length(E, "electron", self._dead_space_Eg)
            l_h = self.ionization.dead_space_length(E, "hole", self._dead_space_Eg)
            Pe, Ph = self.trigger.solve(E, alpha, beta, self.grid.x, l_e=l_e, l_h=l_h)

        x = self.grid.x
        J_total = np.zeros_like(x)
        for comp in self.current.components:
            J_total += comp.compute(x, np.abs(E))
        I_primary = float(np.trapezoid(J_total, x) * self.detector_area)

        M = self._compute_multiplication(E)
        I_dark = I_primary * M

        return {"J_total": J_total, "I_dark": I_dark, "DCR": abs(I_dark),
                "Pe": Pe, "Ph": Ph, "E": E, "M": M}

    # -- Self-consistent PIC -----------------------------------------------------

    def build_self_consistent(
        self, Vbias: float, Rq: float = 1e5, Cspad: float = 1e-15,
        Vbr: float | None = None,
    ) -> SelfConsistentLoop:
        self.mesh = ParticleMesh(self.grid)
        self.transport = DriftDiffusion(self.materials[self.transport_material])
        self.circuit = CircuitSolver(Vbias=Vbias, Rq=Rq, Cspad=Cspad, Vbr=0.0 if Vbr is None else Vbr)

        self.loop = SelfConsistentLoop(
            self.grid, self.device.doping, self.poisson_service.poisson,
            self.mesh, self.transport, self.ionization, self.circuit,
        )
        return self.loop

    def run_pic(self, N_steps: int, inject_x: float | None = None) -> list[dict]:
        if self.loop is None:
            raise RuntimeError("Call build_self_consistent() first.")
        if inject_x is not None:
            self.loop.inject_carrier(inject_x)
        return self.loop.run(N_steps)
