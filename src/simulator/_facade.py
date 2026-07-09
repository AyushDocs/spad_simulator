"""SPADSimulator — thin facade that wires subsystems together."""

from __future__ import annotations

import numpy as np

from ..core.device import Device
from ..core.layer import Layer
from ..poisson.service import PoissonService
from ..avalanche.ionization import IonizationCoefficients
from ..avalanche.trigger import TriggerSolver
from ..transport.drift_diffusion import DriftDiffusion
from ..transport.monte_carlo import MonteCarloSimulator
from ..self_consistent.particle_mesh import ParticleMesh
from ..self_consistent.circuit import CircuitSolver
from ..self_consistent.loop import SelfConsistentLoop
from ..utils._logging import get_logger

from .builder import build_subsystems
from .field_cache import FieldCache
from .photocurrent import (
    compute_photocurrent as _compute_photocurrent,
    compute_pdp_spectrum as _compute_pdp_spectrum,
)
from .breakdown import find_breakdown as _find_breakdown
from .monte_carlo import run_mc_ensemble as _run_mc_ensemble, compute_jitter as _compute_jitter

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
        self.trigger: TriggerSolver = subs["trigger"]
        self.dark_current = subs["dark_current"]
        self.pdp_model = subs["pdp_model"]

        self.transport: DriftDiffusion | None = None
        self.mc_sim: MonteCarloSimulator | None = None
        self.mesh: ParticleMesh | None = None
        self.circuit: CircuitSolver | None = None
        self.loop: SelfConsistentLoop | None = None

        self._Vbr: float | None = None
        self._field_cache = FieldCache(maxlen=cache_maxlen)

    # -- Reconfiguration -------------------------------------------------------

    def set_layers(self, layers: list[Layer]) -> None:
        self.device = Device(layers, self.materials, self.grid.no_of_nodes)
        self.grid = self.device.grid
        self._rebuild()

    def set_doping(self, layer_specs: List[dict]) -> None:
        from ..core.doping import DopingProfile
        new_doping = DopingProfile(layer_specs)
        self.device.doping = new_doping  # type: ignore[misc]
        self._rebuild()

    def _rebuild(self) -> None:
        subs = build_subsystems(self.device, self.grid, self.materials, self.T)
        self.poisson_service = subs["poisson_service"]
        self.ionization = subs["ionization"]
        self.trigger = subs["trigger"]
        self.dark_current = subs["dark_current"]
        self._Vbr = None
        self._field_cache.clear()

    # -- Field solve ------------------------------------------------------------

    def get_fields(
        self, Vbias: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        cached = self._field_cache.get(Vbias)
        if cached is not None:
            return cached

        guess = self._field_cache.interpolate_guess(Vbias)
        phi, E, _ = self.poisson_service.solve(Vbias, guess=guess)
        Pe, Ph = self._ionization_and_trigger(E)
        xl, xr, _ = self.poisson_service.depletion.from_field(E)

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
        return phi, E, info

    # -- Breakdown ---------------------------------------------------------------

    def find_breakdown(
        self,
        V_start: float = 0.0,
        V_max: float = 100.0,
        V_step: float = 0.1,
        force: bool = False,
        criterion: str = "current",
        I_threshold: float = 1e-6,
    ) -> tuple[float | None, list[dict]]:
        Vbr, results = _find_breakdown(
            self.poisson_service.poisson, self.grid, self.ionization,
            self.trigger, self.dark_current, self.device, self.detector_area,
            self._Vbr, V_start, V_max, V_step, force, criterion, I_threshold,
        )
        if Vbr is not None and self._Vbr is None:
            self._Vbr = Vbr
        return Vbr, results

    # -- Ionization helpers ------------------------------------------------------

    def solve_trigger(self, Vbias: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phi, E, _ = self.solve_poisson(Vbias)
        Pe, Ph = self._ionization_and_trigger(E)
        return Pe, Ph, E

    def depletion_width(self, Vbias: float) -> tuple[float, float, float]:
        return self.poisson_service.depletion_width(Vbias)

    def _ionization_and_trigger(self, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        alpha = self.ionization.alpha(E)
        beta = self.ionization.beta(E)
        return self.trigger.solve(E, alpha, beta, self.grid.x)

    def trigger_for_pdp(self, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        alpha = self.ionization.alpha(E)
        beta = self.ionization.beta(E)
        return self.trigger.solve(E, alpha, beta, self.grid.x, field_threshold=1e4)

    # -- Dark current ------------------------------------------------------------

    def compute_dark_current(self, Vbias: float, E: np.ndarray | None = None) -> dict:
        if E is None:
            _, E, Pe, Ph, _, _ = self.get_fields(Vbias)
        else:
            Pe, Ph = self._ionization_and_trigger(E)

        x = self.grid.x
        J_total = self.dark_current.total_dark_current_density(
            x, E, self.device.material.ni, self.device.material.Eg,
            self.device.material.mc, self.device.material.mh,
        )
        I_dark = float(np.trapezoid(J_total, x) * self.detector_area)
        DCR = self.dark_current.compute_dcr(
            x, E, Pe, self.device.material.ni, self.device.material.Eg,
            self.device.material.mc, self.device.material.mh, self.detector_area,
        )
        return {"J_total": J_total, "I_dark": I_dark, "DCR": DCR,
                "Pe": Pe, "Ph": Ph, "E": E}

    # -- Photocurrent / PDP (delegated) ------------------------------------------

    def compute_photocurrent(
        self, Vbias: float, wavelength: float = 1310e-9, power: float = 1e-6,
        E: np.ndarray | None = None, Pe: np.ndarray | None = None,
        Ph: np.ndarray | None = None, xr: float | None = None,
    ) -> float:
        if E is None or Pe is None or Ph is None or xr is None:
            _, E, _ = self.solve_poisson(Vbias)
            Pe, Ph = self._ionization_and_trigger(E)
            _, xr, _ = self.depletion_width(Vbias)
        return _compute_photocurrent(
            self.grid.x, self.device.layers, self.materials, self.pdp_model,
            self.detector_area, wavelength, power, E, Pe, Ph, xr,
        )

    def compute_pdp_spectrum(
        self, wavelengths: np.ndarray, Vex: float, material_name: str = "InGaAs",
        E: np.ndarray | None = None, Pe: np.ndarray | None = None,
        Ph: np.ndarray | None = None, xr: float | None = None,
    ) -> np.ndarray:
        Vbr, _ = self.find_breakdown(V_start=0, V_max=90, V_step=0.5)
        Vbias = Vbr + Vex
        if E is None or Pe is None or Ph is None or xr is None:
            _, E, _ = self.solve_poisson(Vbias)
            Pe, Ph = self._ionization_and_trigger(E)
            _, xr, _ = self.depletion_width(Vbias)
        return _compute_pdp_spectrum(
            self.grid.x, self.grid.dx, self.device.layers, self.pdp_model,
            wavelengths, Vex, xr, Pe, Ph, material_name,
        )

    # -- Monte Carlo / timing jitter (delegated) ---------------------------------

    def run_mc_ensemble(
        self, Vbias: float, x0: float | None = None, N_sim: int = 100,
        N_threshold: int = 100, dt: float = 1e-15,
    ) -> dict:
        return _run_mc_ensemble(
            self._field_cache, self.solve_poisson, self.depletion_width,
            self.materials, self.transport_material, self.ionization,
            self.grid, Vbias, x0, N_sim, N_threshold, dt,
        )

    def compute_jitter(
        self, Vbias: float, x0: float | None = None, N_sim: int = 100,
        N_threshold: int = 100, dt: float = 1e-15,
    ) -> float:
        return _compute_jitter(
            self._field_cache, self.solve_poisson, self.depletion_width,
            self.materials, self.transport_material, self.ionization,
            self.grid, Vbias, x0, N_sim, N_threshold, dt,
        )

    # -- Self-consistent PIC -----------------------------------------------------

    def build_self_consistent(
        self, Vbias: float, Rq: float = 1e5, Cspad: float = 1e-15
    ) -> SelfConsistentLoop:
        from ..avalanche.breakdown import TriggerCriterion, BreakdownVoltage
        self.mesh = ParticleMesh(self.grid)
        self.transport = DriftDiffusion(self.materials[self.transport_material])
        self.circuit = CircuitSolver(Vbias, Rq, Cspad)

        crit = TriggerCriterion(self.ionization, self.trigger, self.grid)
        bv = BreakdownVoltage(self.poisson_service.poisson, self.grid, crit, V_step=2.0)
        Vbr, results = bv.find(0, max(Vbias + 10, 60))
        self.circuit.Vbr = results[-1]["V"] if results else 0.0

        self.loop = SelfConsistentLoop(
            self.grid, self.device.doping, self.poisson_service.poisson,
            self.mesh, self.transport, self.ionization, self.circuit,
        )
        return self.loop

    def run_pic(self, N_steps: int, inject_x: float | None = None    ) -> list[dict]:
        if self.loop is None:
            raise RuntimeError("Call build_self_consistent() first.")
        if inject_x is not None:
            self.loop.inject_carrier(inject_x)
        return self.loop.run(N_steps)
