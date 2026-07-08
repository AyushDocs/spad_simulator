"""SPAD/APD simulator facade — wires subsystems together."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import numpy as np

from .core.device import Device
from .core.layer import Layer
from .core.constants import h, c
from .poisson.solver import PoissonSolver
from .poisson.field import DepletionWidth
from .poisson.service import PoissonService
from .avalanche.breakdown import TriggerCriterion, CurrentCriterion
from .avalanche.ionization import IonizationCoefficients, OkutoCrowellModel
from .avalanche.trigger import TriggerSolver
from .avalanche.dark_current import DarkCurrentModel
from .avalanche.pdp import PDPModel
from .transport.drift_diffusion import DriftDiffusion
from .transport.monte_carlo import MonteCarloSimulator
from .transport.jitter import TimingJitter
from .self_consistent.particle_mesh import ParticleMesh
from .self_consistent.circuit import CircuitSolver
from .self_consistent.loop import SelfConsistentLoop
from .utils._exceptions import ConfigError
from .utils._logging import get_logger

log = get_logger("simulator")


class SPADSimulator:
    """
    Top-level orchestrator for SPAD/APD simulation.

    Constructs all subsystem objects from a ``Device`` and provides
    convenience methods that combine Poisson, avalanche, and current
    computations.  No cached bias state — caller provides guesses.
    """

    def __init__(self, device: Device, detector_area: float = 1e-6,
                 transport_material: str = "InP", cache_maxlen: int = 200) -> None:
        self.device = device
        self.grid = self.device.grid
        self.materials = self.device.materials
        self.T = self.device.T
        self.detector_area = detector_area
        self.transport_material = transport_material

        poisson_solver = PoissonSolver(
            self.grid, self.T, self.device.doping,
            self.device.material.eps, self.device.material.ni,
            max_iter=200)
        depletion = DepletionWidth(self.grid)
        self.poisson_service = PoissonService(poisson_solver, self.grid, depletion)

        self.ionization = IonizationCoefficients(
            OkutoCrowellModel(),
            self.materials,
            Eg_grid=self.device.material.Eg,
            Eth_grid=self.device.material.Eth,
            mc_grid=self.device.material.mc,
            mh_grid=self.device.material.mh,
            E_ie_grid=self.device.material.E_ie,
            E_ih_grid=self.device.material.E_ih,
            grid_x=self.grid.x,
            mat_names=self.device.material.mat_name,
            T=self.T)
        self.trigger = TriggerSolver(self.grid)
        self.dark_current = DarkCurrentModel(
            T=self.T,
            Eg_grid=self.device.material.Eg,
            mc_grid=self.device.material.mc,
            mh_grid=self.device.material.mh,
            grid_x=self.grid.x,
            tau_n_grid=self.device.material.tau_n,
            tau_p_grid=self.device.material.tau_p)
        self.pdp_model = PDPModel(self.materials)

        self.transport: DriftDiffusion | None = None
        self.mc_sim: MonteCarloSimulator | None = None
        self.mesh: ParticleMesh | None = None
        self.circuit: CircuitSolver | None = None
        self.loop: SelfConsistentLoop | None = None

        self._Vbr: float | None = None
        self._field_cache: OrderedDict[float, tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray,
                                                     float, float]] = OrderedDict()
        self._cache_maxlen = cache_maxlen

    def set_layers(self, layers: List[Layer]) -> None:
        self.device = Device(layers, self.materials, self.grid.no_of_nodes)
        self.grid = self.device.grid
        self._rebuild()

    def set_doping(self, layer_specs: List[dict]) -> None:
        from .core.doping import DopingProfile
        new_doping = DopingProfile(layer_specs)
        self.device.doping = new_doping  # type: ignore[misc]
        self._rebuild()

    def _rebuild(self) -> None:
        poisson_solver = PoissonSolver(
            self.grid, self.T, self.device.doping,
            self.device.material.eps, self.device.material.ni,
            max_iter=200)
        depletion = DepletionWidth(self.grid)
        self.poisson_service = PoissonService(poisson_solver, self.grid, depletion)
        self.ionization = IonizationCoefficients(
            OkutoCrowellModel(),
            self.materials,
            Eg_grid=self.device.material.Eg,
            Eth_grid=self.device.material.Eth,
            mc_grid=self.device.material.mc,
            mh_grid=self.device.material.mh,
            E_ie_grid=self.device.material.E_ie,
            E_ih_grid=self.device.material.E_ih,
            grid_x=self.grid.x,
            mat_names=self.device.material.mat_name,
            T=self.T)
        self.trigger = TriggerSolver(self.grid)
        self.dark_current = DarkCurrentModel(
            T=self.T,
            Eg_grid=self.device.material.Eg,
            mc_grid=self.device.material.mc,
            mh_grid=self.device.material.mh,
            grid_x=self.grid.x,
            tau_n_grid=self.device.material.tau_n,
            tau_p_grid=self.device.material.tau_p)
        self._Vbr = None
        self._field_cache.clear()

    def get_fields(self, Vbias: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray, float, float]:
        Vkey = round(Vbias, 6)
        cached = self._field_cache.get(Vkey)
        if cached is not None:
            self._field_cache.move_to_end(Vkey)
            return cached

        guess: np.ndarray | None = None
        if self._field_cache:
            biases = np.array(list(self._field_cache.keys()))
            nearest = biases[np.argmin(np.abs(biases - Vbias))]
            if abs(nearest - Vbias) < 10.0:
                guess = self._field_cache[nearest][0]

        phi, E, _ = self.poisson_service.solve(Vbias, guess=guess)
        Pe, Ph = self._ionization_and_trigger(E)
        xl, xr, _ = self.poisson_service.depletion.from_field(E)

        result = (phi, E, Pe, Ph, xl, xr)
        self._field_cache[Vkey] = result
        if len(self._field_cache) > self._cache_maxlen:
            self._field_cache.popitem(last=False)
        return result

    def solve_poisson(self, Vbias: float,
                      phi_n: float | None = None, phi_p: float = 0.0,
                      guess: np.ndarray | None = None
                      ) -> Tuple[np.ndarray, np.ndarray, dict]:
        cached = self._field_cache.get(round(Vbias, 6))
        if cached is not None:
            return cached[0], self.grid.gradient(cached[0]), {"converged": True, "cached": True}

        if guess is None and self._field_cache:
            biases = np.array(list(self._field_cache.keys()))
            nearest = biases[np.argmin(np.abs(biases - Vbias))]
            if abs(nearest - Vbias) < 10.0:
                guess = self._field_cache[nearest][0]

        phi, E, info = self.poisson_service.solve(Vbias, phi_n, phi_p, guess)
        return phi, E, info

    def find_breakdown(self, V_start: float = 0.0,
                       V_max: float = 100.0,
                       V_step: float = 0.1,
                       force: bool = False,
                       criterion: str = "current",
                       I_threshold: float = 1e-6
                       ) -> Tuple[float | None, List[dict]]:
        if self._Vbr is not None and not force:
            return self._Vbr, []

        if criterion == "current":
            def _compute_current(V: float, phi: np.ndarray,
                                 E: np.ndarray) -> float:
                alpha = self.ionization.alpha(E)
                beta = self.ionization.beta(E)
                Pe, Ph = self.trigger.solve(E, alpha, beta, self.grid.x)
                Ptr = Pe + Ph - Pe * Ph
                Ptr_max = float(np.max(Ptr))
                M = min(1.0 / (1.0 - Ptr_max + 1e-15), 10000.0)
                J_total = self.dark_current.total_dark_current_density(
                    self.grid.x, E, self.device.material.ni,
                    self.device.material.Eg,
                    self.device.material.mc, self.device.material.mh)
                # Trapezoidal rule: O(h²), handles non-uniform grid near heterojunctions
                I_primary = float(np.trapezoid(J_total, self.grid.x)
                                  * self.detector_area)
                return I_primary * M

            crit = CurrentCriterion(_compute_current, I_threshold=I_threshold)
        else:
            crit = TriggerCriterion(self.ionization, self.trigger, self.grid)

        Vbr, results = self.poisson_service.find_breakdown(
            V_start, V_max, crit, V_step)
        self._Vbr = Vbr
        return Vbr, results

    def solve_trigger(self, Vbias: float
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        phi, E, _ = self.solve_poisson(Vbias)
        alpha = self.ionization.alpha(E)
        beta = self.ionization.beta(E)
        Pe, Ph = self.trigger.solve(E, alpha, beta, self.grid.x)
        return Pe, Ph, E

    def depletion_width(self, Vbias: float
                        ) -> Tuple[float, float, float]:
        return self.poisson_service.depletion_width(Vbias)

    def _ionization_and_trigger(self, E: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
        alpha = self.ionization.alpha(E)
        beta = self.ionization.beta(E)
        Pe, Ph = self.trigger.solve(E, alpha, beta, self.grid.x)
        return Pe, Ph

    def _trigger_for_pdp(self, E: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
        alpha = self.ionization.alpha(E)
        beta = self.ionization.beta(E)
        Pe, Ph = self.trigger.solve(E, alpha, beta, self.grid.x,
                                    field_threshold=1e4)
        return Pe, Ph

    def compute_dark_current(self, Vbias: float,
                             E: np.ndarray | None = None) -> dict:
        if E is None:
            _, E, Pe, Ph, _, _ = self.get_fields(Vbias)
        else:
            Pe, Ph = self._ionization_and_trigger(E)

        x = self.grid.x
        ni_arr = self.device.material.ni
        Eg_arr = self.device.material.Eg
        mc_arr = self.device.material.mc
        mh_arr = self.device.material.mh

        J_total = self.dark_current.total_dark_current_density(
            x, E, ni_arr, Eg_arr, mc_arr, mh_arr)
        # Trapezoidal rule: O(h²), handles non-uniform grid near heterojunctions
        I_dark = float(np.trapezoid(J_total, x) * self.detector_area)
        DCR = self.dark_current.compute_dcr(
            x, E, Pe, ni_arr, Eg_arr, mc_arr, mh_arr, self.detector_area)
        return {
            "J_total": J_total, "I_dark": I_dark, "DCR": DCR,
            "Pe": Pe, "Ph": Ph, "E": E
        }

    def compute_photocurrent(self, Vbias: float,
                             wavelength: float = 1310e-9,
                             power: float = 1e-6,
                             E: np.ndarray | None = None,
                             Pe: np.ndarray | None = None,
                             Ph: np.ndarray | None = None,
                             xr: float | None = None) -> float:
        if E is None or Pe is None or Ph is None or xr is None:
            phi, E_loc, _ = self.solve_poisson(Vbias)
            Pe_loc, Ph_loc = self._ionization_and_trigger(E_loc)
            _, xr_loc, _ = self.depletion_width(Vbias)
        else:
            E_loc, Pe_loc, Ph_loc, xr_loc = E, Pe, Ph, xr

        dead_zone_layers, absorber = self.pdp_model.find_absorber(
            self.device.layers, "InGaAs")
        dead_zone = sum(l.thickness for l in dead_zone_layers)
        L_abs = max(min(xr_loc - dead_zone, absorber.thickness), 0.0)

        if L_abs <= 0:
            return 0.0

        absorber_start = dead_zone
        absorber_end = dead_zone + L_abs
        x = self.grid.x

        alpha_arr = np.array([
            self.materials[lyr.material].absorption_coefficient(wavelength)
            for lyr in self.device.layers
        ])
        alpha_grid = np.zeros_like(x)
        xs = 0.0
        for lyr, alpha_val in zip(self.device.layers, alpha_arr):
            xe = xs + lyr.thickness
            mask = (x >= xs - 1e-16) & (x <= xe + 1e-16)
            alpha_grid[mask] = alpha_val
            xs = xe

        Eph = h * c / wavelength
        phi_photon = power / (Eph * self.detector_area)

        J_photo = self.pdp_model.photocurrent_density(
            x, alpha_grid, phi_photon, absorber_start, absorber_end)
        # Trapezoidal rule: O(h²), handles non-uniform grid near heterojunctions
        I_primary = float(np.trapezoid(J_photo, x) * self.detector_area)

        if I_primary <= 0:
            return 0.0

        Ptr_abs = np.interp(
            np.linspace(absorber_start, absorber_end, 10),
            x, Pe_loc + Ph_loc - Pe_loc * Ph_loc)
        Ptr_avg = float(np.mean(Ptr_abs))
        M_raw = 1.0 / (1.0 - Ptr_avg + 1e-15)
        M = min(M_raw, 10000.0)
        return I_primary * M

    def compute_pdp_spectrum(self, wavelengths: np.ndarray,
                             Vex: float,
                             material_name: str = "InGaAs",
                             E: np.ndarray | None = None,
                             Pe: np.ndarray | None = None,
                             Ph: np.ndarray | None = None,
                             xr: float | None = None) -> np.ndarray:
        Vbr, _ = self.find_breakdown(V_start=0, V_max=90, V_step=0.5)
        Vbias = Vbr + Vex

        if E is None or Pe is None or Ph is None or xr is None:
            _, E_loc, _ = self.solve_poisson(Vbias)
            Pe_loc, Ph_loc = self._ionization_and_trigger(E_loc)
            _, xr_loc, _ = self.depletion_width(Vbias)
        else:
            E_loc, Pe_loc, Ph_loc, xr_loc = E, Pe, Ph, xr

        Ptr = Pe_loc + Ph_loc - Pe_loc * Ph_loc

        dead_zone_layers, absorber = self.pdp_model.find_absorber(
            self.device.layers, material_name)
        dead_zone = sum(l.thickness for l in dead_zone_layers)
        L_abs = max(min(xr_loc - dead_zone, absorber.thickness), 0.0)

        if L_abs <= 0:
            return np.zeros(len(wavelengths))

        absorber_start = dead_zone
        absorber_end = dead_zone + L_abs
        mask = (self.grid.x >= absorber_start) & (self.grid.x <= absorber_end)
        xx = self.grid.x[mask] - absorber_start

        pdp_vals = []
        for lam in wavelengths:
            trans = self.pdp_model.dead_zone_transmission(lam, dead_zone_layers)
            pdp = self.pdp_model.pdp_integral(
                lam, xx, Ptr[mask], trans, self.grid.dx,
                material_name=material_name)
            pdp_vals.append(pdp)
        return np.array(pdp_vals)

    def run_mc_ensemble(self, Vbias: float, x0: float | None = None,
                        N_sim: int = 100, N_threshold: int = 100,
                        dt: float = 1e-15) -> dict:
        phi, E, _ = self.solve_poisson(Vbias)
        if x0 is None:
            xl, xr, _ = self.depletion_width(Vbias)
            x0 = (xl + xr) / 2.0 if xr > xl else self.grid.x[self.grid.no_of_nodes // 2]
        self.transport = DriftDiffusion(self.materials[self.transport_material])
        self.mc_sim = MonteCarloSimulator(
            self.transport, self.ionization, self.grid)
        return self.mc_sim.run_ensemble(N_sim, x0, E,
                                        N_threshold=N_threshold, dt=dt)

    def compute_jitter(self, Vbias: float, x0: float | None = None,
                       N_sim: int = 100, N_threshold: int = 100,
                       dt: float = 1e-15) -> float:
        ens = self.run_mc_ensemble(Vbias, x0, N_sim, N_threshold, dt)
        t = TimingJitter.extract_detection_times(ens)
        return float(TimingJitter.statistics(t)["std"])

    def build_self_consistent(self, Vbias: float, Rq: float = 1e5,
                              Cspad: float = 1e-15
                              ) -> SelfConsistentLoop:
        self.mesh = ParticleMesh(self.grid)
        self.transport = DriftDiffusion(self.materials[self.transport_material])
        self.circuit = CircuitSolver(Vbias, Rq, Cspad)

        crit = TriggerCriterion(self.ionization, self.trigger, self.grid)
        Vbr, results = self.poisson_service.find_breakdown(
            0, max(Vbias + 10, 60), crit, V_step=2.0)
        self.circuit.Vbr = results[-1]["V"] if results else 0.0

        self.loop = SelfConsistentLoop(
            self.grid, self.device.doping,
            self.poisson_service.poisson,
            self.mesh, self.transport, self.ionization,
            self.circuit)
        return self.loop

    def run_pic(self, N_steps: int, inject_x: float | None = None
                ) -> List[dict]:
        if self.loop is None:
            raise RuntimeError("Call build_self_consistent() first.")
        if inject_x is not None:
            self.loop.inject_carrier(inject_x)
        return self.loop.run(N_steps)
