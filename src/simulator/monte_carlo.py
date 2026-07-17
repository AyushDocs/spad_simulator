"""Monte Carlo ensemble and timing jitter."""
from __future__ import annotations

from ..transport.drift_diffusion import DriftDiffusion
from ..transport.monte_carlo import MonteCarloTransport
from ..transport.jitter import TimingJitter

from .field_cache import FieldCache


def run_mc_ensemble(
    field_cache: FieldCache,
    solve_poisson,
    depletion_width,
    materials: dict,
    transport_material: str,
    ionization,
    grid,
    Vbias: float,
    x0: float | None = None,
    N_sim: int = 100,
    N_threshold: int = 100,
    dt: float = 1e-15,
) -> dict:
    """Run Monte Carlo avalanche ensemble."""
    phi, E, _ = solve_poisson(Vbias)
    if x0 is None:
        xl, xr, _ = depletion_width(Vbias)
        x0 = (xl + xr) / 2.0 if xr > xl else grid.x[grid.no_of_nodes // 2]
    transport = DriftDiffusion(materials[transport_material])
    mc_sim = MonteCarloSimulator(transport, ionization, grid)
    return mc_sim.run_ensemble(N_sim, x0, E, N_threshold=N_threshold, dt=dt)


def compute_jitter(
    field_cache: FieldCache,
    solve_poisson,
    depletion_width,
    materials: dict,
    transport_material: str,
    ionization,
    grid,
    Vbias: float,
    x0: float | None = None,
    N_sim: int = 100,
    N_threshold: int = 100,
    dt: float = 1e-15,
) -> float:
    """Compute timing jitter (std of detection times)."""
    ens = run_mc_ensemble(
        field_cache, solve_poisson, depletion_width, materials,
        transport_material, ionization, grid,
        Vbias, x0, N_sim, N_threshold, dt,
    )
    t = TimingJitter.extract_detection_times(ens)
    return float(TimingJitter.statistics(t)["std"])
