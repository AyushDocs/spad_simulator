"""Device optimization via Particle Swarm Optimization."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from ..simulator import SPADSimulator
from ..optimization.pso import PSOOptimizer
from ..optimization.cost import CostFunction
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ..utils.loaders import PlotConfig
from ._config import PLOT_DIR

log = get_logger()


def run_optimize_device(sim: SPADSimulator, Vbr: float,
                        layer_indices: list[int] | None = None,
                        bounds_cm3: tuple[float, float] = (1e16, 1e18),
                        n_particles: int = 8, max_iter: int = 20,
                        BV_target: float | None = None,
                        plot_cfg: PlotConfig | None = None) -> dict:
    """Run PSO to find optimal doping profile for target breakdown voltage.

    Parameters
    ----------
    sim : SPADSimulator
    Vbr : float
        Current breakdown voltage (informational, not used as target).
    layer_indices : list[int], optional
        Which layer indices to optimize.  Defaults to charge sheet and
        absorber layers (found by material search).
    bounds_cm3 : tuple[float, float]
        Doping search range (cm⁻³).
    n_particles : int
        PSO swarm size.
    max_iter : int
        Maximum PSO iterations.
    BV_target : float, optional
        Target breakdown voltage.  Defaults to ``Vbr`` if not given.

    Returns
    -------
    dict with keys ``best_doping``, ``best_Vbr``, ``history``.
    """
    if plot_cfg and not plot_cfg.is_enabled("optimization"):
        return {}
    if BV_target is None:
        BV_target = Vbr

    # Default: optimize charge sheet and absorber layers
    if layer_indices is None:
        layer_indices = []
        for i, lyr in enumerate(sim.device.layers):
            if lyr.material == "InP" and lyr.doping_type == "donor" and lyr.doping_A > 1e16:
                layer_indices.append(i)  # charge sheet
            elif lyr.material == "InGaAs":
                layer_indices.append(i)  # absorber

    if not layer_indices:
        log.warning("No optimizable layers found")
        return {"best_doping": {}, "best_Vbr": Vbr, "history": []}

    n_dims = len(layer_indices)
    bounds = [bounds_cm3] * n_dims

    cost_fn = CostFunction(BV_target=BV_target)

    def objective(x: np.ndarray) -> tuple[float, dict]:
        doping_params = {str(idx): float(x[d])
                         for d, idx in enumerate(layer_indices)}
        return cost_fn.evaluate(sim, doping_params)

    pso = PSOOptimizer(n_particles=n_particles, n_dims=n_dims,
                       bounds=bounds, max_iter=max_iter)
    g_best, g_best_val, history = pso.optimize(objective, verbose=True)

    best_doping = {str(idx): float(g_best[d])
                   for d, idx in enumerate(layer_indices)}

    log.info(f"  Optimization complete: J={g_best_val:.4e}")
    for d, idx in enumerate(layer_indices):
        log.info(f"    Layer {idx} ({sim.device.layers[idx].material}): "
                 f"{g_best[d]:.2e} cm⁻³")

    # Plot convergence
    if history:
        get_plotter("param_sweep", plot_dir=PLOT_DIR).plot(
            np.arange(1, len(history) + 1), np.array(history),
            xlabel="Iteration", ylabel="Cost J",
            title="PSO Convergence",
            fname="pso_convergence.png")

    return {"best_doping": best_doping, "best_Vbr": g_best_val,
            "history": history, "layer_indices": layer_indices}
