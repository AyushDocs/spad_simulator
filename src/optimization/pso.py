from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from ..utils._logging import get_logger

log = get_logger("pso")


class PSOOptimizer:
    """
    Particle Swarm Optimisation.

        v_i = w v_i + c1 r1 (p_i - x_i) + c2 r2 (g - x_i)
        x_i = x_i + v_i
    """

    def __init__(self, n_particles: int, n_dims: int,
                 bounds: List[Tuple[float, float]],
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 max_iter: int = 50) -> None:
        self.n = n_particles
        self.dim = n_dims
        self.bounds = np.asarray(bounds, dtype=float)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

        self.x = np.zeros((n_particles, n_dims))
        self.v = np.zeros((n_particles, n_dims))
        for d in range(n_dims):
            lo, hi = self.bounds[d]
            self.x[:, d] = lo + np.random.rand(n_particles) * (hi - lo)
            self.v[:, d] = (np.random.rand(n_particles) - 0.5) * (hi - lo) * 0.1

        self.p_best = self.x.copy()
        self.p_best_val = -np.inf * np.ones(n_particles)
        self.g_best = self.x[0].copy()
        self.g_best_val = -np.inf
        self.history: List[float] = []

    def optimize(self, cost_function: Callable,
                 verbose: bool = True) -> Tuple[np.ndarray, float, List[float]]:
        for it in range(self.max_iter):
            for i in range(self.n):
                val, _ = cost_function(self.x[i])
                if val > self.p_best_val[i]:
                    self.p_best_val[i] = val
                    self.p_best[i] = self.x[i].copy()
                if val > self.g_best_val:
                    self.g_best_val = val
                    self.g_best = self.x[i].copy()
            self.history.append(self.g_best_val)
            if verbose:
                log.info("PSO iter %3d/%d  best J = %.4e",
                         it + 1, self.max_iter, self.g_best_val)
            r1 = np.random.rand(self.n, self.dim)
            r2 = np.random.rand(self.n, self.dim)
            self.v = (self.w * self.v
                      + self.c1 * r1 * (self.p_best - self.x)
                      + self.c2 * r2 * (self.g_best - self.x))
            self.x += self.v
            for d in range(self.dim):
                self.x[:, d] = np.clip(self.x[:, d],
                                       self.bounds[d, 0], self.bounds[d, 1])

        return self.g_best, self.g_best_val, self.history
