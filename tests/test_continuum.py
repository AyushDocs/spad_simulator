"""Tests for the Poisson-only ContinuumSolver."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.grid import Grid1D
from src.core.doping import DopingProfile, LayerSpec
from src.core.material_grid import MaterialGrid
from src.transport.continuum import (
    ContinuumSolver,
    _equilibrium_np,
    _harmonic_mean,
    _solve_tridiagonal,
)


class TestHarmonicMean:
    def test_constant(self):
        a = np.full(10, 5.0)
        h = _harmonic_mean(a)
        assert np.allclose(h, 5.0)

    def test_symmetric(self):
        a = np.array([1.0, 100.0, 1.0])
        h = _harmonic_mean(a)
        assert h[1] < 2.0

    def test_different_values(self):
        a = np.array([2.0, 8.0])
        h = _harmonic_mean(a)
        assert h[0] == pytest.approx(3.2)


class TestEquilibriumNP:
    def test_n_type(self):
        n, p = _equilibrium_np(1e17, 1e10)
        assert n == pytest.approx(1e17, rel=1e-3)
        assert p == pytest.approx(1e3, rel=1e-2)
        assert n * p == pytest.approx(1e20, rel=1e-3)

    def test_p_type(self):
        n, p = _equilibrium_np(-1e17, 1e10)
        assert p == pytest.approx(1e17, rel=1e-3)
        assert n == pytest.approx(1e3, rel=1e-2)
        assert n * p == pytest.approx(1e20, rel=1e-3)

    def test_intrinsic(self):
        n, p = _equilibrium_np(0.0, 1e10)
        assert n == pytest.approx(1e10, rel=1e-3)
        assert p == pytest.approx(1e10, rel=1e-3)


class TestTridiagonal:
    def test_identity(self):
        N = 5
        a, b, c, rhs = np.zeros(N), np.ones(N), np.zeros(N), np.arange(N, dtype=float)
        b[0] = b[-1] = 1.0
        x = _solve_tridiagonal(a, b, c, rhs)
        assert np.allclose(x, rhs)

    def test_simple(self):
        N = 5
        a = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        b = np.array([1.0, -2.0, -2.0, -2.0, 1.0])
        c = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        rhs = np.array([1.0, 0.0, 0.0, 0.0, 2.0])
        x = _solve_tridiagonal(a, b, c, rhs)
        # Build matrix explicitly and check M @ x == rhs
        M = np.diag(b) + np.diag(a[1:], k=-1) + np.diag(c[:-1], k=1)
        assert np.allclose(M @ x, rhs)


@pytest.fixture
def simple_pn_grid() -> Grid1D:
    return Grid1D(L=6e-4, N=201)


@pytest.fixture
def pn_doping() -> DopingProfile:
    return DopingProfile([
        LayerSpec(type="acceptor", A=5e17, m=0.0, x0=0.0,
                  x_start=0.0, x_end=3e-4),
        LayerSpec(type="donor", A=1e17, m=0.0, x0=3e-4,
                  x_start=3e-4, x_end=6e-4),
    ])


@pytest.fixture
def pn_material(simple_pn_grid: Grid1D) -> MaterialGrid:
    N = simple_pn_grid.no_of_nodes
    ni_val = 1e10
    eps_r = 12.5
    eps0_val = 8.854187817e-14
    eps_val = float(eps_r * eps0_val)
    return MaterialGrid(
        eps=np.full(N, eps_val),
        ni=np.full(N, ni_val),
        Eg=np.full(N, 1.35),
        Eth=np.full(N, 2.0),
        mu_n=np.full(N, 5400.0),
        mu_p=np.full(N, 200.0),
        vsat_n=np.full(N, 1e7),
        vsat_p=np.full(N, 1e7),
        mc=np.full(N, 0.077),
        mh=np.full(N, 0.64),
        E_ie=np.full(N, 2.1),
        E_ih=np.full(N, 2.1),
        tau_n=np.full(N, 1e-6),
        tau_p=np.full(N, 1e-6),
        Nc=np.full(N, 5.7e17),
        Nv=np.full(N, 1.1e19),
        mat_name=np.full(N, "InP", dtype="<U12"),
    )


@pytest.fixture
def continuum(simple_pn_grid, pn_doping, pn_material):
    return ContinuumSolver(simple_pn_grid, pn_doping, pn_material, T=300.0)


class TestContinuumSolver:
    def test_solve_zero_bias(self, continuum: ContinuumSolver):
        result = continuum.solve(0.0)
        assert np.all(np.isfinite(result["phi"]))
        assert np.all(np.isfinite(result["n"]))
        assert np.all(np.isfinite(result["p"]))

    def test_np_product(self, continuum: ContinuumSolver):
        result = continuum.solve(0.0)
        np_prod = result["n"] * result["p"]
        assert np.allclose(np_prod, continuum.ni**2, rtol=1e-2)

    def test_carrier_densities_smooth(self, continuum: ContinuumSolver):
        result = continuum.solve(0.0)
        assert np.all(result["n"] > 0)
        assert np.all(result["p"] > 0)

    def test_quasi_fermi_levels_flat(self, continuum: ContinuumSolver):
        result = continuum.solve(0.0)
        assert np.ptp(result["phi_n"]) < 0.1
        assert np.ptp(result["phi_p"]) < 0.1

    def test_solve_reverse_bias(self, continuum: ContinuumSolver):
        result = continuum.solve(5.0)
        assert np.all(np.isfinite(result["phi"]))

    def test_solve_30v(self, continuum: ContinuumSolver):
        result = continuum.solve(30.0)
        assert np.all(np.isfinite(result["phi"]))
        # Poisson-only: with flat QFLs, np = ni² everywhere (no depletion
        # without quasi-Fermi level splitting). Just verify the solve runs.
        assert np.allclose(result["n"] * result["p"], continuum.ni**2, rtol=1e-2)
