"""Numba-accelerated inner Newton loop (pure-Python fallback if numba absent)."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(f=None, **kwargs):  # type: ignore[no-redef]
        if f is not None:
            return f
        return lambda g: g


@njit(cache=True)
def tridiagonal_solve(dl: np.ndarray, d: np.ndarray,
                      du: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = len(d)
    for i in range(1, n):
        w = dl[i] / d[i - 1]
        d[i] -= w * du[i - 1]
        rhs[i] -= w * rhs[i - 1]
    x = np.zeros(n)
    x[-1] = rhs[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - du[i] * x[i + 1]) / d[i]
    return x


@njit(cache=True)
def _carrier(phi: np.ndarray, phi_n: float, phi_p: float,
             ni: np.ndarray, vth: float, max_arg: float) -> tuple[np.ndarray, np.ndarray]:
    """Electron & hole densities from Boltzmann statistics."""
    arg_n = np.clip((phi - phi_n) / vth, -max_arg, max_arg)
    arg_p = np.clip((phi_p - phi) / vth, -max_arg, max_arg)
    return ni * np.exp(arg_n), ni * np.exp(arg_p)


@njit(cache=True)
def _residual(phi: np.ndarray, n_arr: np.ndarray, p_arr: np.ndarray,
              phi_0: float, phi_L: float,
              dx2: float, eps_half: np.ndarray, net: np.ndarray,
              q_val: float) -> float:
    N = len(phi)
    rho = q_val * (p_arr - n_arr + net)
    ep = eps_half[1:]
    em = eps_half[:-1]
    norm = 0.0
    for i in range(1, N - 1):
        Fi = (ep[i - 1] * (phi[i + 1] - phi[i])
              - em[i - 1] * (phi[i] - phi[i - 1])) / dx2 + rho[i]
        aval = Fi if Fi >= 0 else -Fi
        if aval > norm:
            norm = aval
    F0 = phi[0] - phi_0
    F0a = F0 if F0 >= 0 else -F0
    if F0a > norm:
        norm = F0a
    FL = phi[-1] - phi_L
    FLa = FL if FL >= 0 else -FL
    if FLa > norm:
        norm = FLa
    return norm


@njit(cache=True)
def _solve_delta(phi: np.ndarray, n_arr: np.ndarray, p_arr: np.ndarray,
                 phi_0: float, phi_L: float,
                 dx2: float, eps_half: np.ndarray, net: np.ndarray,
                 vth: float, max_arg: float,
                 q_val: float) -> np.ndarray:
    N = len(phi)
    drho = -q_val / vth * (n_arr + p_arr)

    dl = np.zeros(N)
    d = np.zeros(N)
    du = np.zeros(N)
    ep = eps_half[1:]
    em = eps_half[:-1]
    rhs = np.zeros(N)

    rho = q_val * (p_arr - n_arr + net)
    for i in range(1, N - 1):
        dl[i] = em[i - 1] / dx2
        d[i] = -(ep[i - 1] + em[i - 1]) / dx2 + drho[i]
        du[i] = ep[i - 1] / dx2
        rhs[i] = -((ep[i - 1] * (phi[i + 1] - phi[i])
                    - em[i - 1] * (phi[i] - phi[i - 1])) / dx2 + rho[i])
    d[0] = 1.0
    rhs[0] = -(phi[0] - phi_0)
    d[-1] = 1.0
    rhs[-1] = -(phi[-1] - phi_L)

    return tridiagonal_solve(dl.copy(), d.copy(), du.copy(), rhs.copy())


@njit(cache=True)
def newton_iteration(phi: np.ndarray, Vbias: float,
                     phi_n: float, phi_p: float,
                     phi_0: float, phi_L: float,
                     dx2: float, eps_half: np.ndarray,
                     ni: np.ndarray, vth: float, net: np.ndarray,
                     max_dphi: float, q_val: float) -> tuple[np.ndarray, float, int, float]:
    max_arg = np.log(np.finfo(np.float64).max / np.max(ni))

    n_arr, p_arr = _carrier(phi, phi_n, phi_p, ni, vth, max_arg)
    norm = _residual(phi, n_arr, p_arr, phi_0, phi_L, dx2, eps_half, net, q_val)
    delta = _solve_delta(phi, n_arr, p_arr, phi_0, phi_L, dx2, eps_half, net, vth, max_arg, q_val)

    # Clip delta to max_dphi
    clip = 0.0
    for i in range(len(delta)):
        aval = delta[i] if delta[i] >= 0 else -delta[i]
        if aval > clip:
            clip = aval
    if clip > max_dphi:
        factor = max_dphi / clip
        for i in range(len(delta)):
            delta[i] *= factor

    # Line search
    N = len(phi)
    phi_new = np.zeros(N)
    n_ls = 0
    step = 1.0
    for _ in range(20):
        n_ls += 1
        for i in range(N):
            phi_new[i] = phi[i] + step * delta[i]

        n_arr_new, p_arr_new = _carrier(phi_new, phi_n, phi_p, ni, vth, max_arg)
        nnew = _residual(phi_new, n_arr_new, p_arr_new,
                         phi_0, phi_L, dx2, eps_half, net, q_val)

        if nnew < norm and np.isfinite(nnew):
            phi[:] = phi_new
            return phi, nnew, n_ls, step

        step *= 0.5

    phi[:] = phi_new
    return phi, nnew, n_ls, step
