"""Analytic two-body (Keplerian) propagation via universal variables.

``dynamics.propagate`` integrates the equations of motion numerically - correct
and general, but far too slow to call thousands of times inside a search. This
module solves the same two-body coast in closed form (Curtis, Algorithm 3.4):
one Newton iteration on the universal anomaly, then Lagrange f/g coefficients.
Same physics, no integration - so it is both exact and fast.

The Stumpff functions C(z) and S(z) live here and are shared with the Lambert
solver.
"""

import numpy as np


def stumpff_c(z: float) -> float:
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    if z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    return 0.5


def stumpff_s(z: float) -> float:
    if z > 0:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / sz**3
    if z < 0:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / sz**3
    return 1.0 / 6.0


def kepler_propagate(
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    mu: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Coast a two-body state by dt. Returns (r, v)."""
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if dt == 0.0:
        return r0.copy(), v0.copy()

    r0n = np.linalg.norm(r0)
    v0n = np.linalg.norm(v0)
    vr0 = np.dot(r0, v0) / r0n
    alpha = 2.0 / r0n - v0n**2 / mu  # reciprocal of semi-major axis

    sqrt_mu = np.sqrt(mu)
    chi = sqrt_mu * abs(alpha) * dt  # initial guess

    for _ in range(max_iter):
        z = alpha * chi**2
        c, s = stumpff_c(z), stumpff_s(z)
        f = (
            r0n * vr0 / sqrt_mu * chi**2 * c
            + (1 - alpha * r0n) * chi**3 * s
            + r0n * chi
            - sqrt_mu * dt
        )
        df = (
            r0n * vr0 / sqrt_mu * chi * (1 - alpha * chi**2 * s)
            + (1 - alpha * r0n) * chi**2 * c
            + r0n
        )
        ratio = f / df
        chi -= ratio
        if abs(ratio) < tol:
            break
    else:
        raise ValueError("Kepler propagation did not converge.")

    z = alpha * chi**2
    c, s = stumpff_c(z), stumpff_s(z)

    f = 1 - chi**2 / r0n * c
    g = dt - chi**3 * s / sqrt_mu
    r = f * r0 + g * v0
    rn = np.linalg.norm(r)

    f_dot = sqrt_mu / (rn * r0n) * (alpha * chi**3 * s - chi)
    g_dot = 1 - chi**2 / rn * c
    v = f_dot * r0 + g_dot * v0
    return r, v
