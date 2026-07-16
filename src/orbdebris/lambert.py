"""Lambert's problem: find the two-body transfer orbit connecting two position
vectors in a given time of flight.

Universal-variable formulation (Curtis, *Orbital Mechanics for Engineering
Students*, Algorithm 5.2). Because this assumes pure Keplerian two-body motion
- exactly what ``dynamics.propagate`` integrates - a Lambert solution flown
through our propagator arrives on target to integration tolerance.
"""

import numpy as np

from orbdebris.kepler import stumpff_c as _stumpff_c
from orbdebris.kepler import stumpff_s as _stumpff_s


def lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    prograde: bool = True,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Lambert's problem. Returns (v1, v2): the departure velocity at r1
    and arrival velocity at r2 on the connecting transfer orbit.

    Raises ValueError if the transfer angle is ~0 or ~180 deg (the transfer
    plane is undefined there) or if the iteration fails to converge.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    r1n = np.linalg.norm(r1)
    r2n = np.linalg.norm(r2)

    cross = np.cross(r1, r2)
    dtheta = np.arccos(np.clip(np.dot(r1, r2) / (r1n * r2n), -1.0, 1.0))
    # Resolve the transfer angle into (0, 2pi) using the orbit-normal sense.
    if prograde:
        if cross[2] < 0:
            dtheta = 2 * np.pi - dtheta
    else:
        if cross[2] >= 0:
            dtheta = 2 * np.pi - dtheta

    if abs(np.sin(dtheta)) < 1e-6:
        raise ValueError(
            f"Transfer angle {np.degrees(dtheta):.3f} deg is ~0 or ~180 deg; "
            "the transfer plane is undefined (Lambert singularity)."
        )

    a_coef = np.sin(dtheta) * np.sqrt(r1n * r2n / (1 - np.cos(dtheta)))

    def y(z: float) -> float:
        return r1n + r2n + a_coef * (z * _stumpff_s(z) - 1) / np.sqrt(_stumpff_c(z))

    def big_f(z: float) -> float:
        yz = y(z)
        c = _stumpff_c(z)
        s = _stumpff_s(z)
        return (yz / c) ** 1.5 * s + a_coef * np.sqrt(yz) - np.sqrt(mu) * tof

    def big_f_prime(z: float) -> float:
        if z == 0:
            y0 = y(0.0)
            return np.sqrt(2) / 40 * y0**1.5 + a_coef / 8 * (
                np.sqrt(y0) + a_coef * np.sqrt(1 / (2 * y0))
            )
        yz = y(z)
        c = _stumpff_c(z)
        s = _stumpff_s(z)
        term1 = (yz / c) ** 1.5 * (
            1 / (2 * z) * (c - 3 * s / (2 * c)) + 3 * s**2 / (4 * c)
        )
        term2 = a_coef / 8 * (3 * s / c * np.sqrt(yz) + a_coef * np.sqrt(c / yz))
        return term1 + term2

    # Newton iteration from z = 0. Nudge z up until y is positive first, then
    # iterate. Very negative z (deep-hyperbolic) means no single-rev solution
    # for this time of flight; we detect the resulting non-finite values and
    # raise cleanly rather than churning through overflow.
    z = 0.0
    for _ in range(1000):
        if y(z) > 0:
            break
        z += 0.1

    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(max_iter):
            fz = big_f(z)
            if not np.isfinite(fz):
                raise ValueError("Lambert solver did not converge (no single-rev solution).")
            if abs(fz) < tol:
                break
            dfz = big_f_prime(z)
            if not np.isfinite(dfz) or dfz == 0.0:
                raise ValueError("Lambert solver did not converge (degenerate derivative).")
            z = z - fz / dfz
        else:
            raise ValueError("Lambert solver did not converge.")

    yz = y(z)
    f = 1 - yz / r1n
    g = a_coef * np.sqrt(yz / mu)
    g_dot = 1 - yz / r2n

    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    return v1, v2
