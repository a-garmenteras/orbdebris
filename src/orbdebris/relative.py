"""Clohessy-Wiltshire (Hill's) relative-motion toolkit.

All relative quantities live in the target's rotating LVLH/Hill frame:
    x = radial (outward), y = along-track (direction of motion),
    z = cross-track (orbit normal).
"""

import numpy as np


def mean_motion(a: float, mu: float) -> float:
    """Orbital mean motion n = sqrt(mu / a^3) [rad/s]."""
    return np.sqrt(mu / a**3)


def semi_major_axis(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    """Semi-major axis from a Cartesian state via vis-viva (energy)."""
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    return 1.0 / (2.0 / r_norm - v_norm**2 / mu)


def eci_to_hill(target_r: np.ndarray, target_v: np.ndarray, mu: float) -> tuple[np.ndarray, float]:
    """Build the ECI->Hill rotation Q (rows = x_hat, y_hat, z_hat) and the
    frame's mean motion n. r_hill = Q @ (r_eci - target_r_eci)."""
    x_hat = target_r / np.linalg.norm(target_r)
    h = np.cross(target_r, target_v)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    q = np.vstack([x_hat, y_hat, z_hat])
    n = mean_motion(semi_major_axis(target_r, target_v, mu), mu)
    return q, n


def relative_state(
    chaser_r: np.ndarray,
    chaser_v: np.ndarray,
    target_r: np.ndarray,
    target_v: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Chaser state relative to target, in Hill coordinates.

    The velocity carries the -omega x r correction because the Hill frame
    rotates: the CW state uses the velocity as seen in the rotating frame.
    """
    q, n = eci_to_hill(target_r, target_v, mu)
    rel_r = q @ (chaser_r - target_r)
    omega = np.array([0.0, 0.0, n])
    rel_v = q @ (chaser_v - target_v) - np.cross(omega, rel_r)
    return rel_r, rel_v


def hill_state_to_eci(
    rel_r: np.ndarray,
    rel_v: np.ndarray,
    target_r: np.ndarray,
    target_v: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of relative_state: recover the chaser's ECI state."""
    q, n = eci_to_hill(target_r, target_v, mu)
    omega = np.array([0.0, 0.0, n])
    chaser_r = target_r + q.T @ rel_r
    chaser_v = target_v + q.T @ (rel_v + np.cross(omega, rel_r))
    return chaser_r, chaser_v


def cw_stm(t: float, n: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clohessy-Wiltshire state-transition matrix blocks (Phi_rr, Phi_rv,
    Phi_vr, Phi_vv), each 3x3, propagating a Hill-frame relative state by t."""
    s, c = np.sin(n * t), np.cos(n * t)
    nt = n * t

    phi_rr = np.array(
        [
            [4 - 3 * c, 0.0, 0.0],
            [6 * (s - nt), 1.0, 0.0],
            [0.0, 0.0, c],
        ]
    )
    phi_rv = np.array(
        [
            [s / n, 2 * (1 - c) / n, 0.0],
            [-2 * (1 - c) / n, (4 * s - 3 * nt) / n, 0.0],
            [0.0, 0.0, s / n],
        ]
    )
    phi_vr = np.array(
        [
            [3 * n * s, 0.0, 0.0],
            [-6 * n * (1 - c), 0.0, 0.0],
            [0.0, 0.0, -n * s],
        ]
    )
    phi_vv = np.array(
        [
            [c, 2 * s, 0.0],
            [-2 * s, 4 * c - 3, 0.0],
            [0.0, 0.0, c],
        ]
    )
    return phi_rr, phi_rv, phi_vr, phi_vv


def cw_propagate(
    rel_r: np.ndarray, rel_v: np.ndarray, t: float, n: float
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a Hill-frame relative state by t using the CW STM."""
    phi_rr, phi_rv, phi_vr, phi_vv = cw_stm(t, n)
    return phi_rr @ rel_r + phi_rv @ rel_v, phi_vr @ rel_r + phi_vv @ rel_v


def two_impulse_transfer(
    rel_r: np.ndarray, rel_v: np.ndarray, target_rel_r: np.ndarray, t: float, n: float
) -> tuple[np.ndarray, np.ndarray]:
    """Two-impulse CW rendezvous, in Hill-frame velocity components.

    Returns (dv1, dv2): dv1 applied now sets up a transfer arriving at
    target_rel_r after time t; dv2 applied on arrival nulls the relative
    velocity (parking at the target point). Phi_rv is singular when t is a
    whole number of orbital periods; pick t away from those.
    """
    phi_rr, phi_rv, phi_vr, phi_vv = cw_stm(t, n)
    v_plus = np.linalg.solve(phi_rv, target_rel_r - phi_rr @ rel_r)
    dv1 = v_plus - rel_v
    v_arrival = phi_vr @ rel_r + phi_vv @ v_plus
    dv2 = -v_arrival
    return dv1, dv2


def hill_dv_to_eci(dv_hill: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate an impulsive delta-v from Hill components to ECI. Valid for an
    instantaneous burn (position fixed, so no omega x r term)."""
    return q.T @ dv_hill
