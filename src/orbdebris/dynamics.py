"""Two-body orbital propagation."""

import numpy as np
from scipy.integrate import solve_ivp


def two_body_eom(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    """State derivative for pure two-body gravity, y = [rx, ry, rz, vx, vy, vz]."""
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r)
    a = -mu * r / r_norm**3
    return np.concatenate([v, a])


def propagate(
    r0: np.ndarray,
    v0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate a two-body orbit. Returns (t, r(t) [N,3], v(t) [N,3])."""
    y0 = np.concatenate([r0, v0])
    sol = solve_ivp(
        two_body_eom,
        t_span,
        y0,
        args=(mu,),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.t, sol.y[:3].T, sol.y[3:].T
