"""Mission planning: decide *when* to depart and how long to transfer.

Drifting in the initial orbit is free - the chaser's period differs from the
debris', so the phase between them closes on its own. Burning to fix that phase
quickly is expensive. So the planner searches over (departure wait, transfer
time) and picks the cheapest transfer that still finishes inside
``max_mission_time``.

That deadline is the fuel/time knob: generous deadline -> the planner waits for
the natural phasing window and pays near the Hohmann floor; tight deadline ->
it buys speed with delta-v. (See scripts/phasing_tradeoff.py for the curve.)
"""

from dataclasses import dataclass

import numpy as np

from orbdebris.kepler import kepler_propagate
from orbdebris.lambert import lambert
from orbdebris.relative import hill_state_to_eci, mean_motion, semi_major_axis

# Transfer durations to consider, as a fraction of the debris' orbital period.
# Below ~0.3 the transfer is brutally expensive; at/above 1.0 there is no
# single-revolution Lambert solution.
TOF_FRACTION_BOUNDS = (0.30, 0.92)


@dataclass
class TransferPlan:
    t_depart: float  # free drift before the departure burn [s]
    tof: float  # transfer duration [s]
    delta_v: float  # estimated total transfer delta-v [km/s]


def synodic_period(
    chaser_r: np.ndarray,
    chaser_v: np.ndarray,
    debris_r: np.ndarray,
    debris_v: np.ndarray,
    mu: float,
) -> float:
    """Time for the chaser/debris phase angle to lap once. The geometry repeats
    with this period, so searching one synodic period covers every window."""
    n_c = mean_motion(semi_major_axis(chaser_r, chaser_v, mu), mu)
    n_d = mean_motion(semi_major_axis(debris_r, debris_v, mu), mu)
    dn = abs(n_c - n_d)
    if dn < 1e-14:
        return np.inf  # identical orbits never drift relative to each other
    return 2 * np.pi / dn


def _evaluate(
    chaser_r, chaser_v, debris_r, debris_v, handoff_offset, mu, t_dep, tof
) -> float | None:
    """Total delta-v for departing after t_dep and transferring for tof, or
    None if no single-rev transfer exists."""
    cr, cv = kepler_propagate(chaser_r, chaser_v, t_dep, mu)
    dr, dv_deb = kepler_propagate(debris_r, debris_v, t_dep + tof, mu)
    target_r, desired_v = hill_state_to_eci(handoff_offset, np.zeros(3), dr, dv_deb, mu)
    try:
        v1, v2 = lambert(cr, target_r, tof, mu)
    except ValueError:
        return None
    return float(np.linalg.norm(v1 - cv) + np.linalg.norm(v2 - desired_v))


def plan_min_fuel_transfer(
    chaser_r: np.ndarray,
    chaser_v: np.ndarray,
    debris_r: np.ndarray,
    debris_v: np.ndarray,
    handoff_offset: np.ndarray,
    mu: float,
    max_mission_time: float,
    n_depart: int = 90,
    n_tof: int = 40,
) -> TransferPlan:
    """Cheapest transfer completing within max_mission_time.

    Coarse grid search over (t_depart, tof), then a local refinement around the
    best cell - the fuel-optimal window is a narrow notch, so the coarse grid
    locates it and the refinement drops into it.
    """
    if not np.isfinite(max_mission_time) or max_mission_time <= 0:
        raise ValueError(
            f"max_mission_time must be finite and positive, got {max_mission_time}."
        )
    debris_period = 2 * np.pi / mean_motion(semi_major_axis(debris_r, debris_v, mu), mu)
    lo, hi = (f * debris_period for f in TOF_FRACTION_BOUNDS)

    def search(t_deps: np.ndarray, tofs: np.ndarray) -> TransferPlan | None:
        best: TransferPlan | None = None
        for t_dep in t_deps:
            if t_dep < 0:
                continue
            for tof in tofs:
                if tof <= 0 or t_dep + tof > max_mission_time:
                    continue
                dv = _evaluate(
                    chaser_r, chaser_v, debris_r, debris_v, handoff_offset, mu, t_dep, tof
                )
                if dv is not None and (best is None or dv < best.delta_v):
                    best = TransferPlan(float(t_dep), float(tof), dv)
        return best

    coarse = search(np.linspace(0.0, max_mission_time, n_depart), np.linspace(lo, hi, n_tof))
    if coarse is None:
        raise ValueError(
            f"No feasible single-rev transfer within max_mission_time={max_mission_time:.0f} s."
        )

    d_step = max_mission_time / max(n_depart - 1, 1)
    t_step = (hi - lo) / max(n_tof - 1, 1)
    refined = search(
        np.linspace(coarse.t_depart - d_step, coarse.t_depart + d_step, 21),
        np.linspace(max(coarse.tof - t_step, lo), min(coarse.tof + t_step, hi), 21),
    )
    return refined if refined is not None and refined.delta_v < coarse.delta_v else coarse
