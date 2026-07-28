"""Simulation engine: propagate two bodies while a guidance policy issues
impulsive burns to the chaser (satellite).

A ``guidance`` is any callable
``(t, sat_r, sat_v, debris_r, debris_v) -> (dv_eci, coast_duration)`` returning
the impulse to apply to the satellite now and how long to coast before the next
decision. RendezvousPolicy.decide and MissionPolicy.decide both fit this
interface, as would a hand-written closure (see tests for examples).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from orbdebris.dynamics import propagate
from orbdebris.relative import relative_state

Guidance = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, float],
]


@dataclass
class ScenarioResult:
    t: np.ndarray
    sat_r: np.ndarray
    sat_v: np.ndarray
    debris_r: np.ndarray
    debris_v: np.ndarray
    separation: np.ndarray
    rel_r: np.ndarray  # chaser position relative to debris, in the Hill frame [N, 3]
    burn_times: np.ndarray
    # Magnitudes [km/s] of the burns at burn_times, in the same order. This is
    # the engine's unlabelled record of what was *actually flown*; the policies
    # separately record the same burns into a FuelLedger with phase labels, and
    # the two are cross-checked (see tests/test_mission.py).
    burn_dv: np.ndarray


def simulate(
    sat_r0: np.ndarray,
    sat_v0: np.ndarray,
    debris_r0: np.ndarray,
    debris_v0: np.ndarray,
    guidance: Guidance,
    t_final: float,
    mu: float,
    dt: float = 60.0,
    max_samples_per_segment: int = 400,
) -> ScenarioResult:
    """Run the propagate/decide/actuate loop until t_final, sampling every ~dt.

    max_samples_per_segment caps the samples recorded per coast, so a long quiet
    drift is recorded coarsely while short active segments keep the full ~dt
    resolution. This costs no accuracy: t_eval only selects output points, and
    the integrator adapts its own internal steps regardless.
    """
    t_chunks, sat_r_chunks, sat_v_chunks = [], [], []
    debris_r_chunks, debris_v_chunks = [], []
    burn_times: list[float] = []
    burn_dv: list[float] = []

    sat_r, sat_v = np.asarray(sat_r0, float), np.asarray(sat_v0, float)
    debris_r, debris_v = np.asarray(debris_r0, float), np.asarray(debris_v0, float)
    t = 0.0
    first = True

    while t < t_final - 1e-9:
        dv, coast = guidance(t, sat_r, sat_v, debris_r, debris_v)
        sat_v = sat_v + dv
        dv_mag = float(np.linalg.norm(dv))
        if dv_mag > 0:
            burn_times.append(t)
            burn_dv.append(dv_mag)

        coast = min(coast, t_final - t)
        n_samp = min(max(2, int(round(coast / dt)) + 1), max_samples_per_segment)
        t_eval = np.linspace(0.0, coast, n_samp)

        _, sat_r_seg, sat_v_seg = propagate(sat_r, sat_v, (0.0, coast), t_eval, mu)
        _, debris_r_seg, debris_v_seg = propagate(debris_r, debris_v, (0.0, coast), t_eval, mu)

        # Drop the duplicated first sample on every segment after the first.
        sl = slice(None) if first else slice(1, None)
        t_chunks.append(t + t_eval[sl])
        sat_r_chunks.append(sat_r_seg[sl])
        sat_v_chunks.append(sat_v_seg[sl])
        debris_r_chunks.append(debris_r_seg[sl])
        debris_v_chunks.append(debris_v_seg[sl])
        first = False

        sat_r, sat_v = sat_r_seg[-1], sat_v_seg[-1]
        debris_r, debris_v = debris_r_seg[-1], debris_v_seg[-1]
        t += coast

    t_all = np.concatenate(t_chunks)
    sat_r_all = np.concatenate(sat_r_chunks)
    sat_v_all = np.concatenate(sat_v_chunks)
    debris_r_all = np.concatenate(debris_r_chunks)
    debris_v_all = np.concatenate(debris_v_chunks)
    separation = np.linalg.norm(sat_r_all - debris_r_all, axis=1)

    rel_r = np.array(
        [
            relative_state(sat_r_all[k], sat_v_all[k], debris_r_all[k], debris_v_all[k], mu)[0]
            for k in range(len(t_all))
        ]
    )

    return ScenarioResult(
        t_all,
        sat_r_all,
        sat_v_all,
        debris_r_all,
        debris_v_all,
        separation,
        rel_r,
        np.array(burn_times),
        np.array(burn_dv),
    )


