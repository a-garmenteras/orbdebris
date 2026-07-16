import numpy as np

from orbdebris.constants import GM_EARTH, R_EARTH
from orbdebris.relative import eci_to_hill, hill_state_to_eci, mean_motion
from orbdebris.rendezvous import HOLD, RendezvousPolicy
from orbdebris.simulate import simulate
from orbdebris.state import elements_to_state


def _terminal_setup(deadband=np.array([0.02, 0.5, 0.05]), drift_tolerance=1e-7):
    a = R_EARTH + 500.0
    period = 2 * np.pi / mean_motion(a, GM_EARTH)
    debris_r, debris_v = elements_to_state(
        a=a, e=0.0, i=np.radians(51.6), raan=0.0, argp=0.0, nu=0.0, mu=GM_EARTH
    )
    sat_r, sat_v = hill_state_to_eci(
        np.array([0.0, -20.0, 1.0]), np.zeros(3), debris_r, debris_v, GM_EARTH
    )
    hold_offset = np.array([0.0, 1.0, 0.0])
    policy = RendezvousPolicy(
        mu=GM_EARTH,
        hold_offset=hold_offset,
        transfer_time=0.4 * period,
        hold_check_interval=period / 20,
        deadband=deadband,
        drift_tolerance=drift_tolerance,
    )
    return sat_r, sat_v, debris_r, debris_v, policy, hold_offset, period


def test_terminal_rendezvous_closes_and_holds():
    """CW terminal phase on its own: from 20 km behind, close and hold 1 km
    ahead. Independent of the mission/phasing wrapper."""
    sat_r, sat_v, debris_r, debris_v, policy, hold_offset, period = _terminal_setup()
    t_final = 0.4 * period + 2.5 * period

    result = simulate(sat_r, sat_v, debris_r, debris_v, policy.decide, t_final, GM_EARTH, dt=30.0)

    hold_dist = np.linalg.norm(hold_offset)
    assert result.separation[-1] < 0.2 * result.separation[0]

    held = result.separation[result.t > t_final - period]
    assert np.abs(held.mean() - hold_dist) < 0.5
    assert held.max() < hold_dist + 1.0


def test_hold_is_stable_over_a_long_horizon():
    """The hold must not creep: over ~15 orbits the separation stays bounded
    and the radial offset - the thing that actually drives drift - stays tiny."""
    sat_r, sat_v, debris_r, debris_v, policy, _, period = _terminal_setup()
    t_final = 0.4 * period + 15 * period

    result = simulate(sat_r, sat_v, debris_r, debris_v, policy.decide, t_final, GM_EARTH, dt=60.0)

    settled = result.t > 0.4 * period + 1.5 * period
    sep = result.separation[settled]
    radial = result.rel_r[settled][:, 0]

    assert np.ptp(sep) < 0.3  # bounded, not drifting apart
    assert np.abs(radial).max() < 0.02  # radial held tight -> no secular drift

    # No secular trend: the second half must not have walked away from the first.
    half = len(sep) // 2
    assert abs(sep[:half].mean() - sep[half:].mean()) < 0.05


def test_drift_null_burn_enforces_the_cw_no_drift_condition():
    """A radial offset is what causes secular along-track drift. From inside
    the deadband, HOLD must answer with a small along-track-only impulse that
    drives the state onto the CW no-drift condition ydot = -2*n*x, rather than
    an expensive reposition."""
    sat_r, sat_v, debris_r, debris_v, policy, hold_offset, _ = _terminal_setup()
    policy.mode = HOLD

    x = 0.01  # 10 m radial offset: inside the 20 m radial deadband
    rel_r = hold_offset + np.array([x, 0.0, 0.0])
    rel_v = np.zeros(3)
    chaser_r, chaser_v = hill_state_to_eci(rel_r, rel_v, debris_r, debris_v, GM_EARTH)

    q, n = eci_to_hill(debris_r, debris_v, GM_EARTH)
    dv_eci, _ = policy.decide(0.0, chaser_r, chaser_v, debris_r, debris_v)
    dv_hill = q @ dv_eci  # back into Hill components

    assert np.isclose(dv_hill[0], 0.0, atol=1e-12)  # along-track only
    assert np.isclose(dv_hill[2], 0.0, atol=1e-12)
    assert np.isclose(dv_hill[1], -2 * n * x, rtol=1e-9)

    # The CW along-track secular rate is -(6*n*x + 3*ydot); the burn zeroes it.
    ydot_after = rel_v[1] + dv_hill[1]
    assert np.isclose(-(6 * n * x + 3 * ydot_after), 0.0, atol=1e-12)

    # And it is cheap: millimetres per second, not tens of cm/s.
    assert np.linalg.norm(dv_eci) * 1000 < 0.1
