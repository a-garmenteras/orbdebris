import numpy as np

from orbdebris import HANDOFF_OFFSET, HOLD_OFFSET, build_scenario
from orbdebris.constants import GM_EARTH, R_EARTH
from orbdebris.phasing import plan_min_fuel_transfer, synodic_period
from orbdebris.relative import mean_motion
from orbdebris.simulate import simulate
from orbdebris.state import elements_to_state


def _states(chaser_alt=400.0, debris_alt=500.0, phase=np.radians(45.0)):
    inc = np.radians(51.6)
    debris = elements_to_state(
        a=R_EARTH + debris_alt, e=0.0, i=inc, raan=0.0, argp=0.0, nu=0.0, mu=GM_EARTH
    )
    chaser = elements_to_state(
        a=R_EARTH + chaser_alt, e=0.0, i=inc, raan=0.0, argp=0.0, nu=phase, mu=GM_EARTH
    )
    return chaser, debris


def test_generous_deadline_finds_near_hohmann_floor():
    """With time to spare the planner should wait for the natural window and
    pay close to the Hohmann floor (~56 m/s), not brute-force the phase."""
    (cr, cv), (dr, dv) = _states()
    t_syn = synodic_period(cr, cv, dr, dv, GM_EARTH)

    plan = plan_min_fuel_transfer(
        cr, cv, dr, dv, HANDOFF_OFFSET, GM_EARTH, max_mission_time=1.05 * t_syn
    )

    assert plan.delta_v * 1000 < 150.0  # near the ~56 m/s floor
    assert plan.t_depart > 0.5 * t_syn  # it actually chose to wait


def test_tight_deadline_costs_much_more_fuel():
    """The deadline is the fuel/time knob: forcing a fast transfer must cost
    substantially more delta-v than the patient plan."""
    (cr, cv), (dr, dv) = _states()
    period = 2 * np.pi / mean_motion(R_EARTH + 500.0, GM_EARTH)

    urgent = plan_min_fuel_transfer(
        cr, cv, dr, dv, HANDOFF_OFFSET, GM_EARTH, max_mission_time=1.0 * period
    )
    patient = plan_min_fuel_transfer(
        cr,
        cv,
        dr,
        dv,
        HANDOFF_OFFSET,
        GM_EARTH,
        max_mission_time=1.05 * synodic_period(cr, cv, dr, dv, GM_EARTH),
    )

    assert urgent.delta_v > 5 * patient.delta_v
    assert urgent.t_depart + urgent.tof < patient.t_depart + patient.tof


def test_equal_periods_without_deadline_raises_clearly():
    """Equal altitudes -> equal periods -> the phase never closes by drifting,
    so there is no natural window to default the deadline to. This must be a
    clear physics error, not nan-propagation into the Kepler solver."""
    import pytest

    with pytest.raises(ValueError, match="equal periods"):
        build_scenario(chaser_altitude=500.0)

    # With an explicit deadline the planner may spend fuel instead: feasible.
    period = 2 * np.pi / mean_motion(R_EARTH + 500.0, GM_EARTH)
    *_, policy, _, _ = build_scenario(chaser_altitude=500.0, max_mission_time=2 * period)
    assert np.isfinite(policy.plan.delta_v)


def test_full_mission_closes_and_holds_from_far_coplanar_start():
    sat_r, sat_v, debris_r, debris_v, policy, t_final, period = build_scenario()

    result = simulate(sat_r, sat_v, debris_r, debris_v, policy.decide, t_final, GM_EARTH, dt=30.0)

    assert result.separation[0] > 1000.0  # genuinely started far away
    hold_dist = np.linalg.norm(HOLD_OFFSET)
    held = result.separation[result.t > t_final - period]
    assert np.abs(held.mean() - hold_dist) < 0.5
    assert held.max() < hold_dist + 1.0
    # And it should have been cheap, because it drifted to the window.
    assert (policy.depart_dv + policy.arrive_dv) * 1000 < 150.0
