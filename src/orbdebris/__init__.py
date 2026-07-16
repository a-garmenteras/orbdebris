"""Orbital debris-collection simulator.

The reference mission (build_scenario/main): a chaser on an arbitrary coplanar
orbit drifts for free until the cheap phasing window opens, runs a Lambert
transfer into CW range, then hands off to the terminal CW rendezvous and holds
a 1 km lead ahead of the debris.

The mission deadline (max_mission_time) is the fuel/time knob: generous ->
near-Hohmann-floor fuel, slow; tight -> fast, expensive. See ROADMAP.md for
milestones and the architecture decisions log.
"""

import numpy as np

from orbdebris.constants import GM_EARTH, R_EARTH
from orbdebris.mission import MissionPolicy
from orbdebris.phasing import plan_min_fuel_transfer, synodic_period
from orbdebris.relative import mean_motion
from orbdebris.rendezvous import RendezvousPolicy
from orbdebris.simulate import simulate
from orbdebris.state import elements_to_state
from orbdebris.visualize import plot_mission

HANDOFF_OFFSET = np.array([0.0, -20.0, 0.0])  # arrive 20 km behind the debris
HOLD_OFFSET = np.array([0.0, 1.0, 0.0])  # then hold 1 km ahead of it


def build_scenario(
    debris_altitude: float = 500.0,
    chaser_altitude: float = 400.0,
    chaser_phase_offset: float = np.radians(45.0),
    max_mission_time: float | None = None,
):
    """Debris on a circular LEO orbit; chaser on a different coplanar orbit.

    max_mission_time [s]: the deadline the planner must fit inside. Defaults to
    ~1 synodic period, which is enough to reach the fuel-optimal window (the
    geometry repeats after that, so waiting longer buys nothing).
    """
    debris_a = R_EARTH + debris_altitude
    chaser_a = R_EARTH + chaser_altitude
    inclination = np.radians(51.6)
    period = 2 * np.pi / mean_motion(debris_a, GM_EARTH)

    debris_r, debris_v = elements_to_state(
        a=debris_a, e=0.0, i=inclination, raan=0.0, argp=0.0, nu=0.0, mu=GM_EARTH
    )
    sat_r, sat_v = elements_to_state(
        a=chaser_a,
        e=0.0,
        i=inclination,
        raan=0.0,
        argp=0.0,
        nu=chaser_phase_offset,
        mu=GM_EARTH,
    )

    if max_mission_time is None:
        t_syn = synodic_period(sat_r, sat_v, debris_r, debris_v, GM_EARTH)
        if not np.isfinite(t_syn):
            raise ValueError(
                "Chaser and debris have equal periods, so drifting never changes "
                "their phase and there is no natural (near-free) transfer window "
                "to wait for. Pass an explicit max_mission_time to let the "
                "planner spend fuel on the phase change instead."
            )
        max_mission_time = 1.05 * t_syn

    plan = plan_min_fuel_transfer(
        sat_r, sat_v, debris_r, debris_v, HANDOFF_OFFSET, GM_EARTH, max_mission_time
    )

    terminal = RendezvousPolicy(
        mu=GM_EARTH,
        hold_offset=HOLD_OFFSET,
        transfer_time=0.4 * period,
        hold_check_interval=period / 20,
        # Per-axis [radial, along-track, cross-track] km. Radial is the drift
        # driver, so keep it tight; along-track costs nothing, so let it roam
        # rather than paying to buy it back.
        deadband=np.array([0.02, 0.5, 0.05]),
    )
    policy = MissionPolicy(
        mu=GM_EARTH,
        plan=plan,
        handoff_offset=HANDOFF_OFFSET,
        terminal_policy=terminal,
        arrival_coast=period / 20,
    )
    t_final = plan.t_depart + plan.tof + period / 20 + 0.4 * period + 2.5 * period
    return sat_r, sat_v, debris_r, debris_v, policy, t_final, period


def main() -> None:
    sat_r, sat_v, debris_r, debris_v, policy, t_final, period = build_scenario()
    plan = policy.plan

    result = simulate(sat_r, sat_v, debris_r, debris_v, policy.decide, t_final, GM_EARTH, dt=30.0)

    held = result.separation[result.t > t_final - period]
    print("orbdebris milestone 2b - drift to the cheap window, then Lambert + CW")
    print("  plan:")
    print(f"    drift before departing: {plan.t_depart / 3600:.2f} h "
          f"({plan.t_depart / period:.1f} orbits)")
    print(f"    transfer duration:      {plan.tof / 3600:.2f} h")
    print(f"    total mission time:     {t_final / 3600:.2f} h")
    print("  result:")
    print(f"    initial separation:  {result.separation[0]:.1f} km")
    print(f"    final separation:    {result.separation[-1]:.3f} km")
    print(f"    held (last orbit):   mean {held.mean():.3f} km, max {held.max():.3f} km")
    print(f"    phasing dv (depart): {policy.depart_dv * 1000:.1f} m/s")
    print(f"    phasing dv (arrive): {policy.arrive_dv * 1000:.1f} m/s")
    print(f"    phasing dv (total):  {(policy.depart_dv + policy.arrive_dv) * 1000:.1f} m/s")
    print(f"    total burns:         {len(result.burn_times)}")

    plot_mission(result, output_path="milestone2b.png")
    print("  saved plot to milestone2b.png")
