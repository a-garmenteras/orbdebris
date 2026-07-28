"""Milestone 5, stage 1: the whole mission's fuel, in one table.

Runs the reference mission end to end - M2b (drift to the cheap phasing window,
Lambert transfer, CW terminal rendezvous) followed by M4 (split into four, unfurl
the funnel, sweep the cloud, hold the mouth open) - with a single FuelLedger
threaded through every guidance module. Then converts delta-v into kilograms
through the rocket equation.

Three things this answers that nothing before it could:

  1. What does the terminal CW rendezvous actually cost? `simulate` used to
     record only burn *times*, so this number did not exist.
  2. Which satellite is thirstiest? The four do not share a tank, so the
     constellation is limited by its worst-off member, not by the average.
  3. At what point does *loitering* cost more than *getting there*? Holding a
     rigid formation is continuous thrust, so its cost grows with the clock
     while phasing is paid once. Where those cross decides whether you wait on
     station or go home.

Run: uv run python scripts/fuel_budget.py
"""

import numpy as np

from orbdebris import HOLD_OFFSET, build_scenario
from orbdebris.constants import GM_EARTH
from orbdebris.constellation import N_SATS, run_full_mission
from orbdebris.propulsion import (
    FORMATION,
    PHASING,
    STATIONKEEP,
    SWEEP,
    TERMINAL,
    FuelLedger,
    Vehicle,
    split_vehicle,
)
from orbdebris.simulate import simulate

# A 500 kg chaser: 400 kg dry, 100 kg of hydrazine. Isp 220 s is a monopropellant
# hydrazine thruster - the conventional choice for a small LEO servicing vehicle,
# and deliberately the *pessimistic* end (a bipropellant system would give ~320 s,
# a Hall thruster ~1600 s but with thrust too low for the impulsive burns this
# architecture assumes).
DRY_MASS = 400.0
PROP_MASS = 100.0
ISP = 220.0
SAT_DRY_MASS = 100.0  # each of the four, post-split


def steady_hold_rate(seconds: float = 150.0, dt: float = 0.0015) -> float:
    """Measure the *quiescent* formation-keeping rate, in m/s/day summed over
    the four satellites.

    A dedicated run: split, deploy, trim, hold - no sweep burn, no debris
    contact. The rate is fitted over the second half only, so the deployment
    transient (a one-off cost, not a rate) is excluded.

    ``trim_time`` is not optional. Without it the controller holds slots at the
    as-built blueprint radius while the membrane has settled slightly inside it,
    so it leans on the structure forever and the measured rate comes out ~8.5x
    too high (199 m/s/day untrimmed vs 23 trimmed). That is the same tug-of-war
    `trim_slots` exists to end - and a steady hold is the cleanest place to see
    it, because a short transient run hides a constant-rate error inside its
    transient.
    """
    from orbdebris.constellation import funnel_demo

    sim, _, _ = funnel_demo()
    sim.enable_contact = False
    sim.split()
    sim.hold_position()
    res = sim.run(seconds, dt=dt, record_every=200, trim_time=0.4 * seconds)

    half = len(res.t) // 2
    t, dv = res.t[half:], res.formation_dv[half:]
    slope = float(np.polyfit(t, dv, 1)[0])  # m/s per second
    return slope * 86400.0


def thrust_audit() -> None:
    """The 'power' half of the milestone, kept deliberately thin: check that the
    acceleration limits we invented correspond to thrusters that exist."""
    print("\nthrust audit (are our invented acceleration caps physical?)")
    sat_mass = 125.0  # FunnelSim.sat_mass
    for label, accel in (("formation-keeping (max_accel)", 0.05),
                         ("apex tensioning", 0.06)):
        print(f"  {label:<32} {accel:.3f} m/s^2 x {sat_mass:.0f} kg = "
              f"{accel * sat_mass:6.2f} N")
    print("  reference: a 22 N monoprop thruster (Aerojet MR-106) is standard on")
    print("  small spacecraft; 1 N and 10 N classes are common for RCS. Both of")
    print("  our numbers sit inside that range, so the caps are physical - but")
    print("  they are near the top of it, i.e. we assumed a generously actuated")
    print("  satellite. Nothing here needs changing; it needed checking.")
    print("  Electric propulsion (Isp ~1600 s) is deferred: its thrust is ~0.1 N,")
    print("  which cannot produce an impulsive burn, and every guidance module")
    print("  in this project assumes impulses.")


def main() -> None:
    ledger = FuelLedger()
    chaser = ledger.add_vehicle(
        Vehicle(name="chaser", dry_mass=DRY_MASS, prop_mass=PROP_MASS, isp=ISP)
    )

    # ---------------------------------------------------------------- M2b
    sat_r, sat_v, debris_r, debris_v, policy, t_final, period = build_scenario(ledger=ledger)
    result = simulate(
        sat_r, sat_v, debris_r, debris_v, policy.decide, t_final, GM_EARTH, dt=30.0
    )

    print("orbdebris milestone 5 - one fuel ledger for the whole mission\n")
    print(f"M2b: drifted {policy.plan.t_depart / 3600:.1f} h to the cheap window, "
          f"transferred {policy.plan.tof / 3600:.2f} h,")
    print(f"     then held {HOLD_OFFSET[1]:.0f} km ahead for "
          f"{(t_final - policy.plan.t_depart - policy.plan.tof) / 3600:.1f} h "
          f"({len(result.burn_times)} burns total).")

    # Cross-check: the engine's unlabelled record of what was flown must match
    # the sum of what the policies claim they spent. These are computed by
    # different code paths, so agreement is a real check, not a tautology.
    flown = float(result.burn_dv.sum()) * 1000.0
    claimed = ledger.total_dv(vehicle="chaser")
    print(f"     cross-check: simulate flew {flown:.3f} m/s, ledger recorded "
          f"{claimed:.3f} m/s  ({'agree' if abs(flown - claimed) < 1e-6 else 'MISMATCH'})")

    # ------------------------------------------------------- split into four
    remaining = ledger.remaining_mass("chaser")
    names = [f"sat{i}" for i in range(N_SATS)]
    for v in split_vehicle(chaser, remaining, names, dry_mass_each=SAT_DRY_MASS):
        ledger.add_vehicle(v)
    print(f"\nsplit: {remaining:.1f} kg remaining -> 4 x {remaining / N_SATS:.1f} kg "
          f"({SAT_DRY_MASS:.0f} kg dry + {remaining / N_SATS - SAT_DRY_MASS:.1f} kg propellant each)")

    # ----------------------------------------------------------------- M4
    _, funnel, _ = run_full_mission(ledger=ledger, vehicle_names=names)
    stored = funnel.stored_frac[-1]
    print(f"M4:  swept the cloud, {round(stored * len(funnel.stored_mask[-1]))}/"
          f"{len(funnel.stored_mask[-1])} pellets stored, "
          f"secured={funnel.captured}")

    # ------------------------------------------------------------- the table
    print("\n" + ledger.table())

    # -------------------------------------------------------- what dominates
    print("\nwhat dominates")
    total = ledger.total_dv()
    for phase in sorted(ledger.phases(), key=lambda p: -ledger.total_dv(phase=p)):
        dv = ledger.total_dv(phase=phase)
        print(f"  {phase:<12}{dv:9.3f} m/s  ({100 * dv / total:5.1f}% of mission dv)")

    thirsty = max(names, key=lambda n: ledger.propellant_by(vehicle=n))
    print(f"\n  thirstiest satellite: {thirsty} at "
          f"{ledger.propellant_by(vehicle=thirsty):.3f} kg - the constellation is")
    print("  limited by its worst-off member, not by the four-satellite average.")

    # ------------------------------------------------- the loiter crossover
    # Formation-keeping is continuous thrust, so its cost grows with the clock,
    # while getting there is paid once. Where they cross decides loiter-vs-go-home.
    #
    # Careful: the *capture* run is a terrible place to measure a hold rate. Its
    # 200 s contain the deploy transient, the slot re-trim, the sweep burn and
    # the debris impacts - all one-off costs. Extrapolating them to a day
    # over-reports the rate by ~150x. So measure a quiescent hold on its own.
    per_day = steady_hold_rate()
    transient = ledger.total_dv(phase=FORMATION)
    getting_there = (
        ledger.total_dv(phase=PHASING)
        + ledger.total_dv(phase=TERMINAL)
        + ledger.total_dv(phase=SWEEP)
    )
    print("\nthe loiter crossover")
    print(f"  one capture (transient):  {transient:.3f} m/s over {funnel.t[-1]:.0f} s "
          f"- one-off, do NOT extrapolate")
    print(f"  steady hold:              {per_day:.2f} m/s/day (4 sats, quiescent)")
    print(f"  getting there:            {getting_there:.3f} m/s, paid once")
    if per_day > 0:
        print(f"  => holding the funnel open outgrows the entire journey after "
              f"{getting_there / per_day:.1f} days on station.")
        print("  That is what decides loiter-vs-go-home: a constellation parked in")
        print("  formation is spending fuel at a rate a single satellite never does,")
        print("  because a rigid formation is not a natural CW motion.")
    sk = ledger.total_dv(phase=STATIONKEEP)
    sk_hours = (t_final - policy.plan.t_depart - policy.plan.tof) / 3600.0
    print(f"\n  for contrast, the *single* chaser station-keeping at its 1 km hold "
          f"spent {sk:.4f} m/s")
    print(f"  in {sk_hours:.1f} h = {sk / sk_hours * 24:.3f} m/s/day - "
          f"{per_day / max(sk / sk_hours * 24, 1e-12):,.0f}x cheaper than holding")
    print("  the formation. That ratio is the fuel answer to 'why not just fly a net?'")

    print(f"\nfeasible within the {PROP_MASS:.0f} kg tank: {ledger.feasible()}")
    print(f"chaser dv capacity was {chaser.dv_capacity():.1f} m/s; "
          f"the mission used {ledger.total_dv(vehicle='chaser'):.1f} m/s of it.")

    thrust_audit()


if __name__ == "__main__":
    main()
