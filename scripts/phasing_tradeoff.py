"""Phasing delta-v vs mission-time tradeoff study.

Question: to bring the chaser (lower, faster orbit) into CW range of the
debris, how does the required phasing delta-v trade against how long we let the
maneuver take?

Method: the chaser drifts freely in its initial orbit (free, no fuel) for a
wait time t_dep, then executes a single Lambert transfer of duration tof to a
standoff point near the debris. We search over (t_dep, tof) and, for each total
mission time t_dep + tof, keep the minimum total transfer delta-v. The lower
envelope is the fuel/time Pareto front.

Run: uv run python scripts/phasing_tradeoff.py
"""

import matplotlib.pyplot as plt
import numpy as np

from orbdebris.constants import GM_EARTH, R_EARTH
from orbdebris.kepler import kepler_propagate
from orbdebris.lambert import lambert
from orbdebris.relative import hill_state_to_eci, mean_motion
from orbdebris.state import elements_to_state

MU = GM_EARTH
DEBRIS_ALT = 500.0
CHASER_ALT = 400.0
HANDOFF = np.array([0.0, -20.0, 0.0])  # 20 km behind, Hill frame


def hohmann_floor(r1: float, r2: float) -> float:
    """Analytic two-burn Hohmann delta-v between coplanar circular orbits [km/s]."""
    at = 0.5 * (r1 + r2)
    v1c, v2c = np.sqrt(MU / r1), np.sqrt(MU / r2)
    vp = np.sqrt(MU * (2 / r1 - 1 / at))
    va = np.sqrt(MU * (2 / r2 - 1 / at))
    return abs(vp - v1c) + abs(v2c - va)




def main() -> None:
    debris_a = R_EARTH + DEBRIS_ALT
    chaser_a = R_EARTH + CHASER_ALT
    period_d = 2 * np.pi / mean_motion(debris_a, MU)

    # Synodic period: how long for the phase to lap once.
    dn = mean_motion(chaser_a, MU) - mean_motion(debris_a, MU)
    t_syn = 2 * np.pi / abs(dn)

    inc = np.radians(51.6)
    debris_r0, debris_v0 = elements_to_state(
        a=debris_a, e=0.0, i=inc, raan=0.0, argp=0.0, nu=0.0, mu=MU
    )
    chaser_r0, chaser_v0 = elements_to_state(
        a=chaser_a, e=0.0, i=inc, raan=0.0, argp=0.0, nu=np.radians(45.0), mu=MU
    )

    floor = hohmann_floor(chaser_a, debris_a)

    t_deps = np.linspace(0.0, 1.05 * t_syn, 70)
    tofs = np.linspace(0.30, 0.92, 32) * period_d

    times, dvs = [], []
    for t_dep in t_deps:
        cr, cv = kepler_propagate(chaser_r0, chaser_v0, t_dep, MU)
        for tof in tofs:
            dr, dv_deb = kepler_propagate(debris_r0, debris_v0, t_dep + tof, MU)
            target_r, desired_v = hill_state_to_eci(HANDOFF, np.zeros(3), dr, dv_deb, MU)
            try:
                v1, v2 = lambert(cr, target_r, tof, MU)
            except ValueError:
                continue
            dv = np.linalg.norm(v1 - cv) + np.linalg.norm(v2 - desired_v)
            times.append((t_dep + tof) / 3600.0)  # hours
            dvs.append(dv * 1000.0)  # m/s

    times = np.array(times)
    dvs = np.array(dvs)

    # Lower envelope (Pareto): min dv within each time bin.
    bins = np.linspace(times.min(), times.max(), 60)
    idx = np.digitize(times, bins)
    env_t, env_dv = [], []
    for b in range(1, len(bins)):
        sel = idx == b
        if sel.any():
            env_t.append(times[sel].mean())
            env_dv.append(dvs[sel].min())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(times, dvs, s=6, alpha=0.2, color="gray", label="feasible (t_dep, tof)")
    ax.plot(env_t, env_dv, "b-o", ms=3, label="min-fuel Pareto front")
    ax.axhline(floor * 1000, color="green", ls="--", label=f"Hohmann floor ({floor*1000:.0f} m/s)")
    ax.axvline(t_syn / 3600, color="orange", ls=":", label=f"1 synodic period ({t_syn/3600:.1f} h)")
    ax.set_yscale("log")
    ax.set_xlabel("total mission time [hours]")
    ax.set_ylabel("phasing delta-v [m/s]  (log)")
    ax.set_title("Phasing delta-v vs mission time (chaser 400 km -> debris 500 km, 45 deg)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("phasing_tradeoff.png", dpi=150)

    print(f"Hohmann floor:            {floor*1000:.1f} m/s")
    print(f"Synodic period:           {t_syn/3600:.2f} h  ({t_syn/period_d:.1f} orbits)")
    print(f"Cheapest found overall:   {dvs.min():.1f} m/s at {times[dvs.argmin()]:.2f} h")
    fast = times < 1.0
    if fast.any():
        print(f"Cheapest within 1 hour:   {dvs[fast].min():.1f} m/s")
    print("saved plot to phasing_tradeoff.png")


if __name__ == "__main__":
    main()
