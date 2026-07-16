"""Render the whole-process mission animation to mission.gif.

Kept out of `uv run orbdebris` so the normal run stays fast - rendering a few
hundred frames takes a while.

Two sampling decisions do the real work here:

* **Simulate finely** (fine samples everywhere). The trail is drawn from these,
  so orbits render as smooth arcs. Sampling coarsely would draw ~4000 km straight
  chords between drift samples - a chaser at 7.6 km/s moves a long way in a
  single coarse step.
* **Time-lapse the drift stroboscopically**, one frame per *debris orbital
  period*. Naive time-lapse aliases badly: 39 orbits over a few dozen frames
  makes the chaser strobe randomly around the circle. Sampling at exactly the
  debris' period parks the debris at the same phase every frame, so what you
  actually see is the chaser creeping forward - which *is* the phasing.

Run: uv run python scripts/make_animation.py
"""

import numpy as np

from orbdebris import build_scenario
from orbdebris.animate import animate_mission
from orbdebris.constants import GM_EARTH
from orbdebris.simulate import simulate

ACTION_FRAMES = 200


def main() -> None:
    sat_r, sat_v, debris_r, debris_v, policy, t_final, period = build_scenario()
    plan = policy.plan
    print(
        f"drift {plan.t_depart / 3600:.2f} h, transfer {plan.tof / 3600:.2f} h, "
        f"total {t_final / 3600:.2f} h"
    )

    result = simulate(
        sat_r,
        sat_v,
        debris_r,
        debris_v,
        policy.decide,
        t_final,
        GM_EARTH,
        dt=30.0,
        max_samples_per_segment=100_000,  # keep it fine: the trail is drawn from this
    )
    print(f"{len(result.t)} samples, {len(result.burn_times)} burns")

    drift_times = np.arange(0.0, plan.t_depart, period)  # one frame per orbit
    action_times = np.linspace(plan.t_depart, result.t[-1], ACTION_FRAMES)
    frame_times = np.concatenate([drift_times, action_times])
    print(f"{len(drift_times)} drift frames (stroboscopic) + {ACTION_FRAMES} action frames")

    # Without these labels the frozen debris reads as "the debris stopped",
    # when in fact it is orbiting at ~7.6 km/s and only the camera is strobed.
    phases = [
        (0.0, "DRIFT - free, unpowered.  STROBOSCOPIC: 1 frame per debris orbit, "
              "so the debris looks frozen while the chaser creeps ~8 deg/frame"),
        (plan.t_depart, "TRANSFER - Lambert burn, real time (strobe off: the debris moves again)"),
        (plan.t_depart + plan.tof, "TERMINAL - CW rendezvous, then hold 1 km ahead (real time)"),
    ]

    out = animate_mission(
        result,
        GM_EARTH,
        output_path="mission.gif",
        frame_times=frame_times,
        trail_seconds=0.45 * period,
        fps=18,
        phases=phases,
        dpi=76,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
