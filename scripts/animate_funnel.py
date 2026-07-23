"""Render the four-satellite funnel constellation capturing a debris cloud:
deploy the membrane funnel -> hold formation -> sweep through the cloud ->
debris channels down the walls into the apex storage box -> secured. Two Hill
side-views, animated together. The mouth stays open throughout: the satellites
never let go.

The 47-minute half-orbit approach coast is analytic (pure CW flow); the burn
that starts the sweep injects its arrival velocity. See
constellation.run_full_mission.

Run: uv run python scripts/animate_funnel.py
"""

import time

from orbdebris.animate import animate_funnel
from orbdebris.constellation import run_full_mission


def main() -> None:
    t0 = time.perf_counter()
    sim, result, approach = run_full_mission()
    n_pellets = len(result.pellet_radii)
    print(f"simulated {result.t[-1]:.0f} s in {time.perf_counter() - t0:.0f} s wall time; "
          f"{len(result.t)} frames")
    print("events:", {k: round(v, 1) for k, v in sorted(result.events.items(), key=lambda kv: kv[1])})
    print(f"secured: {result.captured}, stored in apex box: "
          f"{round(result.stored_frac[-1] * n_pellets)}/{n_pellets}, "
          f"mouth {result.mouth_radius[-1]:.1f} m (open throughout), "
          f"formation dv: {result.formation_dv[-1]:.2f} m/s")

    out = animate_funnel(result, output_path="funnel.gif", n_frames=220, fps=20)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
