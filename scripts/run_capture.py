"""Milestone 3 scenario: throw a membrane net at a debris pellet cloud,
close the drawstring, verify capture, and tow the bag away.

The chaser starts 30 m ahead of the cloud centroid - the terminal state the
mission-level rendezvous (M2b) delivers, just closer in, per the CLOSE-IN
staging step. Distances here are metres (capture world).

Run: uv run python scripts/run_capture.py
"""

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from orbdebris.capture import CHASER, demo_scenario  # noqa: E402


def main() -> None:
    # The canonical tuned scenario lives in capture.demo_scenario - shared
    # with the end-to-end test so script and suite cannot drift apart.
    sim, eject_kwargs = demo_scenario()
    cloud = sim.cloud

    info = sim.eject(**eject_kwargs)
    print("eject:")
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")

    t0 = time.perf_counter()
    result = sim.run(duration=110.0, dt=0.001, record_every=100)
    print(f"simulated 110 s in {time.perf_counter() - t0:.1f} s wall time")

    print("events:")
    for name, t_ev in sorted(result.events.items(), key=lambda kv: kv[1]):
        print(f"  {name}: t = {t_ev:.2f} s")
    print(f"captured: {result.captured}")
    print(f"final retained fraction: {result.retained_frac[-1]:.2f}")
    print(f"final mouth radius: {result.mouth_radius[-1]:.2f} m")
    print(f"peak tether tension: {result.tether_tension.max():.1f} N")

    # Tow effectiveness: how far did the cloud centroid move after thrust-on?
    w = cloud.masses / cloud.masses.sum()
    centroids = (w[None, :, None] * result.pos[:, result.pellets]).sum(axis=1)
    if "tow_start" in result.events:
        k0 = int(np.searchsorted(result.t, result.events["tow_start"]))
        tow_disp = np.linalg.norm(centroids[-1] - centroids[k0])
        print(f"cloud centroid displacement under tow: {tow_disp:.1f} m")

    plot(result, centroids)
    print("saved plot to capture_summary.png")


def plot(result, centroids) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Final geometry, two Hill projections.
    for ax, (i, j), label in (
        (axes[0, 0], (1, 0), ("along-track y [m]", "radial x [m]")),
        (axes[0, 1], (1, 2), ("along-track y [m]", "cross-track z [m]")),
    ):
        pos = result.pos[-1]
        nodes = pos[result.nodes]
        for a, b in result.net.links:
            ax.plot(
                [nodes[a, i], nodes[b, i]], [nodes[a, j], nodes[b, j]],
                "-", color="tab:blue", lw=0.5, alpha=0.6,
            )
        pel = pos[result.pellets]
        ax.scatter(pel[:, i], pel[:, j], s=8, color="tab:orange", label="pellets", zorder=3)
        ax.plot(pos[CHASER, i], pos[CHASER, j], "s", color="tab:green", ms=8, label="chaser")
        ax.plot(centroids[-1, i], centroids[-1, j], "k+", ms=10, label="cloud centroid")
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_title("Final geometry")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(result.t, result.mouth_radius, label="mouth radius [m]")
    ax.plot(result.t, result.retained_frac, label="retained fraction")
    for name, t_ev in result.events.items():
        ax.axvline(t_ev, color="gray", alpha=0.4, lw=0.8)
        ax.text(t_ev, ax.get_ylim()[1] * 0.95, name, rotation=90, fontsize=6, va="top")
    ax.set_xlabel("time [s]")
    ax.set_title("Capture progress")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(result.t, result.tether_tension, label="tether tension [N]", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(result.t, result.contact_count, label="pellets in contact", color="tab:purple",
             alpha=0.6)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("tension [N]", color="tab:red")
    ax2.set_ylabel("contacts", color="tab:purple")
    ax.set_title("Tension and contact")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("capture_summary.png", dpi=130)


if __name__ == "__main__":
    main()
