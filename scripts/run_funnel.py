"""Milestone 4 scenario: the four-satellite funnel closes on a debris cloud
from behind, sweeps through it, and channels the debris down the membrane
walls into the storage box at the apex satellite. The mouth stays open the
whole time - the satellites never let go.

The approach is reported analytically (the half-orbit coast is pure CW flow -
no reason to grind it at millisecond timesteps), and the terminal sweep is run
in the fine contact sim. See constellation.funnel_demo.

Run: uv run python scripts/run_funnel.py
"""

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from orbdebris.constellation import APEX_SAT, MOUTH_SATS, N_SATS, funnel_demo  # noqa: E402


def main() -> None:
    sim, sweep_velocity, approach = funnel_demo()
    n_pellets = sim.cloud.n_pellets

    burn = approach["burn"]
    print("approach (analytic, from 1 km behind):")
    print(f"  burn:    radial {burn[0]:+.3f}, along {burn[1]:+.3f}, cross {burn[2]:+.3f} m/s"
          "   <- purely radial DOWN")
    print(f"  coast:   {approach['tof'] / 60:.1f} min (half an orbit, CW flow, not fine-stepped)")
    print(f"  arrival: radial {sweep_velocity[0]:+.3f}, along {sweep_velocity[1]:+.3f}, "
          f"cross {sweep_velocity[2]:+.3f} m/s   <- sweeps radially UP through the cloud")

    sim.deploy_open(sweep_velocity)

    t0 = time.perf_counter()
    result = sim.run(duration=140.0, dt=0.0015, record_every=100)
    print(f"\nsimulated 140 s sweep in {time.perf_counter() - t0:.0f} s wall time")

    print("events:")
    for name, t_ev in sorted(result.events.items(), key=lambda kv: kv[1]):
        print(f"  {name}: t = {t_ev:.2f} s")
    print(f"captured (secured): {result.captured}")
    print(f"stored in apex box: {result.stored_frac[-1]:.2f}  "
          f"({round(result.stored_frac[-1] * n_pellets)}/{n_pellets} pellets)")
    print(f"inside funnel:      {result.inside_funnel_frac[-1]:.2f}")
    print(f"final mouth radius: {result.mouth_radius[-1]:.2f} m "
          f"(built {sim.mouth_radius:.1f} - open throughout, satellites never let go)")
    print(f"formation-keeping dv (sweep): {result.formation_dv[-1]:.3f} m/s")

    plot(result)
    print("saved plot to funnel_summary.png")


def _draw_net(ax, pos, net, i, j, color):
    nodes = pos[N_SATS : N_SATS + net.n_nodes]
    for a, b in net.links:
        ax.plot([nodes[a, i], nodes[b, i]], [nodes[a, j], nodes[b, j]],
                "-", color=color, lw=0.4, alpha=0.5)


def plot(result) -> None:
    net = result.net
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Two Hill projections of the final geometry.
    for ax, (i, j), labels in (
        (axes[0, 0], (0, 1), ("radial x [m]", "along-track y [m]")),
        (axes[0, 1], (0, 2), ("radial x [m]", "cross-track z [m]")),
    ):
        pos = result.pos[-1]
        _draw_net(ax, pos, net, i, j, "tab:blue")
        pel = pos[result.pellets]
        stored = result.stored_mask[-1]
        ax.scatter(pel[~stored, i], pel[~stored, j], s=10, color="tab:orange", zorder=3,
                   label="pellets (loose)")
        ax.scatter(pel[stored, i], pel[stored, j], s=14, color="tab:red", zorder=3,
                   label="pellets (stored)")
        # The storage box: a soft one-way container mounted at the apex node.
        apex = pos[N_SATS + net.apex_node]
        ax.add_patch(plt.Circle((apex[i], apex[j]), 3.5, fill=False, color="tab:red",
                                ls="--", lw=1.0, alpha=0.6))
        sats = pos[result.sats]
        ax.scatter(sats[MOUTH_SATS, i], sats[MOUTH_SATS, j], s=60, marker="s",
                   color="tab:green", zorder=4, label="mouth sats")
        ax.scatter(sats[APEX_SAT, i], sats[APEX_SAT, j], s=60, marker="D",
                   color="tab:red", zorder=4, label="apex sat")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title("Final geometry")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(result.t, result.stored_frac, label="stored in apex box", color="tab:orange")
    ax.plot(result.t, result.inside_funnel_frac, label="inside funnel", color="tab:green",
            alpha=0.7)
    ax.plot(result.t, result.mouth_radius / 4.0, label="mouth radius / 4 m", color="tab:blue")
    for name, t_ev in result.events.items():
        ax.axvline(t_ev, color="gray", alpha=0.4, lw=0.8)
        ax.text(t_ev, 1.02, name, rotation=90, fontsize=6, va="bottom")
    ax.set_xlabel("time [s]")
    ax.set_title("Capture progress")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(result.t, result.formation_dv, color="tab:purple")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cumulative formation-keeping dv [m/s]")
    ax.set_title("Formation-keeping cost")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("funnel_summary.png", dpi=130)


if __name__ == "__main__":
    main()
