"""Plotting for scenario results."""

import matplotlib
import numpy as np

# Figures are written to disk, never shown, so pin the non-interactive backend
# rather than depend on a GUI toolkit. See animate.py for the same reasoning.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from orbdebris.constants import R_EARTH  # noqa: E402
from orbdebris.simulate import ScenarioResult  # noqa: E402


def plot_trajectories(result: ScenarioResult, output_path: str | None = None) -> plt.Figure:
    """Static Hill-frame view: chaser path relative to the debris (at origin),
    plus separation vs time. The Hill frame is the natural rendezvous view -
    in absolute ECI the two orbits overlap and reveal nothing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Hill plane: along-track (y) horizontal, radial (x) vertical - the usual
    # rendezvous convention, with the direction of motion pointing right.
    along = result.rel_r[:, 1]
    radial = result.rel_r[:, 0]
    ax1.plot(along, radial, lw=1, label="chaser path")
    ax1.plot(0, 0, "k*", ms=12, label="debris")
    ax1.plot(along[0], radial[0], "go", label="start")
    ax1.plot(along[-1], radial[-1], "rs", label="end")
    ax1.set_xlabel("along-track [km]  (direction of motion ->)")
    ax1.set_ylabel("radial [km]  (outward ^)")
    ax1.set_title("Relative motion in the Hill frame")
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(result.t, result.separation)
    for bt in result.burn_times:
        ax2.axvline(bt, color="orange", alpha=0.4, lw=0.8)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("separation [km]")
    ax2.set_title("Satellite-debris separation (orange = burns)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    return fig


def plot_mission(
    result: ScenarioResult, output_path: str | None = None, terminal_range: float = 50.0
) -> plt.Figure:
    """Whole-process view: ECI transfer (3D, with Earth) + terminal Hill frame +
    separation vs time on a log axis. terminal_range [km] bounds the Hill panel
    to the close-in portion so the km-scale rendezvous is visible."""
    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_earth(ax1)
    ax1.plot(*result.sat_r.T, lw=1, label="chaser")
    ax1.plot(*result.debris_r.T, lw=1, label="debris")
    ax1.plot(*result.sat_r[0], "go", label="chaser start")
    ax1.set_title("ECI: Lambert transfer")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_box_aspect((1, 1, 1))

    ax2 = fig.add_subplot(1, 3, 2)
    close = result.separation < terminal_range
    ax2.plot(result.rel_r[close, 1], result.rel_r[close, 0], lw=1, label="chaser path")
    ax2.plot(0, 0, "k*", ms=12, label="debris")
    ax2.plot(0, 0)  # keep origin in view
    ax2.set_xlabel("along-track [km]  (motion ->)")
    ax2.set_ylabel("radial [km]  (outward ^)")
    ax2.set_title(f"Hill frame (terminal, <{terminal_range:.0f} km)")
    ax2.axis("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.semilogy(result.t, result.separation)
    for bt in result.burn_times:
        ax3.axvline(bt, color="orange", alpha=0.4, lw=0.8)
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("separation [km]  (log)")
    ax3.set_title("Separation over the whole process")
    ax3.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    return fig


def _draw_earth(ax, radius: float = R_EARTH) -> None:
    u, v = np.mgrid[0 : 2 * np.pi : 24j, 0 : np.pi : 12j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_surface(x, y, z, color="steelblue", alpha=0.2, linewidth=0)
