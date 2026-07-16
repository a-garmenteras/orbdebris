"""Animation of a whole mission: drift -> transfer -> terminal rendezvous.

Three panels, animating together:

* **ECI (global)**, projected onto the common orbital plane. Our scenarios are
  coplanar, so both orbits live in one plane and a 2D projection reads far
  better than a 3D wireframe: clean circles, the transfer arc, and Earth. Fixed
  limits - the "where am I around Earth" view.
* **ECI (tracking)**, the same frame and conventions, but the camera follows the
  midpoint of the two bodies and zooms with their separation. Note this panel
  *must* degenerate at close range into two dots on near-parallel lines: both
  bodies share ~7.6 km/s of common orbital motion, which swamps their relative
  geometry. That is exactly the motion the Hill frame subtracts, so this panel
  shows, by contrast, why the Hill frame is needed at all.
* **Hill frame**, debris at the origin, auto-zooming as the chaser closes.

The dotted "predicted" path is a ballistic Kepler forecast - where a body would
go if it never burned again - recomputed from each body's *current* state every
frame, so the curve always emanates from its marker.

Its sample times are deliberately non-uniform (see ``forecast_times``): packed
close near t=0 and spreading out later. Uniform sampling is a trap here. The
panels span ~4 orders of magnitude of zoom, and matplotlib draws straight chords
between samples: at ~7.6 km/s, a uniform 1.6 h forecast puts samples ~365 km
apart, whose chords cut up to ~2.4 km inside the true arc. That is invisible in
the global panel and catastrophic in a panel zoomed to ~1 km, where it detaches
the dotted line from the marker entirely.

Two sampling rates matter and must not be confused:

* the **trail** is drawn from the simulation's own fine samples over a time
  window, so paths render as smooth arcs;
* the **frames** are chosen independently (see ``frame_times``), which is what
  lets 62 h of drift be time-lapsed without the trail degenerating into straight
  chords across the Earth.
"""

import matplotlib
import numpy as np

# We only ever write figures to disk, never open a window, so pin the
# non-interactive backend: rendering must not depend on a working GUI toolkit.
# (Left to its own devices matplotlib picks TkAgg here, and this machine's Tk
# install is incomplete - it fails or not depending on import order.)
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from orbdebris.constants import R_EARTH  # noqa: E402
from orbdebris.kepler import kepler_propagate
from orbdebris.relative import relative_state
from orbdebris.simulate import ScenarioResult


def orbital_plane_basis(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal in-plane basis (e1, e2) for the orbit of state (r, v).

    e1 points along r; e2 completes a right-handed set within the orbit plane,
    so projecting onto (e1, e2) flattens a coplanar scenario into 2D losslessly.
    """
    e1 = r / np.linalg.norm(r)
    h = np.cross(r, v)
    e3 = h / np.linalg.norm(h)
    e2 = np.cross(e3, e1)
    return e1, e2


def project(points: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Project [N,3] ECI points onto the 2D orbital-plane basis -> [N,2]."""
    pts = np.atleast_2d(points)
    return np.column_stack([pts @ e1, pts @ e2])


def forecast_times(horizon: float, n_points: int = 140, bias: float = 3.0) -> np.ndarray:
    """Forecast sample times over [0, horizon], packed near t=0.

    bias=1 is uniform; higher values crowd samples toward the present. This is
    what keeps the dotted line glued to its marker when a panel is zoomed to
    kilometres, while still tracing a smooth conic when zoomed out to Earth.
    """
    return horizon * np.linspace(0.0, 1.0, n_points) ** bias


def _states_along(r: np.ndarray, v: np.ndarray, times: np.ndarray, mu: float) -> list:
    return [kepler_propagate(r, v, t, mu) for t in times]


def ballistic_forecast(r: np.ndarray, v: np.ndarray, times: np.ndarray, mu: float) -> np.ndarray:
    """Sampled 'if I never burn again' Kepler forecast -> [len(times), 3] ECI.
    times[0] == 0 makes the curve start exactly at the body's current position.
    """
    return np.array([s[0] for s in _states_along(r, v, times, mu)])


def hill_forecast(
    sat_r: np.ndarray,
    sat_v: np.ndarray,
    deb_r: np.ndarray,
    deb_v: np.ndarray,
    times: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Ballistic forecast of the chaser *relative to the debris*, in Hill
    coordinates -> [len(times), 3]. Both bodies are propagated forward, because
    the Hill frame itself rides (and rotates with) the debris."""
    sat = _states_along(sat_r, sat_v, times, mu)
    deb = _states_along(deb_r, deb_v, times, mu)
    return np.array([relative_state(s[0], s[1], d[0], d[1], mu)[0] for s, d in zip(sat, deb)])


def _smoothed_extents(
    frames: np.ndarray, wanted: np.ndarray, smoothing: float, floor: float
) -> list[float]:
    """Exponentially smooth the camera half-extent so zooms glide, not snap."""
    out: list[float] = []
    span = 0.0
    for k in frames:
        want = max(wanted[k], floor)
        span = want if not out else span + smoothing * (want - span)
        out.append(span)
    return out


def animate_mission(
    result: ScenarioResult,
    mu: float,
    output_path: str = "mission.gif",
    n_frames: int = 300,
    frame_times: np.ndarray | None = None,
    trail_seconds: float = 2800.0,
    fps: int = 20,
    forecast_horizon: float | None = None,
    zoom_smoothing: float = 0.12,
    phases: list[tuple[float, str]] | None = None,
    dpi: int = 90,
) -> str:
    """Render the three-panel mission animation to a GIF. Returns output_path.

    frame_times: simulation times to render, letting the caller time-lapse quiet
    stretches (see scripts/make_animation.py). Defaults to uniform over samples.
    trail_seconds: how much history the vanishing trail shows. Always drawn from
    the fine simulation samples, independent of frame spacing.
    phases: (start_time, label) pairs annotating what the camera is doing - e.g.
    that the drift is stroboscopic rather than real time. Without this, a viewer
    reasonably assumes the frozen debris means the debris has stopped.
    """
    if frame_times is None:
        frames = np.unique(np.linspace(0, len(result.t) - 1, n_frames).astype(int))
    else:
        frames = np.clip(np.searchsorted(result.t, frame_times), 0, len(result.t) - 1)

    if forecast_horizon is None:
        forecast_horizon = 1.6 * 3600.0
    f_times = forecast_times(forecast_horizon)

    e1, e2 = orbital_plane_basis(result.debris_r[0], result.debris_v[0])
    sat_xy = project(result.sat_r, e1, e2)
    deb_xy = project(result.debris_r, e1, e2)
    mid_xy = 0.5 * (sat_xy + deb_xy)
    sep_xy = np.linalg.norm(sat_xy - deb_xy, axis=1)

    fig, (ax_glob, ax_track, ax_hill) = plt.subplots(1, 3, figsize=(15.5, 5.6))

    def style(ax, title, xlabel, ylabel):
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    # --- Panel 1: ECI, fixed limits - the global view ---------------------
    lim = 1.15 * max(np.abs(sat_xy).max(), np.abs(deb_xy).max())
    ax_glob.add_patch(plt.Circle((0, 0), R_EARTH, color="steelblue", alpha=0.25))
    ax_glob.set_xlim(-lim, lim)
    ax_glob.set_ylim(-lim, lim)
    style(ax_glob, "ECI (orbital plane) - global", "in-plane x [km]", "in-plane y [km]")

    (g_sat_pred,) = ax_glob.plot([], [], ":", color="tab:blue", lw=1, label="chaser predicted")
    (g_deb_pred,) = ax_glob.plot([], [], ":", color="tab:orange", lw=1, label="debris predicted")
    (g_sat_trail,) = ax_glob.plot([], [], "-", color="tab:blue", lw=1.2, label="chaser travelled")
    (g_deb_trail,) = ax_glob.plot([], [], "-", color="tab:orange", lw=1.2, label="debris travelled")
    (g_sat,) = ax_glob.plot([], [], "o", color="tab:blue", ms=7, label="chaser")
    (g_deb,) = ax_glob.plot([], [], "*", color="tab:orange", ms=13, label="debris")
    ax_glob.legend(loc="upper right", fontsize=6)

    # --- Panel 2: ECI, camera tracks the pair -----------------------------
    ax_track.add_patch(plt.Circle((0, 0), R_EARTH, color="steelblue", alpha=0.25))
    style(ax_track, "ECI (orbital plane) - tracking", "in-plane x [km]", "in-plane y [km]")

    (t_sat_pred,) = ax_track.plot([], [], ":", color="tab:blue", lw=1)
    (t_deb_pred,) = ax_track.plot([], [], ":", color="tab:orange", lw=1)
    (t_sat_trail,) = ax_track.plot([], [], "-", color="tab:blue", lw=1.2)
    (t_deb_trail,) = ax_track.plot([], [], "-", color="tab:orange", lw=1.2)
    (t_sat,) = ax_track.plot([], [], "o", color="tab:blue", ms=7)
    (t_deb,) = ax_track.plot([], [], "*", color="tab:orange", ms=13)

    # --- Panel 3: Hill frame, auto-zooming - the local view ---------------
    style(ax_hill, "Hill frame (debris at origin)", "along-track [km]  (motion ->)",
          "radial [km]  (outward ^)")

    (h_pred,) = ax_hill.plot([], [], ":", color="tab:blue", lw=1)
    (h_trail,) = ax_hill.plot([], [], "-", color="tab:blue", lw=1.2)
    (h_sat,) = ax_hill.plot([], [], "o", color="tab:blue", ms=7)
    (h_deb,) = ax_hill.plot([], [], "*", color="tab:orange", ms=13)

    readout = fig.text(0.5, 0.965, "", ha="center", fontsize=10, family="monospace")
    phase_text = fig.text(0.5, 0.925, "", ha="center", fontsize=9, color="dimgray")
    burn_flash = fig.text(0.11, 0.925, "", ha="center", fontsize=10, color="crimson")

    hill_extents = _smoothed_extents(frames, 1.25 * result.separation, zoom_smoothing, 0.35)
    track_extents = _smoothed_extents(frames, 0.85 * sep_xy, zoom_smoothing, 0.35)

    artists = [
        g_sat_pred, g_deb_pred, g_sat_trail, g_deb_trail, g_sat, g_deb,
        t_sat_pred, t_deb_pred, t_sat_trail, t_deb_trail, t_sat, t_deb,
        h_pred, h_trail, h_sat, h_deb,
        readout, phase_text, burn_flash,
    ]

    def update(i: int):
        k = frames[i]
        t_now = result.t[k]
        # Trail spans a time window of the *fine* samples, so it renders as a
        # smooth arc no matter how far apart the frames are.
        lo = int(np.searchsorted(result.t, t_now - trail_seconds))
        window = slice(lo, k + 1)

        # Forecast from where each body is *now*, so the dotted curve always
        # emanates from its marker (and is never a stale arc from an old burn).
        sat_fc = _states_along(result.sat_r[k], result.sat_v[k], f_times, mu)
        deb_fc = _states_along(result.debris_r[k], result.debris_v[k], f_times, mu)
        sat_pred_xy = project(np.array([s[0] for s in sat_fc]), e1, e2)
        deb_pred_xy = project(np.array([d[0] for d in deb_fc]), e1, e2)

        for sp, dp, st, dt_, s, d in (
            (g_sat_pred, g_deb_pred, g_sat_trail, g_deb_trail, g_sat, g_deb),
            (t_sat_pred, t_deb_pred, t_sat_trail, t_deb_trail, t_sat, t_deb),
        ):
            sp.set_data(sat_pred_xy[:, 0], sat_pred_xy[:, 1])
            dp.set_data(deb_pred_xy[:, 0], deb_pred_xy[:, 1])
            st.set_data(sat_xy[window, 0], sat_xy[window, 1])
            dt_.set_data(deb_xy[window, 0], deb_xy[window, 1])
            s.set_data([sat_xy[k, 0]], [sat_xy[k, 1]])
            d.set_data([deb_xy[k, 0]], [deb_xy[k, 1]])

        ph = np.array(
            [relative_state(s[0], s[1], d[0], d[1], mu)[0] for s, d in zip(sat_fc, deb_fc)]
        )
        h_pred.set_data(ph[:, 1], ph[:, 0])  # along-track horizontal, radial vertical
        h_trail.set_data(result.rel_r[window, 1], result.rel_r[window, 0])
        h_sat.set_data([result.rel_r[k, 1]], [result.rel_r[k, 0]])
        h_deb.set_data([0.0], [0.0])

        # Tracking camera: centre on the pair's midpoint in the ECI plane.
        half = track_extents[i]
        ax_track.set_xlim(mid_xy[k, 0] - half, mid_xy[k, 0] + half)
        ax_track.set_ylim(mid_xy[k, 1] - half, mid_xy[k, 1] + half)

        # Hill camera: centre on the chaser/debris midpoint (debris is at 0).
        cx, cy = result.rel_r[k, 1] / 2.0, result.rel_r[k, 0] / 2.0
        hh = hill_extents[i]
        ax_hill.set_xlim(cx - hh, cx + hh)
        ax_hill.set_ylim(cy - hh, cy + hh)

        n_burns = int((result.burn_times <= t_now).sum())
        readout.set_text(
            f"t = {t_now / 3600:6.2f} h   separation = {result.separation[k]:9.3f} km"
            f"   burns = {n_burns}"
        )
        if phases:
            active = [lbl for t0, lbl in phases if t_now >= t0]
            phase_text.set_text(active[-1] if active else "")
        prev_t = result.t[frames[i - 1]] if i > 0 else t_now
        just_burned = np.any((result.burn_times > prev_t - 1e-9) & (result.burn_times <= t_now))
        burn_flash.set_text("BURN" if just_burned else "")
        return artists

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output_path
