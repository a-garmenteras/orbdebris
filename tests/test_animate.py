import numpy as np

from orbdebris.animate import (
    animate_mission,
    ballistic_forecast,
    forecast_times,
    hill_forecast,
    orbital_plane_basis,
    project,
)
from orbdebris.constants import GM_EARTH, R_EARTH
from orbdebris.dynamics import propagate
from orbdebris.relative import hill_state_to_eci, mean_motion
from orbdebris.rendezvous import RendezvousPolicy
from orbdebris.simulate import simulate
from orbdebris.state import elements_to_state


def _circular(a=R_EARTH + 500.0, nu=0.0):
    return elements_to_state(
        a=a, e=0.0, i=np.radians(51.6), raan=np.radians(30), argp=0.0, nu=nu, mu=GM_EARTH
    )


def test_orbital_plane_basis_is_orthonormal():
    r, v = _circular()
    e1, e2 = orbital_plane_basis(r, v)

    assert np.isclose(np.linalg.norm(e1), 1.0)
    assert np.isclose(np.linalg.norm(e2), 1.0)
    assert np.isclose(np.dot(e1, e2), 0.0, atol=1e-12)


def test_projection_flattens_a_coplanar_orbit_losslessly():
    # A circular orbit projected onto its own plane must stay a circle of the
    # same radius: no information is lost, which is why the ECI panel is 2D.
    a = R_EARTH + 500.0
    r0, v0 = _circular(a)
    e1, e2 = orbital_plane_basis(r0, v0)
    period = 2 * np.pi / mean_motion(a, GM_EARTH)
    t_eval = np.linspace(0, period, 60)
    _, r, _ = propagate(r0, v0, (0, period), t_eval, GM_EARTH)

    xy = project(r, e1, e2)

    assert np.allclose(np.linalg.norm(xy, axis=1), a, rtol=1e-6)
    assert np.allclose(np.linalg.norm(r, axis=1), np.linalg.norm(xy, axis=1), rtol=1e-9)


def test_ballistic_forecast_matches_numerical_propagation():
    r0, v0 = _circular()
    times = np.linspace(0, 1800.0, 25)

    fc = ballistic_forecast(r0, v0, times, GM_EARTH)

    _, r_num, _ = propagate(r0, v0, (0, times[-1]), times, GM_EARTH)
    assert np.allclose(fc, r_num, atol=1e-5)


def test_forecast_starts_exactly_at_the_current_position():
    """The dotted curve must emanate from the body's marker. Forecasting from a
    stale state (e.g. the last burn) detaches it - visibly so when zoomed in."""
    r0, v0 = _circular()
    fc = ballistic_forecast(r0, v0, forecast_times(5000.0), GM_EARTH)
    assert np.allclose(fc[0], r0, atol=1e-9)


def test_forecast_times_are_packed_near_the_present():
    """Uniform sampling puts ~365 km between points at orbital speed, whose
    chords cut kilometres inside the true arc - fatal at kilometre zoom."""
    times = forecast_times(5760.0, n_points=140, bias=3.0)

    assert times[0] == 0.0
    assert np.isclose(times[-1], 5760.0)
    assert np.all(np.diff(times) > 0)
    first_gap = times[1] - times[0]
    last_gap = times[-1] - times[-2]
    assert first_gap < 0.01 * last_gap  # dense now, sparse later


def test_hill_forecast_starts_at_the_current_relative_position():
    debris_r, debris_v = _circular()
    rel0 = np.array([0.0, -5.0, 0.5])
    sat_r, sat_v = hill_state_to_eci(rel0, np.zeros(3), debris_r, debris_v, GM_EARTH)

    fc = hill_forecast(sat_r, sat_v, debris_r, debris_v, np.linspace(0, 600.0, 10), GM_EARTH)

    assert fc.shape == (10, 3)
    assert np.allclose(fc[0], rel0, atol=1e-6)


def test_animate_mission_writes_a_gif(tmp_path):
    """Smoke test: few frames, short scenario - just prove it renders."""
    a = R_EARTH + 500.0
    period = 2 * np.pi / mean_motion(a, GM_EARTH)
    debris_r, debris_v = _circular(a)
    sat_r, sat_v = hill_state_to_eci(
        np.array([0.0, -20.0, 0.0]), np.zeros(3), debris_r, debris_v, GM_EARTH
    )
    policy = RendezvousPolicy(
        mu=GM_EARTH,
        hold_offset=np.array([0.0, 1.0, 0.0]),
        transfer_time=0.4 * period,
        hold_check_interval=period / 20,
        deadband=np.array([0.02, 0.5, 0.05]),
    )
    result = simulate(
        sat_r, sat_v, debris_r, debris_v, policy.decide, 0.6 * period, GM_EARTH, dt=120.0
    )

    out = tmp_path / "smoke.gif"
    animate_mission(result, GM_EARTH, output_path=str(out), n_frames=6, trail_seconds=600.0, dpi=50)

    assert out.exists()
    assert out.stat().st_size > 0


def test_animate_mission_accepts_explicit_frame_times(tmp_path):
    """frame_times is what lets a caller time-lapse quiet stretches."""
    a = R_EARTH + 500.0
    period = 2 * np.pi / mean_motion(a, GM_EARTH)
    debris_r, debris_v = _circular(a)
    sat_r, sat_v = hill_state_to_eci(
        np.array([0.0, -20.0, 0.0]), np.zeros(3), debris_r, debris_v, GM_EARTH
    )
    policy = RendezvousPolicy(
        mu=GM_EARTH,
        hold_offset=np.array([0.0, 1.0, 0.0]),
        transfer_time=0.4 * period,
        hold_check_interval=period / 20,
        deadband=np.array([0.02, 0.5, 0.05]),
    )
    result = simulate(
        sat_r, sat_v, debris_r, debris_v, policy.decide, 0.6 * period, GM_EARTH, dt=120.0
    )

    out = tmp_path / "times.gif"
    animate_mission(
        result,
        GM_EARTH,
        output_path=str(out),
        frame_times=np.linspace(0.0, result.t[-1], 5),
        trail_seconds=600.0,
        dpi=50,
    )

    assert out.exists()
    assert out.stat().st_size > 0
