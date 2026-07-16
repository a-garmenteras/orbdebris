import numpy as np

from orbdebris.constants import GM_EARTH, R_EARTH
from orbdebris.dynamics import propagate
from orbdebris.relative import (
    cw_propagate,
    eci_to_hill,
    hill_state_to_eci,
    mean_motion,
    relative_state,
    two_impulse_transfer,
)
from orbdebris.state import elements_to_state


def circular_target(a=R_EARTH + 500.0, i=np.radians(51.6)):
    r, v = elements_to_state(a=a, e=0.0, i=i, raan=0.0, argp=0.0, nu=0.0, mu=GM_EARTH)
    return r, v, a


def test_frame_round_trip():
    target_r, target_v, _ = circular_target()
    rel_r = np.array([1.2, -3.4, 0.7])
    rel_v = np.array([0.002, -0.001, 0.0005])

    chaser_r, chaser_v = hill_state_to_eci(rel_r, rel_v, target_r, target_v, GM_EARTH)
    back_r, back_v = relative_state(chaser_r, chaser_v, target_r, target_v, GM_EARTH)

    assert np.allclose(back_r, rel_r, atol=1e-9)
    assert np.allclose(back_v, rel_v, atol=1e-12)


def test_cw_stm_matches_nonlinear_truth_for_small_separation():
    # Compare CW propagation against the real two-body dynamics: seed a small
    # relative offset, propagate both bodies nonlinearly, transform the
    # difference into the Hill frame, and check the CW STM tracks it.
    target_r, target_v, a = circular_target()
    n = mean_motion(a, GM_EARTH)

    rel_r0 = np.array([0.5, 1.0, 0.3])  # km, small vs ~6878 km radius
    rel_v0 = np.array([0.0, 0.0, 0.0])
    chaser_r0, chaser_v0 = hill_state_to_eci(rel_r0, rel_v0, target_r, target_v, GM_EARTH)

    period = 2 * np.pi / n
    t_eval = np.linspace(0, period / 4, 40)
    _, tgt_r, tgt_v = propagate(target_r, target_v, (0, t_eval[-1]), t_eval, GM_EARTH)
    _, chs_r, chs_v = propagate(chaser_r0, chaser_v0, (0, t_eval[-1]), t_eval, GM_EARTH)

    for k, t in enumerate(t_eval):
        truth_rel_r, _ = relative_state(chs_r[k], chs_v[k], tgt_r[k], tgt_v[k], GM_EARTH)
        cw_rel_r, _ = cw_propagate(rel_r0, rel_v0, t, n)
        # Linearization error grows with time/separation; a few metres over a
        # quarter orbit is well within tolerance for a ~1 km offset.
        assert np.linalg.norm(truth_rel_r - cw_rel_r) < 0.02


def test_two_impulse_transfer_reaches_target_in_cw():
    _, _, a = circular_target()
    n = mean_motion(a, GM_EARTH)

    rel_r = np.array([2.0, 5.0, 1.0])
    rel_v = np.array([0.0, 0.0, 0.0])
    target_rel_r = np.array([0.0, 1.0, 0.0])  # park 1 km ahead (along-track)
    transfer_time = 0.4 * (2 * np.pi / n)

    dv1, dv2 = two_impulse_transfer(rel_r, rel_v, target_rel_r, transfer_time, n)

    arrived_r, arrived_v = cw_propagate(rel_r, rel_v + dv1, transfer_time, n)
    assert np.allclose(arrived_r, target_rel_r, atol=1e-9)
    assert np.allclose(arrived_v + dv2, np.zeros(3), atol=1e-9)


def test_along_track_offset_is_cw_equilibrium():
    _, _, a = circular_target()
    n = mean_motion(a, GM_EARTH)

    rel_r = np.array([0.0, 1.0, 0.0])  # 1 km ahead, on the same orbit
    rel_v = np.array([0.0, 0.0, 0.0])

    for frac in np.linspace(0, 1, 5):
        r_t, v_t = cw_propagate(rel_r, rel_v, frac * (2 * np.pi / n), n)
        assert np.allclose(r_t, rel_r, atol=1e-9)
        assert np.allclose(v_t, rel_v, atol=1e-9)


def test_hill_axes_are_orthonormal():
    target_r, target_v, _ = circular_target()
    q, _ = eci_to_hill(target_r, target_v, GM_EARTH)
    assert np.allclose(q @ q.T, np.eye(3), atol=1e-12)
