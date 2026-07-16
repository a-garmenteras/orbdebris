import numpy as np

from orbdebris.constants import GM_EARTH
from orbdebris.dynamics import propagate
from orbdebris.guidance import naive_burn


def test_naive_burn_reduces_separation_versus_no_burn():
    sat_r = np.array([7000.0, 0.0, 0.0])
    sat_v = np.array([0.0, 7.5, 0.0])
    debris_r = np.array([7000.0, 100.0, 0.0])

    burned_v = naive_burn(sat_r, sat_v, debris_r, delta_v_mag=0.05)

    # Short horizon, well before the satellite's ~7.5 km/s motion carries it
    # past the (stationary) debris point at ~13s and the comparison inverts.
    duration = 10.0
    t_eval = np.linspace(0, duration, 10)
    _, r_burn, _ = propagate(sat_r, burned_v, (0, duration), t_eval, GM_EARTH)
    _, r_no_burn, _ = propagate(sat_r, sat_v, (0, duration), t_eval, GM_EARTH)

    final_sep_burn = np.linalg.norm(r_burn[-1] - debris_r)
    final_sep_no_burn = np.linalg.norm(r_no_burn[-1] - debris_r)

    assert final_sep_burn < final_sep_no_burn


def test_naive_burn_is_noop_when_colocated():
    sat_r = np.array([7000.0, 0.0, 0.0])
    sat_v = np.array([0.0, 7.5, 0.0])

    result = naive_burn(sat_r, sat_v, debris_r=sat_r, delta_v_mag=0.05)

    assert np.array_equal(result, sat_v)
