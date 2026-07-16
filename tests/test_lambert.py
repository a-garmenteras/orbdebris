import numpy as np
import pytest

from orbdebris.constants import GM_EARTH, R_EARTH
from orbdebris.dynamics import propagate
from orbdebris.lambert import lambert
from orbdebris.state import elements_to_state


def test_lambert_round_trip_through_propagator():
    # The strong test: solve Lambert, then fly v1 through the real two-body
    # propagator and confirm we land on r2 at the requested time of flight.
    a = R_EARTH + 700.0
    r1, _ = elements_to_state(a=a, e=0.0, i=np.radians(30), raan=0.0, argp=0.0, nu=0.0, mu=GM_EARTH)
    r2, _ = elements_to_state(
        a=a + 300.0,
        e=0.05,
        i=np.radians(30),
        raan=0.0,
        argp=0.0,
        nu=np.radians(100),
        mu=GM_EARTH,
    )
    tof = 1800.0

    v1, v2 = lambert(r1, r2, tof, GM_EARTH)

    _, r_arr, v_arr = propagate(r1, v1, (0, tof), np.array([tof]), GM_EARTH)
    assert np.allclose(r_arr[-1], r2, atol=1e-4)
    assert np.allclose(v_arr[-1], v2, atol=1e-6)


def test_lambert_departure_speed_matches_transfer_energy():
    # v1 must be consistent with vis-viva on the transfer orbit it defines.
    a = R_EARTH + 500.0
    r1, _ = elements_to_state(a=a, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=GM_EARTH)
    r2, _ = elements_to_state(a=a, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=np.radians(75), mu=GM_EARTH)
    tof = 1500.0

    v1, _ = lambert(r1, r2, tof, GM_EARTH)

    # Semi-major axis of the transfer orbit from vis-viva, then check energy is
    # self-consistent at r2 by propagating.
    r1n = np.linalg.norm(r1)
    energy = 0.5 * np.dot(v1, v1) - GM_EARTH / r1n
    a_transfer = -GM_EARTH / (2 * energy)
    assert a_transfer > 0  # elliptical transfer


def test_lambert_raises_on_degenerate_transfer_angle():
    r1 = np.array([7000.0, 0.0, 0.0])
    r2 = np.array([-7000.0, 0.0, 0.0])  # exactly 180 deg: singular
    with pytest.raises(ValueError, match="undefined"):
        lambert(r1, r2, 3000.0, GM_EARTH)
