import numpy as np
import pytest

from orbdebris.constants import GM_EARTH, R_EARTH
from orbdebris.dynamics import propagate
from orbdebris.kepler import kepler_propagate
from orbdebris.state import elements_to_state


@pytest.mark.parametrize("e", [0.0, 0.1, 0.4])
@pytest.mark.parametrize("dt_frac", [0.25, 1.0, 5.0])
def test_analytic_matches_numerical_propagation(e, dt_frac):
    # The analytic (universal-variable) coast and the numerical integrator
    # solve the same two-body problem, so they must agree.
    a = R_EARTH + 600.0
    r0, v0 = elements_to_state(
        a=a, e=e, i=np.radians(40), raan=np.radians(20), argp=np.radians(10), nu=0.3, mu=GM_EARTH
    )
    period = 2 * np.pi * np.sqrt(a**3 / GM_EARTH)
    dt = dt_frac * period

    r_an, v_an = kepler_propagate(r0, v0, dt, GM_EARTH)
    _, r_num, v_num = propagate(r0, v0, (0.0, dt), np.array([dt]), GM_EARTH)

    assert np.allclose(r_an, r_num[-1], atol=1e-5)
    assert np.allclose(v_an, v_num[-1], atol=1e-8)


def test_zero_dt_is_identity():
    r0 = np.array([7000.0, 0.0, 0.0])
    v0 = np.array([0.0, 7.5, 0.1])
    r, v = kepler_propagate(r0, v0, 0.0, GM_EARTH)
    assert np.array_equal(r, r0)
    assert np.array_equal(v, v0)


def test_full_period_returns_to_start():
    a = R_EARTH + 500.0
    r0, v0 = elements_to_state(a=a, e=0.15, i=0.4, raan=0.0, argp=0.0, nu=0.0, mu=GM_EARTH)
    period = 2 * np.pi * np.sqrt(a**3 / GM_EARTH)

    r, v = kepler_propagate(r0, v0, period, GM_EARTH)

    assert np.allclose(r, r0, atol=1e-6)
    assert np.allclose(v, v0, atol=1e-9)
