import numpy as np

from orbdebris.constants import GM_EARTH
from orbdebris.dynamics import propagate
from orbdebris.state import elements_to_state


def orbital_period(a: float, mu: float = GM_EARTH) -> float:
    return 2 * np.pi * np.sqrt(a**3 / mu)


def test_circular_orbit_holds_constant_radius():
    a = 7000.0
    r0, v0 = elements_to_state(
        a=a, e=0.0, i=np.radians(30), raan=np.radians(10), argp=0.0, nu=0.0, mu=GM_EARTH
    )
    t_eval = np.linspace(0, orbital_period(a), 200)

    _, r, _ = propagate(r0, v0, (0, t_eval[-1]), t_eval, GM_EARTH)

    radii = np.linalg.norm(r, axis=1)
    assert np.allclose(radii, a, rtol=1e-8)


def test_energy_and_angular_momentum_conserved():
    a, e = 7000.0, 0.1
    r0, v0 = elements_to_state(a=a, e=e, i=np.radians(15), raan=0.0, argp=0.0, nu=0.0, mu=GM_EARTH)
    t_eval = np.linspace(0, 5 * orbital_period(a), 500)

    _, r, v = propagate(r0, v0, (0, t_eval[-1]), t_eval, GM_EARTH)

    r_norm = np.linalg.norm(r, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    specific_energy = 0.5 * v_norm**2 - GM_EARTH / r_norm
    angular_momentum = np.linalg.norm(np.cross(r, v), axis=1)

    assert np.ptp(specific_energy) / abs(specific_energy[0]) < 1e-8
    assert np.ptp(angular_momentum) / angular_momentum[0] < 1e-8


def test_elements_to_state_matches_vis_viva_at_periapsis():
    a, e = 7000.0, 0.2
    r0, v0 = elements_to_state(a=a, e=e, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=GM_EARTH)

    r_periapsis = a * (1 - e)
    expected_speed = np.sqrt(GM_EARTH * (2 / r_periapsis - 1 / a))

    assert np.isclose(np.linalg.norm(r0), r_periapsis)
    assert np.isclose(np.linalg.norm(v0), expected_speed)
