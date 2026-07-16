"""Conversions between classical orbital elements and Cartesian state vectors."""

import numpy as np


def elements_to_state(
    a: float, e: float, i: float, raan: float, argp: float, nu: float, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert classical orbital elements to a Cartesian (r, v) state, in ECI.

    a: semi-major axis [km]. e: eccentricity. i, raan, argp, nu: inclination,
    right ascension of ascending node, argument of periapsis, true anomaly
    [rad]. mu: gravitational parameter [km^3/s^2].
    """
    p = a * (1 - e**2)
    r_norm = p / (1 + e * np.cos(nu))
    r_pf = r_norm * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_argp, sin_argp = np.cos(argp), np.sin(argp)

    # Perifocal -> ECI rotation, the 3-1-3 Euler sequence Rz(raan) Rx(i) Rz(argp).
    rotation = np.array(
        [
            [
                cos_raan * cos_argp - sin_raan * sin_argp * cos_i,
                -cos_raan * sin_argp - sin_raan * cos_argp * cos_i,
                sin_raan * sin_i,
            ],
            [
                sin_raan * cos_argp + cos_raan * sin_argp * cos_i,
                -sin_raan * sin_argp + cos_raan * cos_argp * cos_i,
                -cos_raan * sin_i,
            ],
            [sin_argp * sin_i, cos_argp * sin_i, cos_i],
        ]
    )

    return rotation @ r_pf, rotation @ v_pf
