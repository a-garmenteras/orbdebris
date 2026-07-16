"""Guidance laws for the chaser satellite."""

import numpy as np


def naive_burn(
    sat_r: np.ndarray, sat_v: np.ndarray, debris_r: np.ndarray, delta_v_mag: float
) -> np.ndarray:
    """Return a new satellite velocity after a fixed-size impulse aimed at
    the debris' current position (no intercept-point prediction)."""
    direction = debris_r - sat_r
    norm = np.linalg.norm(direction)
    if norm == 0:
        return sat_v
    return sat_v + delta_v_mag * direction / norm
