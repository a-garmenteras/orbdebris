"""Debris pellet cloud. Units: METRES (the capture world).

The mission-level simulator treats "the debris" as one point on a circular
orbit. Here that point is revealed to be the *centroid* of a cloud of small
pellets (cm scale) - the actual small-to-medium debris population the net is
for. Pellets sit near the Hill-frame origin with small random offsets and
mm/s relative velocities; over a capture's ~minute the cloud barely disperses,
but each pellet is a full dynamic body once the membrane starts pushing it.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PelletCloud:
    radii: np.ndarray  # [P] pellet radii [m]
    masses: np.ndarray  # [P] [kg]
    positions: np.ndarray  # [P, 3] initial positions about the cloud centroid [m]
    velocities: np.ndarray  # [P, 3] initial velocities [m/s]

    @property
    def n_pellets(self) -> int:
        return len(self.radii)


def make_pellet_cloud(
    n_pellets: int = 30,
    sigma_pos: float = 1.5,
    sigma_vel: float = 0.005,
    diameter_range: tuple[float, float] = (0.01, 0.10),
    density: float = 2700.0,
    seed: int = 7,
) -> PelletCloud:
    """Random cloud: diameters log-uniform in diameter_range (small debris is
    far more common than large - a log-uniform draw is a crude stand-in for
    the real power-law size distribution), masses as solid spheres of the
    given density (aluminium-ish), positions Gaussian about the centroid.
    The sampled positions/velocities are re-centred so the cloud centroid
    starts exactly at the origin at rest - by construction, the point the
    mission-level rendezvous targeted."""
    rng = np.random.default_rng(seed)
    d_lo, d_hi = diameter_range
    diameters = np.exp(rng.uniform(np.log(d_lo), np.log(d_hi), n_pellets))
    radii = diameters / 2.0
    masses = density * 4.0 / 3.0 * np.pi * radii**3

    positions = rng.normal(0.0, sigma_pos, (n_pellets, 3))
    velocities = rng.normal(0.0, sigma_vel, (n_pellets, 3))
    # Re-centre: centroid at the origin, at rest (mass-weighted).
    w = masses / masses.sum()
    positions -= (w[:, None] * positions).sum(axis=0)
    velocities -= (w[:, None] * velocities).sum(axis=0)

    return PelletCloud(radii=radii, masses=masses, positions=positions, velocities=velocities)
