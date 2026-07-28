"""Closed-loop rendezvous guidance: a small APPROACH -> HOLD state machine.

The policy is called at each burn/decision point with the current ECI states
and returns (dv_eci, coast_duration): the impulse to apply now and how long to
coast before the next decision. Each decision applies at most one burn, so a
two-impulse transfer spans two decisions (set-up burn, then null-velocity burn
on arrival). Burns are computed with the CW two-impulse solution but always
from the *actual* measured relative state, so the loop self-corrects for the
linearization error against the real two-body truth.

Station-keeping exploits the asymmetry of the Hill axes. An along-track offset
is free (same orbit, different phase, identical period - it simply sits there);
a radial offset is not (different radius -> different period -> along-track
drift growing at 3*pi*da per orbit). So HOLD:

  * uses a per-axis deadband - tight radially, loose along-track;
  * prefers a cheap "drift-null" burn that enforces the CW no-drift condition
    ydot = -2*n*x, stopping secular drift *wherever it is* for a few tenths of
    a mm/s, over an expensive two-impulse transfer that buys back an
    along-track position error that was never costing anything.

Repositioning is kept only as a fallback for genuinely large excursions.
"""

import numpy as np

from orbdebris.propulsion import STATIONKEEP as STATIONKEEP_PHASE
from orbdebris.propulsion import TERMINAL as TERMINAL_PHASE
from orbdebris.propulsion import FuelLedger
from orbdebris.relative import (
    eci_to_hill,
    hill_dv_to_eci,
    relative_state,
    two_impulse_transfer,
)

APPROACH = "APPROACH"
ARRIVING = "ARRIVING"
HOLD = "HOLD"


class RendezvousPolicy:
    def __init__(
        self,
        mu: float,
        hold_offset: np.ndarray,
        transfer_time: float,
        hold_check_interval: float,
        deadband: float | np.ndarray,
        drift_tolerance: float = 1e-7,
        ledger: FuelLedger | None = None,
        vehicle: str = "chaser",
    ):
        """deadband: per-axis position tolerance [radial, along-track,
        cross-track] in km. A scalar broadcasts to an isotropic band.
        drift_tolerance: along-track velocity error [km/s] above which the
        cheap drift-null burn fires.
        ledger: optional FuelLedger; every burn is recorded against `vehicle`
        under the TERMINAL phase. Station-keeping is a *continuing* cost, so
        without a ledger it was previously unmeasurable - simulate only kept
        burn times."""
        self.mu = mu
        self.hold_offset = np.asarray(hold_offset, dtype=float)
        self.transfer_time = transfer_time
        self.hold_check_interval = hold_check_interval
        self.deadband = np.broadcast_to(np.asarray(deadband, dtype=float), (3,)).copy()
        self.drift_tolerance = drift_tolerance
        self.ledger = ledger
        self.vehicle = vehicle
        self.mode = APPROACH

    def _spend(self, t: float, dv_eci: np.ndarray, phase: str = TERMINAL_PHASE) -> np.ndarray:
        """Record a burn (km/s in, m/s stored) and hand it back unchanged."""
        if self.ledger is not None:
            self.ledger.record_kms(t, float(np.linalg.norm(dv_eci)), phase, self.vehicle)
        return dv_eci

    def decide(
        self,
        t: float,
        chaser_r: np.ndarray,
        chaser_v: np.ndarray,
        target_r: np.ndarray,
        target_v: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        q, n = eci_to_hill(target_r, target_v, self.mu)
        rel_r, rel_v = relative_state(chaser_r, chaser_v, target_r, target_v, self.mu)

        if self.mode == APPROACH:
            dv1, _ = two_impulse_transfer(rel_r, rel_v, self.hold_offset, self.transfer_time, n)
            self.mode = ARRIVING
            return self._spend(t, hill_dv_to_eci(dv1, q)), self.transfer_time

        if self.mode == ARRIVING:
            # Null the actual arrival relative velocity to park at the hold point.
            self.mode = HOLD
            return self._spend(t, hill_dv_to_eci(-rel_v, q)), self.hold_check_interval

        # HOLD. Fallback: a genuinely large excursion on any axis needs a real
        # reposition (expensive - it buys back position).
        if np.any(np.abs(rel_r - self.hold_offset) > self.deadband):
            dv1, _ = two_impulse_transfer(rel_r, rel_v, self.hold_offset, self.transfer_time, n)
            self.mode = ARRIVING
            return self._spend(t, hill_dv_to_eci(dv1, q)), self.transfer_time

        # Otherwise just kill the secular drift. CW drifts along-track at
        # -(6*n*x + 3*ydot) per unit time, so it vanishes exactly when
        # ydot = -2*n*x. One small along-track impulse enforces that.
        dv_y = -(2 * n * rel_r[0] + rel_v[1])
        if abs(dv_y) > self.drift_tolerance:
            dv = hill_dv_to_eci(np.array([0.0, dv_y, 0.0]), q)
            # Billed to STATIONKEEP, not TERMINAL: getting there is paid once,
            # staying there is paid forever, and mixing them hides which is
            # which. This is the only recurring burn the chaser makes.
            return self._spend(t, dv, STATIONKEEP_PHASE), self.hold_check_interval

        return np.zeros(3), self.hold_check_interval
