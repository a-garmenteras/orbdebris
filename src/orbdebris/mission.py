"""Full-mission guidance: drift for free until the cheap phasing window, run a
Lambert transfer into CW range, then hand off to the terminal CW rendezvous.

State machine DRIFT -> DEPART -> ARRIVE -> TERMINAL, wrapping a
RendezvousPolicy. Same ``decide()`` signature as RendezvousPolicy, so
simulate.simulate drives it unchanged.

The schedule (when to depart, how long to transfer) comes from
phasing.plan_min_fuel_transfer up front - mission design happens before flight.
The burns themselves are still solved from the *actual* measured state at
execution time, so the loop stays closed.
"""

import numpy as np

from orbdebris.kepler import kepler_propagate
from orbdebris.lambert import lambert
from orbdebris.phasing import TransferPlan
from orbdebris.propulsion import PHASING as PHASING_PHASE
from orbdebris.propulsion import FuelLedger
from orbdebris.relative import hill_state_to_eci
from orbdebris.rendezvous import RendezvousPolicy

DRIFT = "DRIFT"
DEPART = "DEPART"
ARRIVE = "ARRIVE"
TERMINAL = "TERMINAL"


class MissionPolicy:
    def __init__(
        self,
        mu: float,
        plan: TransferPlan,
        handoff_offset: np.ndarray,
        terminal_policy: RendezvousPolicy,
        arrival_coast: float,
        ledger: FuelLedger | None = None,
        vehicle: str = "chaser",
    ):
        """ledger: optional FuelLedger. The two transfer burns are recorded
        under PHASING; the nested terminal policy records its own under
        TERMINAL, so the split between "getting there" and "staying there"
        falls out of the ledger rather than needing to be reconstructed."""
        self.mu = mu
        self.plan = plan
        self.handoff_offset = np.asarray(handoff_offset, dtype=float)
        self.terminal = terminal_policy
        self.arrival_coast = arrival_coast
        self.ledger = ledger
        self.vehicle = vehicle
        self.mode = DRIFT
        self.depart_dv = 0.0
        self.arrive_dv = 0.0

    def _spend(self, t: float, dv: np.ndarray) -> float:
        """Record a phasing burn (km/s in, m/s stored); returns its magnitude."""
        mag = float(np.linalg.norm(dv))
        if self.ledger is not None:
            self.ledger.record_kms(t, mag, PHASING_PHASE, self.vehicle)
        return mag

    def decide(
        self,
        t: float,
        sat_r: np.ndarray,
        sat_v: np.ndarray,
        debris_r: np.ndarray,
        debris_v: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if self.mode == DRIFT:
            self.mode = DEPART
            # Coast, unpowered, until the planned phasing window opens.
            if self.plan.t_depart > 1.0:
                return np.zeros(3), self.plan.t_depart

        if self.mode == DEPART:
            # Re-solve from the actual state: aim at a standoff point offset
            # from where the debris will be at arrival.
            dr, dv_deb = kepler_propagate(debris_r, debris_v, self.plan.tof, self.mu)
            target_r, _ = hill_state_to_eci(
                self.handoff_offset, np.zeros(3), dr, dv_deb, self.mu
            )
            v1, _ = lambert(sat_r, target_r, self.plan.tof, self.mu)
            dv = v1 - sat_v
            self.depart_dv = self._spend(t, dv)
            self.mode = ARRIVE
            return dv, self.plan.tof

        if self.mode == ARRIVE:
            # Match the velocity that parks us at the standoff with ~zero
            # relative velocity (a CW equilibrium), handing a clean state to CW.
            _, desired_v = hill_state_to_eci(
                self.handoff_offset, np.zeros(3), debris_r, debris_v, self.mu
            )
            dv = desired_v - sat_v
            self.arrive_dv = self._spend(t, dv)
            self.mode = TERMINAL
            return dv, self.arrival_coast

        return self.terminal.decide(t, sat_r, sat_v, debris_r, debris_v)
