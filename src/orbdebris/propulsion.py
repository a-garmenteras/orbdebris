"""Fuel accounting: one ledger for every delta-v the mission spends, and the
rocket equation that turns it into kilograms.

Why this module exists
----------------------
Delta-v is *created* in three different shapes in this codebase, which is how we
ended up with three disjoint fuel stories and no single one:

  * discrete ECI impulses in km/s              (simulate.ScenarioResult)
  * continuous Hill-frame thrust in m/s,
    integrated per timestep                    (constellation.FormationController)
  * a planner's *estimate*, never reconciled
    against what was actually flown            (phasing.TransferPlan)

Different units, different frames, different time bases. So the fix is a
*boundary*, not a rewrite: every source converts once, at the point it hands
delta-v to the ledger. Canonical unit here is **m/s, magnitude only** - the same
single-conversion-point discipline ``capture.CaptureSim.from_hill_state`` uses
for km -> m.

Honest caveat on "magnitude only": summing |dv| is exactly right for propellant
(every burn costs fuel regardless of which way it points) but it is *not* the
vector sum and it is not invertible. Nothing in a trajectory can be
reconstructed from this ledger, and nothing should try.

Entries, never totals
---------------------
The ledger stores individual burns and *derives* every total by query. That way
the headline number and the breakdown come from one source of truth and cannot
drift apart, and "which satellite is thirstiest" is a question you ask rather
than a field somebody has to remember to update.

The rocket equation, and where it gets interesting
--------------------------------------------------
Tsiolkovsky: ``dv = v_e * ln(m0 / m1)`` with ``v_e = Isp * g0``.

For one vehicle burning one propellant, propellant computed from the *summed*
delta-v is exactly equal to replaying the burns one at a time, because the mass
ratios telescope::

    m0/m1 * m1/m2 * ... = m0/mN

So you would think firing order never matters. It matters here in two real
places:

  * **The split.** A 500 kg chaser becomes 4 x 125 kg satellites. Phasing
    delta-v is paid at 500 kg; everything afterwards is paid per-satellite at
    125. That is a mass discontinuity the telescoping argument does not cross,
    which is why entries are ordered and attributed to a named vehicle, and why
    ``split_vehicle`` exists.
  * **Per-phase attribution.** Propellant for a *subset* of burns depends on the
    mass the vehicle had when those particular burns fired. So per-phase
    propellant is computed by replay, not by applying Tsiolkovsky to the phase's
    delta-v from the wet mass. Done this way the per-phase numbers sum exactly
    to the total - see ``FuelLedger.propellant_by``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from orbdebris.constants import G0, KM_TO_M

# Phase labels. Free strings are accepted; these are the ones the reference
# mission uses, named so typos become import errors instead of silent extra rows.
PHASING = "PHASING"  # drift-to-window + Lambert transfer (M2b)
TERMINAL = "TERMINAL"  # CW terminal transfer: closing the last ~20 km (M2)
STATIONKEEP = "STATIONKEEP"  # holding the standoff once there - a *continuing* cost
SPLIT = "SPLIT"  # the chaser dividing into four (M4)
SWEEP = "SWEEP"  # the radial-down burn that closes on the cloud (M4)
TENSION = "TENSION"  # apex thruster unfurling the funnel (M4)
FORMATION = "FORMATION"  # holding the funnel's mouth open (M4)


@dataclass(frozen=True)
class Vehicle:
    """A thing with a tank. Masses in kg, Isp in seconds."""

    name: str
    dry_mass: float
    prop_mass: float
    isp: float

    def __post_init__(self) -> None:
        if self.dry_mass <= 0:
            raise ValueError(f"{self.name}: dry_mass must be positive, got {self.dry_mass}.")
        if self.prop_mass < 0:
            raise ValueError(f"{self.name}: prop_mass must be >= 0, got {self.prop_mass}.")
        if self.isp <= 0:
            raise ValueError(f"{self.name}: isp must be positive, got {self.isp}.")

    @property
    def wet_mass(self) -> float:
        return self.dry_mass + self.prop_mass

    @property
    def exhaust_velocity(self) -> float:
        """v_e = Isp * g0 [m/s]. This is the only place g0 enters."""
        return self.isp * G0

    def dv_capacity(self) -> float:
        """Total delta-v available if the tank is run dry [m/s]."""
        return self.exhaust_velocity * math.log(self.wet_mass / self.dry_mass)

    def propellant_for(self, dv: float, start_mass: float | None = None) -> float:
        """Propellant [kg] to produce dv [m/s], starting from start_mass
        (default: wet mass). Note this is *not* linear in dv - the same delta-v
        costs less propellant later in a mission, when the vehicle is lighter."""
        if dv < 0:
            raise ValueError(f"dv must be >= 0, got {dv}.")
        m0 = self.wet_mass if start_mass is None else start_mass
        return m0 * (1.0 - math.exp(-dv / self.exhaust_velocity))

    def dv_for(self, propellant: float, start_mass: float | None = None) -> float:
        """Delta-v [m/s] obtainable by burning `propellant` kg from start_mass."""
        if propellant < 0:
            raise ValueError(f"propellant must be >= 0, got {propellant}.")
        m0 = self.wet_mass if start_mass is None else start_mass
        if propellant >= m0:
            raise ValueError(
                f"{self.name}: cannot burn {propellant:.3f} kg from a {m0:.3f} kg vehicle."
            )
        return self.exhaust_velocity * math.log(m0 / (m0 - propellant))


@dataclass(frozen=True)
class BurnEntry:
    """One recorded expenditure. dv is a magnitude in m/s - see module docstring."""

    t: float  # mission elapsed time [s]
    dv: float  # [m/s], magnitude
    phase: str
    vehicle: str


def split_vehicle(
    parent: Vehicle,
    remaining_mass: float,
    names: list[str],
    dry_mass_each: float,
    isp: float | None = None,
) -> list[Vehicle]:
    """Divide a parent vehicle's *remaining* mass into N children.

    This is the mass discontinuity the telescoping argument cannot cross: the
    parent's propellant is already partly spent, so the children inherit what is
    left, not what launched. Mass is conserved exactly - the test asserts it.
    """
    n = len(names)
    if n == 0:
        raise ValueError("split_vehicle needs at least one child name.")
    share = remaining_mass / n
    prop_each = share - dry_mass_each
    if prop_each < 0:
        raise ValueError(
            f"Splitting {remaining_mass:.1f} kg into {n} gives {share:.1f} kg each, "
            f"which is less than the {dry_mass_each:.1f} kg dry mass of a child."
        )
    return [
        Vehicle(name=name, dry_mass=dry_mass_each, prop_mass=prop_each, isp=isp or parent.isp)
        for name in names
    ]


@dataclass
class FuelLedger:
    """Every delta-v the mission spends, in one ordered, attributed list."""

    vehicles: dict[str, Vehicle] = field(default_factory=dict)
    entries: list[BurnEntry] = field(default_factory=list)

    # ------------------------------------------------------------- recording
    def add_vehicle(self, vehicle: Vehicle) -> Vehicle:
        if vehicle.name in self.vehicles:
            raise ValueError(f"Vehicle {vehicle.name!r} is already in the ledger.")
        self.vehicles[vehicle.name] = vehicle
        return vehicle

    def record(self, t: float, dv: float, phase: str, vehicle: str) -> None:
        """Record a burn. dv in **m/s**, magnitude. Zero-dv burns are dropped."""
        if vehicle not in self.vehicles:
            raise KeyError(f"Unknown vehicle {vehicle!r}; add_vehicle it first.")
        if not math.isfinite(dv) or dv < 0:
            raise ValueError(f"dv must be finite and >= 0, got {dv}.")
        if dv == 0.0:
            return
        self.entries.append(BurnEntry(float(t), float(dv), phase, vehicle))

    def record_kms(self, t: float, dv_kms: float, phase: str, vehicle: str) -> None:
        """Convenience for the orbital modules, which work in km/s. This is the
        unit boundary: km/s goes in, m/s is stored."""
        self.record(t, dv_kms * KM_TO_M, phase, vehicle)

    # --------------------------------------------------------------- queries
    def _select(self, phase: str | None, vehicle: str | None) -> list[BurnEntry]:
        return [
            e
            for e in self.entries
            if (phase is None or e.phase == phase) and (vehicle is None or e.vehicle == vehicle)
        ]

    def total_dv(self, phase: str | None = None, vehicle: str | None = None) -> float:
        """Summed delta-v [m/s] over the selected entries."""
        return math.fsum(e.dv for e in self._select(phase, vehicle))

    def phases(self) -> list[str]:
        """Phase labels in first-appearance order."""
        return list(dict.fromkeys(e.phase for e in self.entries))

    def burn_count(self, phase: str | None = None, vehicle: str | None = None) -> int:
        return len(self._select(phase, vehicle))

    # ------------------------------------------------------------ mass replay
    def _replay(self, vehicle: str) -> list[tuple[BurnEntry, float]]:
        """Replay one vehicle's burns in time order, returning (entry,
        propellant_kg) pairs. Propellant is path-dependent, so this - not
        ``propellant_for(total_dv)`` - is what makes per-phase numbers add up."""
        v = self.vehicles[vehicle]
        ve = v.exhaust_velocity
        mass = v.wet_mass
        out = []
        for e in sorted(self._select(None, vehicle), key=lambda e: e.t):
            after = mass * math.exp(-e.dv / ve)
            out.append((e, mass - after))
            mass = after
        return out

    def propellant_by(self, phase: str | None = None, vehicle: str | None = None) -> float:
        """Propellant [kg] burned by the selected entries, via mass replay.

        Because replay is exact, ``sum(propellant_by(phase=p) for p in phases())``
        equals ``propellant_by()`` - and so does the per-vehicle partition. Both
        are asserted in the tests; a mismatch means double counting.
        """
        names = [vehicle] if vehicle is not None else list(self.vehicles)
        return math.fsum(
            prop
            for name in names
            for e, prop in self._replay(name)
            if phase is None or e.phase == phase
        )

    def remaining_propellant(self, vehicle: str) -> float:
        """Propellant [kg] left in the tank. Negative means the mission was
        infeasible - the ledger reports it rather than clamping it, because a
        silently clamped budget is a budget that never binds."""
        return self.vehicles[vehicle].prop_mass - self.propellant_by(vehicle=vehicle)

    def remaining_mass(self, vehicle: str) -> float:
        """Current total mass [kg] after every recorded burn."""
        return self.vehicles[vehicle].wet_mass - self.propellant_by(vehicle=vehicle)

    def feasible(self) -> bool:
        return all(self.remaining_propellant(n) >= 0.0 for n in self.vehicles)

    # ---------------------------------------------------------------- report
    def table(self) -> str:
        """The one table: per phase, per vehicle, m/s and kg."""
        rows = [f"{'phase':<12}{'vehicle':<10}{'burns':>7}{'dv [m/s]':>12}{'prop [kg]':>12}"]
        rows.append("-" * len(rows[0]))
        for phase in self.phases():
            names = [n for n in self.vehicles if self.burn_count(phase=phase, vehicle=n)]
            for name in names:
                rows.append(
                    f"{phase:<12}{name:<10}"
                    f"{self.burn_count(phase=phase, vehicle=name):>7}"
                    f"{self.total_dv(phase=phase, vehicle=name):>12.3f}"
                    f"{self.propellant_by(phase=phase, vehicle=name):>12.3f}"
                )
            if len(names) > 1:
                rows.append(
                    f"{'':<12}{'(all)':<10}{self.burn_count(phase=phase):>7}"
                    f"{self.total_dv(phase=phase):>12.3f}"
                    f"{self.propellant_by(phase=phase):>12.3f}"
                )
        rows.append("-" * len(rows[0]))
        rows.append(
            f"{'TOTAL':<12}{'':<10}{len(self.entries):>7}"
            f"{self.total_dv():>12.3f}{self.propellant_by():>12.3f}"
        )
        for name, v in self.vehicles.items():
            left = self.remaining_propellant(name)
            flag = "" if left >= 0 else "   <- OVER BUDGET"
            rows.append(
                f"  {name:<10} spent {self.propellant_by(vehicle=name):7.3f} kg of "
                f"{v.prop_mass:.3f} kg  ({left:+.3f} kg left, "
                f"{self.total_dv(vehicle=name):.2f} of {v.dv_capacity():.1f} m/s){flag}"
            )
        return "\n".join(rows)
