"""Fuel ledger and rocket-equation tests.

The interesting ones are `test_telescoping` (why firing order usually does NOT
matter) and `test_partition_invariant` (the guard against double counting, the
likeliest failure mode for accounting code).
"""

import math

import pytest

from orbdebris.propulsion import (
    FORMATION,
    PHASING,
    TERMINAL,
    FuelLedger,
    Vehicle,
    split_vehicle,
)


def chaser() -> Vehicle:
    return Vehicle(name="chaser", dry_mass=400.0, prop_mass=100.0, isp=220.0)


# ------------------------------------------------------------------ Tsiolkovsky
def test_dv_capacity_matches_hand_calculation():
    v = chaser()
    expected = 220.0 * 9.80665 * math.log(500.0 / 400.0)
    assert v.dv_capacity() == pytest.approx(expected)


def test_round_trip_dv_and_propellant():
    v = chaser()
    for dv in (1.0, 65.0, 300.0):
        assert v.dv_for(v.propellant_for(dv)) == pytest.approx(dv, rel=1e-12)


def test_burning_the_whole_tank_gives_exactly_dv_capacity():
    v = chaser()
    assert v.dv_for(v.prop_mass) == pytest.approx(v.dv_capacity())


def test_same_dv_costs_less_propellant_when_lighter():
    """Non-obvious but central to campaign planning: delta-v gets *cheaper* in
    kilograms as the mission burns down, which is why later targets are cheaper
    than they look."""
    v = chaser()
    assert v.propellant_for(50.0, start_mass=450.0) < v.propellant_for(50.0, start_mass=500.0)


def test_rejects_nonsense_vehicles():
    with pytest.raises(ValueError):
        Vehicle(name="x", dry_mass=0.0, prop_mass=10.0, isp=220.0)
    with pytest.raises(ValueError):
        Vehicle(name="x", dry_mass=10.0, prop_mass=10.0, isp=0.0)


def test_cannot_burn_more_than_you_weigh():
    with pytest.raises(ValueError):
        chaser().dv_for(600.0)


# ----------------------------------------------------------------- the ledger
def test_telescoping_one_vehicle_order_does_not_matter():
    """For a single vehicle at one Isp, propellant from the summed delta-v equals
    replaying the burns one at a time - the mass ratios telescope. This is why
    the ledger can sum |dv| freely *within* a vehicle. Pinned so nobody
    "optimises" the replay away and then trips over the split, where it fails."""
    led = FuelLedger()
    v = led.add_vehicle(chaser())
    for i, dv in enumerate([12.0, 3.5, 40.0, 0.25]):
        led.record(t=float(i), dv=dv, phase=PHASING, vehicle="chaser")

    assert led.propellant_by() == pytest.approx(v.propellant_for(led.total_dv()), rel=1e-12)


def test_partition_invariant_phases_and_vehicles_both_sum_to_total():
    """Per-phase and per-vehicle propellant must each partition the total
    exactly. If they do not, something is being counted twice."""
    led = FuelLedger()
    led.add_vehicle(chaser())
    led.add_vehicle(Vehicle(name="sat0", dry_mass=100.0, prop_mass=25.0, isp=220.0))
    led.record(0.0, 60.0, PHASING, "chaser")
    led.record(1.0, 5.0, TERMINAL, "chaser")
    led.record(2.0, 3.2, FORMATION, "sat0")
    led.record(3.0, 1.1, FORMATION, "chaser")

    total = led.propellant_by()
    by_phase = sum(led.propellant_by(phase=p) for p in led.phases())
    by_vehicle = sum(led.propellant_by(vehicle=n) for n in led.vehicles)
    assert by_phase == pytest.approx(total, rel=1e-12)
    assert by_vehicle == pytest.approx(total, rel=1e-12)


def test_replay_is_time_ordered_not_insertion_ordered():
    """Sources flush out of order (the constellation flushes per checkpoint while
    the mission records per burn), so replay must sort by time."""
    a, b = FuelLedger(), FuelLedger()
    a.add_vehicle(chaser())
    b.add_vehicle(chaser())
    a.record(0.0, 60.0, PHASING, "chaser")
    a.record(9.0, 5.0, TERMINAL, "chaser")
    b.record(9.0, 5.0, TERMINAL, "chaser")  # recorded first, happens later
    b.record(0.0, 60.0, PHASING, "chaser")
    for phase in (PHASING, TERMINAL):
        assert a.propellant_by(phase=phase) == pytest.approx(b.propellant_by(phase=phase))


def test_zero_burns_are_dropped_and_negative_rejected():
    led = FuelLedger()
    led.add_vehicle(chaser())
    led.record(0.0, 0.0, PHASING, "chaser")
    assert led.entries == []
    with pytest.raises(ValueError):
        led.record(0.0, -1.0, PHASING, "chaser")
    with pytest.raises(KeyError):
        led.record(0.0, 1.0, PHASING, "nobody")


def test_remaining_propellant_goes_negative_rather_than_clamping():
    """A budget that silently clamps is a budget that never binds."""
    led = FuelLedger()
    led.add_vehicle(chaser())
    led.record(0.0, 10_000.0, PHASING, "chaser")
    assert led.remaining_propellant("chaser") < 0
    assert not led.feasible()


# ------------------------------------------------------- the mass discontinuity
def test_split_conserves_mass():
    parent = chaser()
    led = FuelLedger()
    led.add_vehicle(parent)
    led.record(0.0, 65.0, PHASING, "chaser")
    remaining = led.remaining_mass("chaser")

    kids = split_vehicle(parent, remaining, [f"sat{i}" for i in range(4)], dry_mass_each=100.0)
    assert len(kids) == 4
    assert sum(k.wet_mass for k in kids) == pytest.approx(remaining)


def test_split_charges_pre_split_dv_at_the_heavier_mass():
    """The whole point of ordering entries: 65 m/s costs more propellant on a
    500 kg chaser than the same 65 m/s spread over four 125 kg satellites."""
    parent = Vehicle(name="chaser", dry_mass=400.0, prop_mass=100.0, isp=220.0)
    led = FuelLedger()
    led.add_vehicle(parent)
    led.record(0.0, 65.0, PHASING, "chaser")
    before = led.propellant_by(phase=PHASING)

    kids = split_vehicle(parent, led.remaining_mass("chaser"), ["s0", "s1", "s2", "s3"], 100.0)
    after = FuelLedger()
    for k in kids:
        after.add_vehicle(k)
        after.record(1.0, 65.0 / 4, FORMATION, k.name)

    assert before > after.propellant_by()


def test_split_rejects_impossible_division():
    with pytest.raises(ValueError, match="dry mass"):
        split_vehicle(chaser(), remaining_mass=300.0, names=["a", "b", "c", "d"],
                      dry_mass_each=100.0)


# ------------------------------------------------------------- the unit boundary
def test_record_kms_converts_at_the_boundary():
    led = FuelLedger()
    led.add_vehicle(chaser())
    led.record_kms(0.0, 0.065, PHASING, "chaser")  # km/s in
    assert led.total_dv() == pytest.approx(65.0)  # m/s stored


def test_table_reports_every_phase_and_vehicle():
    led = FuelLedger()
    led.add_vehicle(chaser())
    led.add_vehicle(Vehicle(name="sat0", dry_mass=100.0, prop_mass=25.0, isp=220.0))
    led.record(0.0, 60.0, PHASING, "chaser")
    led.record(2.0, 3.2, FORMATION, "sat0")
    out = led.table()
    assert PHASING in out and FORMATION in out
    assert "chaser" in out and "sat0" in out
    assert "TOTAL" in out
