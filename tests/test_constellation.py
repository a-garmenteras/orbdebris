import numpy as np

from orbdebris.capture import cw_accelerations
import pytest

from orbdebris.constellation import (
    APEX_SAT,
    MOUTH_SATS,
    N_SATS,
    FunnelSim,
    approach_burn,
    centre_slots,
    formation_slots,
    funnel_demo,
)
from orbdebris.constants import GM_EARTH, R_EARTH
from orbdebris.debris import make_pellet_cloud
from orbdebris.net import build_funnel_net
from orbdebris.relative import cw_propagate, mean_motion

N_REF = mean_motion(R_EARTH + 500.0, GM_EARTH)
PERIOD = 2 * np.pi / N_REF


def make_sim(**params) -> FunnelSim:
    # n_sectors must be divisible by 3 so the mouth satellites sit 120 deg apart.
    net = build_funnel_net(n_rings=5, n_sectors=12, mouth_radius=5.0, length=12.0)
    cloud = make_pellet_cloud(n_pellets=12, sigma_pos=1.2, seed=17)
    params.setdefault("mouth_radius", 5.0)  # match the net built above
    params.setdefault("funnel_length", 12.0)
    return FunnelSim.from_hill_state(
        net,
        cloud,
        N_REF,
        chaser_rel_r_km=np.array([0.0, -1.0, 0.0]),  # 1 km BEHIND the cloud
        chaser_rel_v_km=np.zeros(3),
        **params,
    )


def test_shared_cw_engine_still_matches_the_analytic_stm():
    """The M3 anchor, re-asserted at the new shared-engine seam: a free body
    integrated with cw_accelerations must reproduce relative.cw_propagate."""
    r = np.array([[3.0, -20.0, 1.5]])
    v = np.array([[0.02, -0.01, 0.005]])
    pos, vel = r.copy(), v.copy()
    dt, duration = 0.01, 200.0
    for _ in range(int(duration / dt)):
        vel += cw_accelerations(pos, vel, N_REF) * dt
        pos += vel * dt

    r_exp, v_exp = cw_propagate(r[0], v[0], duration, N_REF)
    assert np.linalg.norm(pos[0] - r_exp) < 0.01
    assert np.linalg.norm(vel[0] - v_exp) < 1e-4


def test_catching_from_behind_is_a_purely_radial_burn():
    """The project's recurring lesson, pinned. To close on something 1 km
    AHEAD over half an orbit, you burn straight DOWN - not forward. Prograde
    thrust couples into radial motion and balloons you over the target."""
    rel_r = np.array([0.0, -1000.0, 0.0])  # 1 km behind
    dv = approach_burn(rel_r, np.zeros(3), 0.5 * PERIOD, N_REF)

    assert abs(dv[1]) < 1e-3  # no along-track component at all
    assert abs(dv[2]) < 1e-9  # stays in plane
    assert dv[0] < 0  # radially DOWNWARD: drop to a lower, faster orbit
    assert 0.2 < np.linalg.norm(dv) < 0.4  # ~0.28 m/s

    # And it works: the burn actually closes the gap.
    r_end, _ = cw_propagate(rel_r, dv, 0.5 * PERIOD, N_REF)
    assert np.linalg.norm(r_end) < 1.0  # arrives at the cloud centroid


def test_naive_prograde_burn_balloons_and_misses():
    """The trap the radial burn avoids - asserted so the contrast is on record."""
    rel_r = np.array([0.0, -1000.0, 0.0])
    r_end, _ = cw_propagate(rel_r, np.array([0.0, 2.0, 0.0]), 500.0, N_REF)

    assert r_end[0] > 500.0  # ballooned radially upward
    assert r_end[1] < -100.0  # and still hasn't closed the along-track gap


def test_formation_slots_geometry():
    slots = formation_slots(mouth_radius=5.0, length=12.0)
    assert slots.shape == (N_SATS, 3)

    mouth = slots[MOUTH_SATS]
    assert np.allclose(mouth[:, 2], 0.0)  # mouth ring is the sweep face
    assert np.allclose(np.linalg.norm(mouth[:, :2], axis=1), 5.0)
    assert np.allclose(mouth.sum(axis=0), 0.0, atol=1e-9)  # 120 deg apart -> balanced
    assert np.allclose(slots[APEX_SAT], [0.0, 0.0, -12.0])  # apex trails behind


def test_split_conserves_momentum():
    """The separation is internal, so momentum over everything it pushes
    (satellites AND the net unfurling with them) must be unchanged."""
    sim = make_sim(enable_contact=False)
    deployed = slice(0, N_SATS + sim.net.n_nodes)
    p_before = (sim.mass[deployed, None] * sim.vel[deployed]).sum(axis=0)

    sim.split()

    p_after = (sim.mass[deployed, None] * sim.vel[deployed]).sum(axis=0)
    assert np.linalg.norm(p_after - p_before) < 1e-9


def test_formation_slots_must_be_centred_on_the_satellite_centroid():
    """Three satellites at z=0 and one at z=-length average to z=-length/4.
    Steering to un-centred slots gives a fixed point that cannot be reached:
    the controller thrusts against itself forever at a fixed offset."""
    raw = formation_slots(mouth_radius=5.0, length=12.0)
    assert np.isclose(raw.mean(axis=0)[2], -3.0)  # the trap

    centred, offset = centre_slots(raw)
    assert np.allclose(centred.mean(axis=0), 0.0, atol=1e-12)  # now reachable
    assert np.allclose(offset, [0.0, 0.0, -3.0])
    # Relative geometry is preserved - only the origin moved.
    assert np.allclose(centred - centred[APEX_SAT], raw - raw[APEX_SAT])


def test_formation_holds_the_mouth_open_and_it_costs_fuel():
    """A rigid formation is NOT a natural CW motion, so holding the funnel
    open must (a) work and (b) cost delta-v continuously. After deployment the
    slots are trimmed to the membrane's settled shape - a controller that
    keeps demanding the blueprint radius leans on the structure forever."""
    sim = make_sim(enable_contact=False)
    sim.split()
    sim.run(duration=180.0, dt=0.005, record_every=200)
    sim.trim_slots()

    result = sim.run(duration=120.0, dt=0.005, record_every=200)

    assert result.formation_error[-1] < 0.3  # trimmed slots are reachable
    assert result.mouth_radius[-1] > 4.0  # mouth held open (~5 m as built)
    assert result.formation_dv[-1] > 0.0  # it is not free
    assert np.all(np.diff(result.formation_dv) >= -1e-12)  # monotonic spend


def test_membrane_holds_the_rim_round():
    """With a continuous membrane (shell elements) the rim stays *round*: the
    balloon skin transmits the three satellites' pull all the way around, so
    the deep sag-between-supports of a tension-only cord rim is gone. (The old
    cord-only funnel sagged from 5.0 to ~3.5 m between supports - that sag is
    exactly why it leaked debris.)"""
    sim = make_sim(enable_contact=False)
    sim.split()
    sim.run(duration=250.0, dt=0.005, record_every=5000)

    mouth = sim.pos[N_SATS + sim.net.mouth_nodes]
    radii = np.linalg.norm(mouth - mouth.mean(axis=0), axis=1)

    # Measured: bonded nodes ~4.68 (pulled slightly in), free arcs ~5.27
    # (bowed slightly out by the membrane) - round to within ~12%, versus the
    # cord-only rim collapsing to ~1.5 m between supports.
    assert radii.min() > 4.5  # near the built 5 m everywhere on the ring
    assert radii.min() > 0.85 * radii.max()  # round: no deep inter-support sag


def test_mouth_satellites_must_divide_the_rim_evenly():
    """A rim whose sectors do not divide by 3 leaves one oversized unsupported
    arc, which sags badly. Fail loudly rather than quietly capture less."""
    net = build_funnel_net(n_rings=4, n_sectors=10, mouth_radius=5.0, length=12.0)
    cloud = make_pellet_cloud(n_pellets=5, seed=1)
    with pytest.raises(ValueError, match="divisible"):
        FunnelSim.from_hill_state(net, cloud, N_REF, np.array([0.0, -1.0, 0.0]), np.zeros(3))


def test_apex_box_forces_are_internal():
    """The storage box is mounted on the apex node, so wall force on a pellet
    must have an equal-and-opposite reaction on the apex. A box without the
    reaction is a force from nowhere: the assembly self-accelerates and the
    thrusters burn ~40 m/s fighting the phantom (measured before the fix)."""
    sim, sweep_velocity, _ = funnel_demo()
    sim.deploy_open(np.zeros(3))
    apex_row = N_SATS + sim.net.apex_node

    # One pellet latched in the box, drifting outward; others parked far away.
    sim.pos[sim.pellets] = sim.pos[apex_row] + np.array([0.0, 0.0, 60.0])
    sim.vel[sim.pellets] = 0.0
    p0 = N_SATS + sim.net.n_nodes
    sim.pos[p0] = sim.pos[apex_row] + np.array([0.0, 0.0, 1.0])
    sim.vel[p0] = sim.vel[apex_row] + np.array([0.0, 0.0, 0.8])
    sim._in_box[0] = True
    sim.pos[p0] = sim.pos[apex_row] + np.array([0.0, 0.0, sim.box_radius + 0.5])  # escaping

    sim.enable_contact = False
    sim._phase = "SPLIT"  # no controller thrust
    sim._hold_target = None
    accel, _ = sim._accelerations(0.0, 0.0)

    # Total force minus the (state-proportional) CW pseudo-forces must vanish:
    # every modelled force, box included, is an internal action-reaction pair.
    from orbdebris.capture import cw_accelerations

    cw = cw_accelerations(sim.pos, sim.vel, sim.n)
    net_force = (sim.mass[:, None] * (accel - cw)).sum(axis=0)
    assert np.linalg.norm(net_force) < 1e-9


def test_apex_box_is_one_way():
    """The storage box lets debris in but not out: a pellet parked just inside
    the box with a small outward velocity must be pushed back, not escape."""
    sim, sweep_velocity, approach = funnel_demo()
    sim.deploy_open(sweep_velocity)
    apex_row = N_SATS + sim.net.apex_node
    apex = sim.pos[apex_row]

    # Put one pellet just inside the box moving outward; freeze the rest far off.
    sim.pos[sim.pellets] = apex + np.array([0.0, 0.0, 50.0])  # everyone else parked away
    sim.vel[sim.pellets] = 0.0
    p0 = N_SATS + sim.net.n_nodes  # first pellet row
    sim.pos[p0] = apex + np.array([0.0, 0.0, 1.0])  # 1 m out, inside box_radius=1.8
    sim.vel[p0] = sim.vel[apex_row] + np.array([0.0, 0.0, 0.6])  # drifting outward

    sim.enable_contact = False  # isolate the box constraint from membrane hits
    for _ in range(4000):
        accel, _ = sim._accelerations(0.0, 0.0015)
        sim.vel += accel * 0.0015
        sim.pos += sim.vel * 0.0015

    d = np.linalg.norm(sim.pos[p0] - sim.pos[apex_row])
    assert d < sim.box_radius + 0.3  # held near the box, did not escape


@pytest.mark.slow
def test_end_to_end_funnel_collects_debris_into_the_apex_box():
    """The canonical M4 funnel-to-storage scenario (constellation.funnel_demo,
    the same one the script runs): the constellation sweeps the cloud and the
    funnel channels a majority of the debris down its walls into the apex box,
    where it settles - the three satellites NEVER let go and the mouth stays
    open the whole time. Also guards the centred-slot regression (a broken
    invariant makes the controller thrust forever)."""
    sim, sweep_velocity, approach = funnel_demo()
    n_pellets = sim.cloud.n_pellets

    # The approach is the project's recurring lesson: catching from behind is a
    # purely radial burn, and the arrival sweep is purely radial too.
    assert abs(approach["burn"][1]) < 1e-3 and approach["burn"][0] < 0
    assert abs(sweep_velocity[1]) < 1e-3

    sim.deploy_open(sweep_velocity)
    n_bonds_start = len(sim._bonds)
    result = sim.run(duration=200.0, dt=0.0015, record_every=200)

    assert result.captured
    assert "secured" in result.events
    # A majority ends stored and settled in the apex box.
    assert result.stored_frac[-1] >= 0.6
    assert round(result.stored_frac[-1] * n_pellets) >= 0.6 * n_pellets
    # The satellites never released: bonds intact, mouth held open throughout.
    assert len(sim._bonds) == n_bonds_start
    assert result.mouth_radius[-1] > 2.5  # still open (built 4 m, minus sag)
    # Formation cost stays bounded (no controller runaway).
    assert result.formation_dv[-1] < 8.0
