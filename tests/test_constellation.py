import numpy as np

from orbdebris.capture import cw_accelerations
from orbdebris.constellation import (
    APEX_SAT,
    MOUTH_SATS,
    N_SATS,
    FunnelSim,
    approach_burn,
    centre_slots,
    formation_slots,
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
    open must (a) work and (b) cost delta-v continuously. Both are asserted:
    the cost is the honest price of the concept, not a bug."""
    sim = make_sim(enable_contact=False)
    sim.split()

    result = sim.run(duration=300.0, dt=0.005, record_every=200)

    assert result.formation_error[-1] < 0.5  # satellites reach and hold their slots
    assert result.mouth_radius[-1] > 4.0  # mouth open (~5 m as built, minus sag)
    assert result.formation_dv[-1] > 0.0  # it is not free
    assert np.all(np.diff(result.formation_dv) >= -1e-12)  # monotonic spend


def test_rim_sags_between_its_three_supports():
    """Real physics worth pinning: tension-only cords cannot push, so the rim
    arcs sag inward between the three satellites holding it - exactly like a
    trawl mouth between its spreaders. The satellites' own nodes stay out at
    the built radius; the free nodes between them do not."""
    sim = make_sim(enable_contact=False)
    sim.split()
    sim.run(duration=250.0, dt=0.005, record_every=5000)

    mouth = sim.pos[N_SATS + sim.net.mouth_nodes]
    radii = np.linalg.norm(mouth - mouth.mean(axis=0), axis=1)
    held = np.arange(0, len(radii), len(radii) // len(MOUTH_SATS))  # bonded nodes

    assert radii[held].min() > 4.8  # supported nodes hold the built radius
    assert radii.min() < radii[held].min()  # unsupported arcs sag inward
    assert radii.min() > 3.0  # but only mildly - the mouth stays usable


def test_mouth_satellites_must_divide_the_rim_evenly():
    """A rim whose sectors do not divide by 3 leaves one oversized unsupported
    arc, which sags badly. Fail loudly rather than quietly capture less."""
    import pytest

    net = build_funnel_net(n_rings=4, n_sectors=10, mouth_radius=5.0, length=12.0)
    cloud = make_pellet_cloud(n_pellets=5, seed=1)
    with pytest.raises(ValueError, match="divisible"):
        FunnelSim.from_hill_state(net, cloud, N_REF, np.array([0.0, -1.0, 0.0]), np.zeros(3))
