import numpy as np
import pytest

from orbdebris.capture import CHASER, KM_TO_M, CaptureSim, demo_scenario, membrane_contact_forces
from orbdebris.constants import GM_EARTH, R_EARTH
from orbdebris.debris import PelletCloud, make_pellet_cloud
from orbdebris.net import build_net
from orbdebris.relative import cw_propagate, mean_motion

N_REF = mean_motion(R_EARTH + 500.0, GM_EARTH)  # reference orbit, rad/s


def make_sim(**params) -> CaptureSim:
    net = build_net(n_side=5, spacing=1.0, total_mass=8.0, corner_mass=0.8)
    cloud = make_pellet_cloud(n_pellets=12, sigma_pos=1.0, seed=3)
    return CaptureSim.from_hill_state(
        net,
        cloud,
        N_REF,
        chaser_rel_r_km=np.array([0.0, 0.03, 0.0]),  # 30 m ahead of the cloud
        chaser_rel_v_km=np.zeros(3),
        **params,
    )


def test_km_to_m_boundary():
    sim = make_sim()
    assert np.allclose(sim.pos[CHASER], [0.0, 30.0, 0.0])  # metres inside
    assert KM_TO_M == 1000.0


def test_cloud_centroid_starts_at_the_rendezvous_target():
    # The mission-level "debris position" is the cloud centroid: at the Hill
    # origin, at rest, by construction.
    sim = make_sim()
    assert np.linalg.norm(sim.cloud_centroid()) < 1e-9


def test_integrator_matches_cw_stm_for_a_free_body():
    # The anchor test, project style: with no bridle (eject never called) and
    # contact off, the chaser is a free body and the capture integrator must
    # reproduce the analytic CW STM.
    sim = make_sim(enable_contact=False)
    r0 = sim.pos[CHASER].copy()
    v0 = np.array([0.05, -0.03, 0.02])  # some relative motion [m/s]
    sim.vel[CHASER] = v0.copy()

    duration, dt = 200.0, 0.01
    sim.run(duration, dt=dt, record_every=10_000)

    r_expected, v_expected = cw_propagate(r0, v0, duration, N_REF)  # unit-agnostic
    assert np.linalg.norm(sim.pos[CHASER] - r_expected) < 0.01  # metres
    assert np.linalg.norm(sim.vel[CHASER] - v_expected) < 1e-4


def test_eject_conserves_momentum():
    sim = make_sim(enable_contact=False)
    sim.eject(flight_time=10.0)
    total_p = (sim.mass[:, None] * sim.vel).sum(axis=0)
    assert np.linalg.norm(total_p) < 1e-9  # started at rest; throw + recoil cancel


def test_ejected_centroid_lands_on_the_aim_point():
    # CW is linear, so internal (tension-only) spring forces cannot move the
    # net's centroid: it must follow the CW trajectory the targeting solve
    # aimed, despite corners flying outward and links snapping taut en route.
    sim = make_sim(enable_contact=False)
    flight_time = 10.0
    info = sim.eject(flight_time=flight_time, aim_past=1.0)
    aim_target = np.array([0.0, -1.0, 0.0])  # cloud centroid + 1 m past, along -y

    sim.run(flight_time, dt=0.002, record_every=10_000)

    node_pos = sim.pos[sim.nodes]
    centroid = (sim.net.masses[:, None] * node_pos).sum(axis=0) / sim.net.masses.sum()
    assert np.linalg.norm(centroid - aim_target) < 0.05  # metres
    assert info["cw_aim_off"] > 0.0  # straight-line aiming would have missed


def membrane_test_sim(enable_contact: bool) -> CaptureSim:
    """A flat 3x3 net sheet in the x-y plane and one pellet flying at the
    interior of a mesh quad - a spot with no node and no cord anywhere near."""
    net = build_net(n_side=3, spacing=1.0, total_mass=2.0, corner_mass=0.2)
    cloud = PelletCloud(
        radii=np.array([0.05]),
        masses=np.array([0.02]),
        positions=np.array([[0.5, 0.25, 0.5]]),  # over a triangle face interior
        velocities=np.array([[0.0, 0.0, -1.0]]),  # flying at the sheet
    )
    sim = CaptureSim.from_hill_state(
        net, cloud, N_REF, np.array([0.0, 0.2, 0.0]), np.zeros(3),
        enable_contact=enable_contact,
    )
    sim.pos[sim.nodes] = net.positions  # unfold the sheet flat at the origin
    return sim


def test_membrane_stops_a_pellet_a_bare_mesh_would_miss():
    # With membrane contact, the pellet must stay on its side of the sheet and
    # rebound; with it off, it sails through a hole. This contrast is the
    # membrane's entire reason for existing: cm pellets vs m-scale mesh holes.
    sim = membrane_test_sim(enable_contact=True)
    sim.run(1.5, dt=0.001, record_every=1500)
    rel_z = sim.pos[sim.pellets][0, 2] - sim.pos[sim.nodes][:, 2].mean()
    assert rel_z > 0.0  # still on the near side
    assert sim.vel[sim.pellets][0, 2] > 0.0  # bounced back

    ghost = membrane_test_sim(enable_contact=False)
    ghost.run(1.5, dt=0.001, record_every=1500)
    rel_z_ghost = ghost.pos[ghost.pellets][0, 2] - ghost.pos[ghost.nodes][:, 2].mean()
    assert rel_z_ghost < 0.0  # passed straight through the hole


@pytest.mark.slow
def test_end_to_end_capture_and_tow():
    """The canonical M3 scenario (capture.demo_scenario - same one the script
    runs), truncated to the capture verdict plus a short tow. ~1 min of wall
    time; deselect with -m "not slow" for quick iterations.

    Scaled-down variants proved misleading: a shallower 5x5 pocket leaks
    pellets through self-intersecting folds (we model no self-collision), so
    the honest regression test is the geometry that actually works.
    """
    sim, eject_kwargs = demo_scenario()
    cloud = sim.cloud
    sim.eject(**eject_kwargs)

    result = sim.run(duration=60.0, dt=0.001, record_every=250)

    assert result.captured
    assert result.events["drawstring_start"] > result.events["first_contact"] - 1e-9
    assert result.retained_frac[-1] >= 0.6  # majority still bagged at t=60

    w = cloud.masses / cloud.masses.sum()
    centroids = (w[None, :, None] * result.pos[:, result.pellets]).sum(axis=1)
    k0 = int(np.searchsorted(result.t, result.events["tow_start"]))
    tow_disp = np.linalg.norm(centroids[-1] - centroids[k0])
    assert tow_disp > 2.0  # metres: the bag demonstrably follows the chaser


def _fire_pellet_at_flat_membrane(v_in, zeta, tangent_zeta, omega=150.0):
    """Integrate one pellet bouncing off a fixed flat membrane (z=0 plane).
    Returns the outgoing velocity. Nodes are held fixed (infinite mass)."""
    nodes = np.array([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]], float)
    node_vel = np.zeros((4, 3))
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    p = np.array([[0.0, 0.0, 0.06]])
    v = np.array([v_in], float)
    m = np.array([0.05])
    r = np.array([0.05])
    dt = 0.0005
    for _ in range(8000):
        forces = np.zeros((5, 3))
        membrane_contact_forces(
            forces, nodes, node_vel, np.arange(4), tris, p, v, np.array([4]), r, m,
            omega, zeta, tangent_zeta,
        )
        v += (forces[4] / m[0]) * dt
        p += v * dt
        if p[0, 2] > 0.3 and v[0, 2] > 0:  # bounced clear of the membrane
            break
    return v[0]


def test_membrane_absorbs_normal_impact_but_lets_glancing_debris_slide():
    """The funnel-to-storage capture relies on the membrane being an energy
    absorber: a pellet's velocity component *normal* to the fabric is dissipated
    (no bounce), while its *tangential* (glancing) component is largely
    preserved so it slides on down the funnel toward the apex. With high normal
    damping and low tangential damping, that asymmetry must hold."""
    # 45-degree hit: equal normal and tangential speed coming in.
    v_out = _fire_pellet_at_flat_membrane([1.0, 0.0, -1.0], zeta=0.95, tangent_zeta=0.05)
    tangential, normal_rebound = v_out[0], v_out[2]

    assert normal_rebound < 0.2  # normal impact absorbed - barely rebounds
    assert tangential > 0.5  # most of the glancing slide survives
    assert tangential > 5 * normal_rebound  # strongly asymmetric: slide >> bounce

    # The old grippy tune (elastic normal, heavy tangential drag) does the
    # opposite - it would bat debris back out rather than funnel it.
    v_old = _fire_pellet_at_flat_membrane([1.0, 0.0, -1.0], zeta=0.3, tangent_zeta=0.5)
    assert v_old[2] > normal_rebound  # bounces more
    assert v_old[0] < tangential  # slides less


def test_membrane_contact_conserves_momentum():
    # Contact forces are internal (pellet vs barycentric node reaction), so
    # the bounce must not create momentum.
    sim = membrane_test_sim(enable_contact=True)
    p_before = (sim.mass[:, None] * sim.vel).sum(axis=0)
    sim.run(1.5, dt=0.001, record_every=1500)
    p_after = (sim.mass[:, None] * sim.vel).sum(axis=0)
    # CW pseudo-forces do slowly change frame-momentum, but over 1.5 s at
    # n ~ 1e-3 rad/s that contribution is ~1e-3 of the contact impulse.
    assert np.linalg.norm(p_after - p_before) < 1e-3
