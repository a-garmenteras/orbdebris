import numpy as np

from orbdebris.net import (
    build_funnel_net,
    build_net,
    build_tetra_net,
    closest_points_on_triangles,
    link_forces,
    membrane_element_forces,
    membrane_rest_data,
)


def two_nodes(gap: float):
    """Two nodes on the x axis, one link with rest length 1."""
    pos = np.array([[0.0, 0.0, 0.0], [gap, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    links = np.array([[0, 1]])
    rest = np.array([1.0])
    return pos, vel, links, rest


def test_compressed_link_transmits_nothing():
    # Cords cannot push: a link shorter than its rest length must go slack.
    pos, vel, links, rest = two_nodes(gap=0.5)
    f = link_forces(pos, vel, links, rest, k=100.0, c=1.0)
    assert np.allclose(f, 0.0)


def test_taut_link_pulls_with_hookes_law_and_newtons_third():
    pos, vel, links, rest = two_nodes(gap=1.5)  # stretch = 0.5
    f = link_forces(pos, vel, links, rest, k=100.0, c=1.0)

    assert np.allclose(f[0], [100.0 * 0.5, 0.0, 0.0])  # pulled toward node 1
    assert np.allclose(f[0] + f[1], 0.0)  # action = -reaction


def test_damping_opposes_separation_but_cannot_push():
    pos, vel, links, rest = two_nodes(gap=1.5)
    vel[1, 0] = 2.0  # separating
    f_separating = link_forces(pos, vel, links, rest, k=100.0, c=10.0)
    assert f_separating[0, 0] > 100.0 * 0.5  # spring + damping

    vel[1, 0] = -100.0  # closing so fast damping would exceed the spring force
    f_closing = link_forces(pos, vel, links, rest, k=100.0, c=10.0)
    assert np.allclose(f_closing, 0.0)  # clamped: a cord never pushes


def test_build_net_topology():
    n = 5
    net = build_net(n_side=n, spacing=2.0, total_mass=10.0, corner_mass=1.0)

    assert net.n_nodes == n * n
    assert np.isclose(net.masses.sum(), 10.0 + 4 * 1.0)
    assert len(net.corners) == 4
    assert np.all(net.masses[net.corners] > net.masses.max() / 2)

    n_structural = 2 * n * (n - 1)
    n_shear = 2 * (n - 1) ** 2
    assert len(net.links) == n_structural + n_shear
    # Drawstring = the outer ring: 4 sides of (n-1) links each.
    assert net.perimeter.sum() == 4 * (n - 1)
    # Structural links have rest length = spacing; shear = spacing * sqrt(2).
    assert np.isclose(net.rest_lengths.min(), 2.0)
    assert np.isclose(net.rest_lengths.max(), 2.0 * np.sqrt(2))
    # Membrane: every grid quad is two triangles.
    assert len(net.triangles) == 2 * (n - 1) ** 2


def test_funnel_topology():
    rings, sectors = 5, 8
    net = build_funnel_net(
        n_rings=rings, n_sectors=sectors, mouth_radius=5.0, length=12.0,
        total_mass=20.0, mouth_mass=1.0,
    )

    assert net.n_nodes == rings * sectors + 1  # ring nodes + the apex
    assert net.apex_node == rings * sectors
    assert len(net.mouth_nodes) == sectors

    # The mouth is the wide end at z=0; the apex is the tip at -length.
    mouth = net.positions[net.mouth_nodes]
    assert np.allclose(mouth[:, 2], 0.0)
    assert np.allclose(np.linalg.norm(mouth[:, :2], axis=1), 5.0)
    assert np.isclose(net.positions[net.apex_node, 2], -12.0)

    # Radius must taper monotonically from mouth to apex.
    radii = [
        np.linalg.norm(net.positions[r * sectors : (r + 1) * sectors, :2], axis=1).mean()
        for r in range(rings)
    ]
    assert np.all(np.diff(radii) < 0)

    # The drawstring is exactly the mouth ring: one circumferential link per sector.
    assert net.perimeter.sum() == sectors
    drawstring_nodes = np.unique(net.links[net.perimeter].ravel())
    assert set(drawstring_nodes) == set(net.mouth_nodes)

    # A heavy rim: mouth nodes outweigh interior ones.
    assert net.masses[net.mouth_nodes].min() > net.masses[net.apex_node]


def test_funnel_membrane_is_closed_around_each_ring():
    # Every sector must be spanned, or pellets leak through the gap.
    net = build_funnel_net(n_rings=4, n_sectors=6, mouth_radius=3.0, length=6.0)
    tri_nodes = set(net.triangles.ravel().tolist())
    for node in range(net.n_nodes):
        assert node in tri_nodes  # no node is left out of the surface
    # The apex is closed by a fan of one triangle per sector.
    apex_tris = (net.triangles == net.apex_node).any(axis=1).sum()
    assert apex_tris == 6


def test_tetra_topology():
    """The tetrahedron's shape is exactly the tetrahedron of the 4 satellites:
    3 mouth corners 120 deg apart plus one apex."""
    r, length = 6.2, 16.0
    net = build_tetra_net(mouth_radius=r, length=length, subdiv=6)

    assert len(net.corners) == 3
    corners = net.positions[net.corners]
    assert np.allclose(corners[:, 2], 0.0)  # mouth lies in the z=0 plane
    assert np.allclose(np.linalg.norm(corners[:, :2], axis=1), r)
    assert np.allclose(corners.sum(axis=0), 0.0, atol=1e-9)  # 120 deg apart -> balanced
    assert np.allclose(net.positions[net.apex_node], [0.0, 0.0, -length])


def test_tetra_is_a_closed_surface_except_the_mouth():
    """Every interior edge must be shared by exactly two triangles; only the
    mouth boundary is open. A hole anywhere else would leak debris."""
    from collections import Counter

    net = build_tetra_net(subdiv=6)
    shared = Counter()
    for t in net.triangles:
        for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            shared[(min(u, v), max(u, v))] += 1

    counts = Counter(shared.values())
    assert set(counts) == {1, 2}  # boundary edges and interior edges only
    boundary = [e for e, c in shared.items() if c == 1]
    # The open boundary is precisely the mouth.
    mouth = set(net.mouth_nodes.tolist())
    assert all(u in mouth and v in mouth for u, v in boundary)


def test_tetra_mouth_edges_are_straight_between_held_corners():
    """The reason for the tetrahedron: the mouth's edges are straight lines
    between the three satellite-held corners, so there is no sag-between-
    supports like a circular rim has (measured 4.68 vs 5.27 on the cone)."""
    r = 6.2
    net = build_tetra_net(mouth_radius=r, length=16.0, subdiv=6)
    mouth = net.positions[net.mouth_nodes]

    # Every mouth node sits on one of the three straight edges, so its distance
    # from the axis is between the inradius (edge midpoints) and circumradius.
    dist = np.linalg.norm(mouth[:, :2], axis=1)
    assert dist.max() <= r + 1e-9
    assert dist.min() >= r / 2 - 1e-9  # inradius = R/2 for an equilateral triangle


def test_tetra_faces_are_shallower_than_the_cone_it_replaces():
    """Equal capture area, gentler impacts: the face inclination is set by the
    inradius (R/2), so an area-matched tetrahedron presents ~11 deg faces where
    the cone presented 14 deg - more glancing, which is the capture condition."""
    r, length = 6.2, 16.0
    tetra_angle = np.degrees(np.arctan((r / 2) / length))
    cone_angle = np.degrees(np.arctan(4.0 / length))  # the cone it replaces

    assert tetra_angle < cone_angle
    assert 10.0 < tetra_angle < 12.0

    # And it is area-matched to that cone's circular mouth.
    tri_area = 3 * np.sqrt(3) / 4 * r**2
    assert abs(tri_area - np.pi * 4.0**2) / (np.pi * 4.0**2) < 0.05


def _one_triangle():
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tris = np.array([[0, 1, 2]])
    dm_inv, area = membrane_rest_data(rest, tris)
    return rest, tris, dm_inv, area


def test_membrane_element_is_rotation_invariant():
    """A shell element must produce zero force under rigid rotation - only
    genuine deformation loads it. (Green strain, not linear strain.)"""
    rest, tris, dm_inv, area = _one_triangle()
    th = np.radians(40)
    rot = rest @ np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    ).T
    f = membrane_element_forces(rot, np.zeros((3, 3)), tris, dm_inv, area, 100.0, 0.3, 0.0)
    assert np.abs(f).max() < 1e-9


def test_membrane_element_resists_stretch_and_compression():
    """The whole point of shells over cords: the surface resists in-plane
    compression (pushes back out) as well as stretch (pulls back in). A
    tension-only cord lattice goes slack under compression - which is exactly
    why the cord-based funnel could not hold its cone shape."""
    rest, tris, dm_inv, area = _one_triangle()
    vel = np.zeros((3, 3))

    stretched = rest.copy()
    stretched[1, 0] = 1.5
    f = membrane_element_forces(stretched, vel, tris, dm_inv, area, 100.0, 0.3, 0.0)
    assert f[1, 0] < -1.0  # pulled back inward

    compressed = rest.copy()
    compressed[1, 0] = 0.6
    f = membrane_element_forces(compressed, vel, tris, dm_inv, area, 100.0, 0.3, 0.0)
    assert f[1, 0] > 1.0  # pushed back outward - impossible for a cord


def test_membrane_wrinkles_when_severely_folded():
    """Crumpled fabric goes (nearly) limp: a deeply folded triangle transmits
    only a faint force - the 5% floor standing in for bending stiffness, which
    lets folded cloth push itself open instead of locking folded. Without the
    gate, deploying from a fold stores absurd elastic energy and detonates."""
    rest, tris, dm_inv, area = _one_triangle()
    vel = np.zeros((3, 3))
    folded = rest * 0.05  # 5% scale: deep fold

    f_folded = membrane_element_forces(folded, vel, tris, dm_inv, area, 100.0, 0.3, 0.0)
    # Same strain state without gating would be enormous; compare against a
    # mildly compressed (ungated) element to show the fold is ~limp.
    mild = rest * 0.8
    f_mild = membrane_element_forces(mild, vel, tris, dm_inv, area, 100.0, 0.3, 0.0)

    assert np.abs(f_folded).max() < np.abs(f_mild).max()  # deep fold ~limp
    assert np.abs(f_folded).max() > 0.0  # but faintly pushes itself open


def test_membrane_element_conserves_momentum():
    rest, tris, dm_inv, area = _one_triangle()
    deformed = rest * np.array([1.3, 0.8, 1.0])  # arbitrary in-plane deformation
    f = membrane_element_forces(deformed, np.zeros((3, 3)), tris, dm_inv, area, 100.0, 0.3, 0.0)
    assert np.allclose(f.sum(axis=0), 0.0, atol=1e-9)  # internal forces only


def test_closest_point_on_triangle_regions():
    # One right triangle in the z=0 plane; probe every region class.
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    c = np.array([[0.0, 1.0, 0.0]])
    points = np.array(
        [
            [0.25, 0.25, 1.0],  # above the face interior
            [-1.0, -1.0, 0.5],  # beyond vertex a
            [0.5, -1.0, 0.0],  # beyond edge ab
            [1.0, 1.0, 0.0],  # beyond edge bc
        ]
    )

    closest, bary = closest_points_on_triangles(points, a, b, c)

    assert np.allclose(closest[0, 0], [0.25, 0.25, 0.0])
    assert np.allclose(bary[0, 0], [0.5, 0.25, 0.25])
    assert np.allclose(closest[1, 0], [0.0, 0.0, 0.0])
    assert np.allclose(bary[1, 0], [1.0, 0.0, 0.0])
    assert np.allclose(closest[2, 0], [0.5, 0.0, 0.0])
    assert np.allclose(bary[2, 0], [0.5, 0.5, 0.0])
    assert np.allclose(closest[3, 0], [0.5, 0.5, 0.0])
    assert np.allclose(bary[3, 0], [0.0, 0.5, 0.5])
    # Barycentric weights always sum to 1 (they distribute contact reactions).
    assert np.allclose(bary.sum(axis=-1), 1.0)
