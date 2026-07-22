import numpy as np

from orbdebris.net import (
    build_funnel_net,
    build_net,
    closest_points_on_triangles,
    link_forces,
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
