import numpy as np

from orbdebris.net import build_net, closest_points_on_triangles, link_forces


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
