"""Mass-spring net model. Units: METRES (the capture world), not km.

The net is an n x n grid of point masses joined by links. The one physical
subtlety that matters: cords carry **tension only**. A rope pulled taut resists
with k*stretch; a rope pushed together just goes slack - it cannot transmit
compression. Modelling links as ordinary (bidirectional) springs would make the
net behave like a trampoline mesh and push itself off the debris instead of
wrapping it.

Link topology: structural links between grid neighbours carry the loads;
diagonal shear links resist the lattice collapsing sideways. The perimeter
structural links double as the drawstring - the capture sim shrinks their rest
lengths after first contact to close the net's mouth.

The net also carries a **membrane**: the fabric surface spanned between the
cords, represented by triangulating each grid quad. Cords alone are a sieve -
cm-scale pellets would sail through metre-scale holes - so contact is done
against the triangles (see ``closest_points_on_triangles``), which makes the
surface continuous, like a balloon skin. The membrane's own stretch stiffness
is not modelled separately; the cord lattice carries it (stated limitation).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Net:
    positions: np.ndarray  # [N, 3] node positions in the net's local frame [m]
    masses: np.ndarray  # [N] node masses [kg]
    links: np.ndarray  # [L, 2] node-index pairs
    rest_lengths: np.ndarray  # [L] unstretched link lengths [m]
    perimeter: np.ndarray  # [L] bool: links forming the outer ring (drawstring)
    corners: np.ndarray  # [4] node indices of the corners (bridle attach points)
    triangles: np.ndarray  # [T, 3] node-index triples: the membrane surface
    # Funnel nets only (build_funnel_net); None for the square net.
    mouth_nodes: np.ndarray | None = None  # node indices on the mouth ring
    apex_node: int | None = None  # node index at the funnel's tip

    @property
    def n_nodes(self) -> int:
        return len(self.masses)


def build_net(
    n_side: int = 7,
    spacing: float = 1.0,
    total_mass: float = 10.0,
    corner_mass: float = 1.0,
) -> Net:
    """Square n_side x n_side net in the local x-y plane, centred on the origin.

    corner_mass is *added* to the four corner nodes (weighted corners carry
    momentum outward during the throw, opening the net - how real capture nets
    deploy).
    """
    n = n_side
    idx = lambda i, j: i * n + j  # noqa: E731

    coords = (np.arange(n) - (n - 1) / 2) * spacing
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    positions = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n * n)])

    masses = np.full(n * n, total_mass / (n * n))
    corners = np.array([idx(0, 0), idx(0, n - 1), idx(n - 1, 0), idx(n - 1, n - 1)])
    masses[corners] += corner_mass

    links: list[tuple[int, int]] = []
    rests: list[float] = []
    perim: list[bool] = []
    triangles: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n):
            if j + 1 < n:  # structural, along +y of the grid
                links.append((idx(i, j), idx(i, j + 1)))
                rests.append(spacing)
                perim.append(i == 0 or i == n - 1)
            if i + 1 < n:  # structural, along +x of the grid
                links.append((idx(i, j), idx(i + 1, j)))
                rests.append(spacing)
                perim.append(j == 0 or j == n - 1)
            if i + 1 < n and j + 1 < n:  # shear diagonals
                links.append((idx(i, j), idx(i + 1, j + 1)))
                rests.append(spacing * np.sqrt(2))
                perim.append(False)
                links.append((idx(i, j + 1), idx(i + 1, j)))
                rests.append(spacing * np.sqrt(2))
                perim.append(False)
                # Membrane: split the quad into two triangles.
                a, b = idx(i, j), idx(i + 1, j)
                c, d = idx(i + 1, j + 1), idx(i, j + 1)
                triangles.append((a, b, c))
                triangles.append((a, c, d))

    return Net(
        positions=positions,
        masses=masses,
        links=np.array(links),
        rest_lengths=np.array(rests),
        perimeter=np.array(perim),
        corners=corners,
        triangles=np.array(triangles),
    )


def build_funnel_net(
    n_rings: int = 6,
    n_sectors: int = 12,
    mouth_radius: float = 5.0,
    length: float = 12.0,
    total_mass: float = 20.0,
    mouth_mass: float = 1.0,
) -> Net:
    """Conical funnel: a wide mouth ring tapering back to a single apex node.

    Local frame: the mouth ring lies in the x-y plane at z=0 and the funnel
    extends toward -z, so +z is the direction the mouth faces (the direction
    the constellation sweeps). Ring 0 is the mouth; the apex is one node.

    Topology mirrors the square net: circumferential links around each ring,
    longitudinal links between rings, shear diagonals, and a triangulated
    membrane. The mouth ring's circumferential links are the drawstring
    (``perimeter``), so cinching it closes the mouth exactly as the square
    net's perimeter does.

    mouth_mass is *added* to each mouth-ring node: a heavy rim resists the
    mouth being deformed by debris impacts and gives the holding satellites
    something substantial to pull against.
    """
    # Radius tapers linearly from mouth_radius to (almost) zero at the apex.
    ring_r = np.linspace(mouth_radius, 0.0, n_rings + 1)[:-1]  # drop the 0 ring
    ring_z = np.linspace(0.0, -length, n_rings + 1)[:-1]
    theta = np.arange(n_sectors) * 2 * np.pi / n_sectors

    positions = []
    for r, z in zip(ring_r, ring_z):
        for th in theta:
            positions.append([r * np.cos(th), r * np.sin(th), z])
    apex_node = len(positions)
    positions.append([0.0, 0.0, -length])
    positions = np.array(positions)

    masses = np.full(len(positions), total_mass / len(positions))
    mouth_nodes = np.arange(n_sectors)  # ring 0
    masses[mouth_nodes] += mouth_mass

    def idx(ring: int, sector: int) -> int:
        return ring * n_sectors + (sector % n_sectors)

    links: list[tuple[int, int]] = []
    rests: list[float] = []
    perim: list[bool] = []
    triangles: list[tuple[int, int, int]] = []

    def add(i: int, j: int, is_perimeter: bool = False) -> None:
        links.append((i, j))
        rests.append(float(np.linalg.norm(positions[i] - positions[j])))
        perim.append(is_perimeter)

    for ring in range(n_rings):
        for s in range(n_sectors):
            a = idx(ring, s)
            b = idx(ring, s + 1)
            add(a, b, is_perimeter=(ring == 0))  # circumferential; ring 0 = drawstring
            if ring + 1 < n_rings:
                c = idx(ring + 1, s)
                d = idx(ring + 1, s + 1)
                add(a, c)  # longitudinal
                add(a, d)  # shear
                triangles.append((a, b, d))
                triangles.append((a, d, c))
            else:
                # Last ring closes onto the apex as a fan of triangles.
                add(a, apex_node)
                triangles.append((a, b, apex_node))

    return Net(
        positions=positions,
        masses=masses,
        links=np.array(links),
        rest_lengths=np.array(rests),
        perimeter=np.array(perim),
        corners=mouth_nodes,  # satellites bond here (3 of them, 120 deg apart)
        triangles=np.array(triangles),
        mouth_nodes=mouth_nodes,
        apex_node=apex_node,
    )


def link_forces(
    pos: np.ndarray,
    vel: np.ndarray,
    links: np.ndarray,
    rest_lengths: np.ndarray,
    k: float,
    c: float,
) -> np.ndarray:
    """Tension-only spring-damper forces -> per-node force array [N, 3].

    For a taut link the force magnitude is k*stretch + c*separation_rate,
    clamped at zero so damping can never turn a cord into a strut. Slack links
    (stretch <= 0) transmit nothing.
    """
    i, j = links[:, 0], links[:, 1]
    d = pos[j] - pos[i]  # [L, 3]
    length = np.linalg.norm(d, axis=1)
    # Coincident nodes (length 0) are always slack, so the direction does not
    # matter - but 0/0 would poison the force array with nans. Guard it.
    u = d / np.maximum(length, 1e-12)[:, None]  # unit vectors i -> j

    stretch = length - rest_lengths
    rate = np.einsum("ij,ij->i", vel[j] - vel[i], u)  # separation rate along the link
    magnitude = np.where(stretch > 0.0, np.maximum(k * stretch + c * rate, 0.0), 0.0)

    f = magnitude[:, None] * u  # force on node i, toward j; node j gets the opposite
    forces = np.zeros_like(pos)
    np.add.at(forces, i, f)
    np.add.at(forces, j, -f)
    return forces


def closest_points_on_triangles(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest point on each triangle (a, b, c) [T, 3] to each point [P, 3].

    Returns (closest [P, T, 3], barycentric weights [P, T, 3]). The weights let
    a contact force applied at the closest point be distributed to the three
    membrane nodes that span the triangle. Vectorised version of the standard
    seven-region algorithm (Ericson, Real-Time Collision Detection, ch. 5).
    """
    p = points[:, None, :]  # [P, 1, 3]
    ab = (b - a)[None, :, :]
    ac = (c - a)[None, :, :]
    ap = p - a[None, :, :]

    d1 = (ab * ap).sum(axis=-1)  # [P, T] via broadcasting
    d2 = (ac * ap).sum(axis=-1)
    bp = p - b[None, :, :]
    d3 = (ab * bp).sum(axis=-1)
    d4 = (ac * bp).sum(axis=-1)
    cp = p - c[None, :, :]
    d5 = (ab * cp).sum(axis=-1)
    d6 = (ac * cp).sum(axis=-1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    eps = 1e-12

    shape = d1.shape
    bary = np.zeros(shape + (3,))
    assigned = np.zeros(shape, dtype=bool)

    def claim(mask, u_w, v_w, w_w):
        sel = mask & ~assigned
        bary[sel, 0] = u_w[sel] if isinstance(u_w, np.ndarray) else u_w
        bary[sel, 1] = v_w[sel] if isinstance(v_w, np.ndarray) else v_w
        bary[sel, 2] = w_w[sel] if isinstance(w_w, np.ndarray) else w_w
        assigned[:] = assigned | sel

    with np.errstate(divide="ignore", invalid="ignore"):
        claim((d1 <= 0) & (d2 <= 0), 1.0, 0.0, 0.0)  # vertex a
        claim((d3 >= 0) & (d4 <= d3), 0.0, 1.0, 0.0)  # vertex b
        t_ab = d1 / np.where(np.abs(d1 - d3) > eps, d1 - d3, eps)
        claim((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1.0 - t_ab, t_ab, 0.0)  # edge ab
        claim((d6 >= 0) & (d5 <= d6), 0.0, 0.0, 1.0)  # vertex c
        t_ac = d2 / np.where(np.abs(d2 - d6) > eps, d2 - d6, eps)
        claim((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1.0 - t_ac, 0.0, t_ac)  # edge ac
        denom_bc = (d4 - d3) + (d5 - d6)
        t_bc = (d4 - d3) / np.where(np.abs(denom_bc) > eps, denom_bc, eps)
        claim(
            (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), 0.0, 1.0 - t_bc, t_bc
        )  # edge bc
        denom = va + vb + vc
        v = vb / np.where(np.abs(denom) > eps, denom, eps)
        w = vc / np.where(np.abs(denom) > eps, denom, eps)
        claim(np.ones(shape, dtype=bool), 1.0 - v - w, v, w)  # face interior

    closest = (
        bary[:, :, 0:1] * a[None, :, :]
        + bary[:, :, 1:2] * b[None, :, :]
        + bary[:, :, 2:3] * c[None, :, :]
    )
    return closest, bary
