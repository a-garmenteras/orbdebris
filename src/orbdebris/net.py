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


def build_tetra_net(
    mouth_radius: float = 6.2,
    length: float = 16.0,
    subdiv: int = 6,
    total_mass: float = 20.0,
    corner_mass: float = 1.0,
) -> Net:
    """Tetrahedral funnel: three flat triangular faces meeting at an apex, with
    an open triangular mouth. Same local frame as the cone (mouth in the x-y
    plane at z=0, apex at -length, so +z is the sweep direction).

    Why a tetrahedron rather than a cone: its shape is *exactly* the tetrahedron
    defined by the four satellite positions - 3 mouth corners + 1 apex - so
    commanding the formation IS commanding the funnel geometry. A cone's mouth
    is a circle with only three support points (it sags between them) and its
    surface is unsupported along its length, which is where the cone
    concertina'd - folding back on itself and reaching only 51% of its design
    length. A taut tetrahedron can sag inward but cannot fold back axially.

    Sizing note: a triangle inscribed at circumradius R has area (3*sqrt3/4)R^2,
    only ~41% of the circle's pi*R^2, so R is chosen larger for equal capture
    area. The *impact angle* is set by the faces' distance from the axis - the
    inradius R/2 - so an equal-area tetrahedron actually presents shallower
    (more glancing) faces than the cone it replaces.
    """
    r, depth = float(mouth_radius), float(length)
    theta = np.arange(3) * 2 * np.pi / 3
    mouth_corners = np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros(3)])
    apex_pos = np.array([0.0, 0.0, -depth])

    # Deduplicate nodes shared along the three corner->apex seams.
    positions: list[np.ndarray] = []
    index: dict[tuple, int] = {}

    def node(p: np.ndarray) -> int:
        key = tuple(np.round(p, 6))
        if key not in index:
            index[key] = len(positions)
            positions.append(np.asarray(p, dtype=float))
        return index[key]

    triangles: list[tuple[int, int, int]] = []
    for f in range(3):
        a, b, c = mouth_corners[f], mouth_corners[(f + 1) % 3], apex_pos
        # Barycentric lattice over the face: rows i from the mouth edge (a-b)
        # down to the apex, each row one node shorter.
        rows: list[list[int]] = []
        for i in range(subdiv + 1):
            row = []
            for j in range(subdiv - i + 1):
                w_c = i / subdiv
                rem = 1.0 - w_c
                w_b = rem * (j / (subdiv - i)) if subdiv - i > 0 else 0.0
                w_a = rem - w_b
                row.append(node(w_a * a + w_b * b + w_c * c))
            rows.append(row)
        for i in range(subdiv):
            upper, lower = rows[i], rows[i + 1]
            for j in range(len(lower)):
                triangles.append((upper[j], upper[j + 1], lower[j]))
                if j + 1 < len(lower):
                    triangles.append((upper[j + 1], lower[j + 1], lower[j]))

    positions = np.array(positions)
    tri = np.array(triangles)

    # Links = unique triangulation edges.
    edges = set()
    for t in tri:
        for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges.add((min(u, v), max(u, v)))
    links = np.array(sorted(edges))
    rests = np.linalg.norm(positions[links[:, 0]] - positions[links[:, 1]], axis=1)

    corners = np.array([node(p) for p in mouth_corners])
    apex_node = node(apex_pos)
    mouth_nodes = np.flatnonzero(np.isclose(positions[:, 2], 0.0))
    on_mouth = np.isin(links[:, 0], mouth_nodes) & np.isin(links[:, 1], mouth_nodes)

    masses = np.full(len(positions), total_mass / len(positions))
    masses[corners] += corner_mass

    return Net(
        positions=positions,
        masses=masses,
        links=links,
        rest_lengths=rests,
        perimeter=on_mouth,
        corners=corners,
        triangles=tri,
        mouth_nodes=mouth_nodes,
        apex_node=int(apex_node),
    )


def build_funnel_net(
    n_rings: int = 8,
    n_sectors: int = 12,
    mouth_radius: float = 4.0,
    length: float = 16.0,
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


def membrane_rest_data(
    positions: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-triangle rest quantities for shell elements: the inverse
    of the 2D rest edge matrix (Dm_inv) and the rest area. Call once at build."""
    a = positions[triangles[:, 0]]
    b = positions[triangles[:, 1]]
    c = positions[triangles[:, 2]]
    e1, e2 = b - a, c - a
    # 2D basis in each rest triangle's own plane.
    x_hat = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    normal = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(normal, axis=1)
    z_hat = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    y_hat = np.cross(z_hat, x_hat)
    # Rest edges in local 2D coords -> Dm (2x2 per triangle), then invert.
    dm = np.empty((len(triangles), 2, 2))
    dm[:, 0, 0] = np.einsum("ij,ij->i", e1, x_hat)
    dm[:, 1, 0] = np.einsum("ij,ij->i", e1, y_hat)
    dm[:, 0, 1] = np.einsum("ij,ij->i", e2, x_hat)
    dm[:, 1, 1] = np.einsum("ij,ij->i", e2, y_hat)
    return np.linalg.inv(dm), area


def membrane_element_forces(
    pos: np.ndarray,
    vel: np.ndarray,
    triangles: np.ndarray,
    dm_inv: np.ndarray,
    area: np.ndarray,
    k: float,
    poisson: float,
    damping: float,
) -> np.ndarray:
    """Continuous-membrane (shell) forces from constant-strain triangles, using
    a St-Venant-Kirchhoff material -> per-node force array [N, 3].

    Unlike 1D cords, a shell element resists in-plane *stretch, shear AND
    compression* of the surface itself - it is what makes the fabric a
    continuous membrane (a balloon skin) rather than a slack net. Because it
    uses the rotation-invariant Green strain E = 1/2 (F^T F - I), a rigid
    rotation produces zero force; only genuine deformation does.

    One correction to raw St-Venant-Kirchhoff: heavily *folded* fabric does not
    store huge elastic energy - it wrinkles, losing compressive stiffness. So
    each triangle's stress is gated by its area ratio: below ~70% of rest area
    the element fades out (wrinkled, limp), recovering full stiffness as it
    unfolds. Without this, deploying from a folded state detonates.

    k folds the sheet's Young's modulus and thickness into one areal stiffness.
    """
    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ds = np.stack([pos[i1] - pos[i0], pos[i2] - pos[i0]], axis=-1)  # [T,3,2]
    f_grad = ds @ dm_inv  # deformation gradient F [T,3,2]

    green = 0.5 * (np.einsum("tki,tkj->tij", f_grad, f_grad) - np.eye(2))  # E [T,2,2]
    mu = k / (2 * (1 + poisson))
    lam = k * poisson / (1 - poisson**2)
    tr = green[:, 0, 0] + green[:, 1, 1]
    stress = 2 * mu * green + lam * tr[:, None, None] * np.eye(2)  # 2nd PK S [T,2,2]

    piola = f_grad @ stress  # first PK P = F S [T,3,2]

    # Wrinkle gating: fade stiffness for *severely* folded triangles (area
    # < 50% of rest, mostly gone by 20%) - crumpled fabric goes limp instead
    # of storing elastic energy, while mild compression keeps full stiffness
    # so the cone still holds its shape. A small floor (5%) stands in for the
    # fabric's bending stiffness: folded cloth still faintly pushes itself
    # open, so deployment progressively unfolds instead of locking into a
    # zero-force folded state.
    cur_normal = np.cross(pos[i1] - pos[i0], pos[i2] - pos[i0])
    cur_area = 0.5 * np.linalg.norm(cur_normal, axis=1)
    wrinkle = 0.05 + 0.95 * np.clip(
        (cur_area / np.maximum(area, 1e-12) - 0.2) / 0.3, 0.0, 1.0
    )
    piola *= wrinkle[:, None, None]

    # Nodal forces: H = -area * P Dm_inv^T; columns act on nodes 1,2; node 0 = -sum.
    h = -area[:, None, None] * (piola @ np.transpose(dm_inv, (0, 2, 1)))  # [T,3,2]
    f1, f2 = h[:, :, 0], h[:, :, 1]
    f0 = -(f1 + f2)

    forces = np.zeros_like(pos)
    np.add.at(forces, i0, f0)
    np.add.at(forces, i1, f1)
    np.add.at(forces, i2, f2)

    if damping > 0.0:  # light viscous damping toward each triangle's mean velocity
        for idx in (i0, i1, i2):
            v_mean = (vel[i0] + vel[i1] + vel[i2]) / 3.0
            np.add.at(forces, idx, -damping * (vel[idx] - v_mean))
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
