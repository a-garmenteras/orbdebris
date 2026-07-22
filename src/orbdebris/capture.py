"""Net-capture simulation in the Hill frame. UNITS: METRES inside this module.

Everything else in this project speaks km; contact and cloth physics is far
more readable in SI (spring constants in N/m, gaps in metres), so this module
converts at its boundary (``CaptureSim.from_hill_state``) and never leaks km.

Frame: the Hill frame of the debris cloud's circular reference orbit (mean
motion ``n``). The mission-level "debris position" is the cloud *centroid*;
here the cloud resolves into individual pellets (debris.py), each a dynamic
body. Every body - chaser, pellets, net nodes - feels the linear CW
pseudo-accelerations ``(3n²x + 2nẏ, −2nẋ, −n²z)`` plus model forces:

* net cords and the chaser->corner bridle: tension-only spring-dampers (net.py)
* **membrane contact**: pellets collide with the fabric surface spanned
  between the cords - sphere-vs-triangle penalty contact against the net's
  triangulation, reaction distributed to the triangle's nodes by barycentric
  weights. Cords alone are a sieve to cm-scale pellets; the membrane makes the
  net a continuous surface (balloon skin), and it is two-sided by construction.
* a perimeter "drawstring": after first contact, perimeter rest lengths shrink
  over a few seconds, closing the mouth (how real capture nets work)
* continuous chaser thrust during TOW (a tug jerk would snap the tether)

Contact stiffness is derived per pellet from its mass: ``k_i = m_i * omega²``
with one chosen contact frequency omega and damping ratio zeta. A fixed k
would give a 2 g pellet a ~kHz contact oscillation - far outside the stable
timestep - while scaling k with mass gives every pellet the *same* contact
timescale, safely inside it.

Integrator: fixed-step semi-implicit (symplectic) Euler - update velocity from
current forces, then position from the *new* velocity. The standard cloth-sim
integrator: explicit Euler pumps energy into oscillatory systems, while the
semi-implicit variant stays bounded, buying stiff-spring stability for a
one-line reordering.
"""

from dataclasses import dataclass, field

import numpy as np

from orbdebris.debris import PelletCloud
from orbdebris.net import Net, closest_points_on_triangles, link_forces
from orbdebris.relative import two_impulse_transfer

KM_TO_M = 1000.0

# Body array layout: row 0 = chaser, rows 1..1+N = net nodes, then pellets.
CHASER, NODES0 = 0, 1

EJECT, FLY, WRAP, TOW = "EJECT", "FLY", "WRAP", "TOW"


# --------------------------------------------------------------------------
# Shared force engine. These are the pieces that are identical for any
# Hill-frame net simulation, so constellation.py (the 4-satellite funnel)
# uses them verbatim rather than duplicating the physics.
# --------------------------------------------------------------------------


def cw_accelerations(pos: np.ndarray, vel: np.ndarray, n: float) -> np.ndarray:
    """Linear Clohessy-Wiltshire pseudo-accelerations for every body [B, 3].

    (3n²x + 2nẏ, −2nẋ, −n²z) - linear in the state, hence fully vectorised
    across bodies of any kind: satellites, net nodes and pellets alike.
    """
    a = np.zeros_like(pos)
    a[:, 0] = 3 * n**2 * pos[:, 0] + 2 * n * vel[:, 1]
    a[:, 1] = -2 * n * vel[:, 0]
    a[:, 2] = -(n**2) * pos[:, 2]
    return a


def membrane_contact_forces(
    forces: np.ndarray,
    node_pos: np.ndarray,
    node_vel: np.ndarray,
    node_rows: np.ndarray,
    triangles: np.ndarray,
    pellet_pos: np.ndarray,
    pellet_vel: np.ndarray,
    pellet_rows: np.ndarray,
    pellet_radii: np.ndarray,
    pellet_masses: np.ndarray,
    omega: float,
    zeta: float,
    tangent_zeta: float,
) -> int:
    """Add pellet-vs-membrane penalty contact into ``forces`` in place.

    Contact is against the fabric *surface* (the triangulation), not the cords:
    a cord lattice is a sieve to cm-scale pellets. The reaction on the
    membrane is split barycentrically over the triangle's three nodes, so the
    pair of forces is internal and momentum is conserved.

    Stiffness is per pellet, ``k_i = m_i * omega²``: one shared contact
    timescale for every pellet regardless of mass, which keeps a 2 g pellet
    inside the integrator's stable timestep alongside a 1.4 kg one.

    Returns the number of pellets in contact.
    """
    a = node_pos[triangles[:, 0]]
    b = node_pos[triangles[:, 1]]
    c = node_pos[triangles[:, 2]]
    closest, bary = closest_points_on_triangles(pellet_pos, a, b, c)

    diff = pellet_pos[:, None, :] - closest  # [P, T, 3]
    dist = np.linalg.norm(diff, axis=-1)
    pen = pellet_radii[:, None] - dist
    pi, ti = np.nonzero(pen > 0.0)
    if not len(pi):
        return 0

    n_hat = diff[pi, ti] / np.maximum(dist[pi, ti], 1e-9)[:, None]
    w_bary = bary[pi, ti]  # [K, 3]
    v_mem = (
        w_bary[:, 0:1] * node_vel[triangles[ti, 0]]
        + w_bary[:, 1:2] * node_vel[triangles[ti, 1]]
        + w_bary[:, 2:3] * node_vel[triangles[ti, 2]]
    )
    v_rel = pellet_vel[pi] - v_mem
    v_n = np.einsum("ij,ij->i", v_rel, n_hat)
    m_p = pellet_masses[pi]
    k_i = m_p * omega**2
    c_i = 2.0 * zeta * m_p * omega
    f_n = np.maximum(k_i * pen[pi, ti] - c_i * v_n, 0.0)
    v_t = v_rel - v_n[:, None] * n_hat
    c_t = 2.0 * tangent_zeta * m_p * omega
    f_vec = f_n[:, None] * n_hat - c_t[:, None] * v_t

    np.add.at(forces, pellet_rows[pi], f_vec)
    for k in range(3):
        np.add.at(forces, node_rows[triangles[ti, k]], -w_bary[:, k : k + 1] * f_vec)
    return int(np.unique(pi).size)


@dataclass
class CaptureResult:
    t: np.ndarray  # [S]
    pos: np.ndarray  # [S, B, 3] all body positions [m]
    tether_tension: np.ndarray  # [S] total bridle force on the chaser [N]
    contact_count: np.ndarray  # [S] pellets touching the membrane
    mouth_radius: np.ndarray  # [S] perimeter-node ring radius [m]
    retained_frac: np.ndarray  # [S] fraction of pellets inside the bag
    events: dict  # phase name -> time [s]
    captured: bool
    net: Net  # topology, for drawing
    pellet_radii: np.ndarray  # [P]
    n_nodes: int

    @property
    def nodes(self) -> slice:
        return slice(NODES0, NODES0 + self.n_nodes)

    @property
    def pellets(self) -> slice:
        return slice(NODES0 + self.n_nodes, None)


def demo_scenario(n_pellets: int = 30, seed: int = 11) -> tuple["CaptureSim", dict]:
    """The canonical, tuned M3 capture scenario - single source of truth for
    both scripts/run_capture.py and the end-to-end test, so they cannot drift.

    Returns (sim, eject_kwargs); call sim.eject(**eject_kwargs) then run.
    Tuning notes (each learned from a failure): slow ~1.2 m/s approach so the
    membrane herds instead of batting pellets away; inelastic contact (high
    damping ratios); drawstring triggered by mouth-passes-centroid geometry,
    not first contact; a soft heavily-damped bridle so the arrest cannot fling
    pellets back out; mouth cinched to 5% so cm pellets cannot slip the gap.
    """
    from orbdebris.constants import GM_EARTH, R_EARTH
    from orbdebris.debris import make_pellet_cloud
    from orbdebris.net import build_net
    from orbdebris.relative import mean_motion

    net = build_net(n_side=7, spacing=1.0, total_mass=10.0, corner_mass=1.0)
    cloud = make_pellet_cloud(n_pellets=n_pellets, sigma_pos=1.2, seed=seed)
    sim = CaptureSim.from_hill_state(
        net,
        cloud,
        mean_motion(R_EARTH + 500.0, GM_EARTH),
        chaser_rel_r_km=np.array([0.0, 0.03, 0.0]),  # 30 m ahead
        chaser_rel_v_km=np.zeros(3),
        contact_omega=150.0,
        contact_zeta=0.8,
        tangent_zeta=0.5,
        k_bridle=10.0,
        c_bridle=25.0,
        drawstring_close_time=2.5,
        drawstring_min_frac=0.05,
        bridle_slack=5.0,
        bag_margin=1.5,
    )
    return sim, {"flight_time": 25.0, "spread_speed": 0.8}


@dataclass
class CaptureSim:
    net: Net
    cloud: PelletCloud
    n: float  # reference-orbit mean motion [rad/s]
    chaser_mass: float = 500.0
    # Net cord stiffness/damping [N/m, N.s/m], bridle likewise (softer, longer).
    k_link: float = 400.0
    c_link: float = 4.0
    k_bridle: float = 40.0
    c_bridle: float = 8.0
    bridle_slack: float = 15.0  # bridle rest length = throw distance + this [m]
    # Membrane contact: per-pellet k_i = m_i*omega^2 (see module docstring).
    contact_omega: float = 60.0  # [rad/s] shared contact timescale
    contact_zeta: float = 0.3  # normal damping ratio
    tangent_zeta: float = 0.2  # tangential damping ratio (poor man's friction)
    enable_contact: bool = True
    # Drawstring: perimeter rest lengths shrink to min_frac over close_time,
    # triggered once the mouth ring has swept trigger_past metres beyond the
    # cloud centroid (deep enough that cinching doesn't sweep through pellets
    # still sitting near the rim of a shallow pocket).
    drawstring_close_time: float = 6.0
    drawstring_min_frac: float = 0.15
    drawstring_trigger_past: float = 0.0
    # Capture criterion (sustained).
    capture_retain_frac: float = 0.8  # fraction of pellets inside the bag
    capture_mouth_frac: float = 0.35  # mouth radius < frac * open mouth radius
    capture_sustain: float = 3.0
    bag_margin: float = 0.5  # "inside" = within bag radius + this [m]
    # Tow.
    tow_accel: float = 0.02  # [m/s^2]
    tow_delay: float = 2.0  # settle time between CAPTURED and thrust-on [s]

    # --- internal state (set up in from_hill_state / eject) -----------------
    pos: np.ndarray = field(default=None, repr=False)
    vel: np.ndarray = field(default=None, repr=False)
    mass: np.ndarray = field(default=None, repr=False)
    _rest: np.ndarray = field(default=None, repr=False)
    _rest0: np.ndarray = field(default=None, repr=False)
    _bridle_links: np.ndarray = field(default=None, repr=False)
    _bridle_rest: np.ndarray = field(default=None, repr=False)
    _boundary_nodes: np.ndarray = field(default=None, repr=False)
    _mouth_open_radius: float = None
    _phase: str = EJECT
    _events: dict = field(default_factory=dict)
    _capture_ok_since: float = None
    _tow_dir: np.ndarray = field(default=None, repr=False)
    _throw_dir: np.ndarray = field(default=None, repr=False)
    _captured: bool = False

    @property
    def nodes(self) -> slice:
        return slice(NODES0, NODES0 + self.net.n_nodes)

    @property
    def pellets(self) -> slice:
        return slice(NODES0 + self.net.n_nodes, None)

    @classmethod
    def from_hill_state(
        cls,
        net: Net,
        cloud: PelletCloud,
        n: float,
        chaser_rel_r_km: np.ndarray,
        chaser_rel_v_km: np.ndarray,
        **params,
    ) -> "CaptureSim":
        """Boundary from the mission world: the chaser's Hill state relative to
        the debris (= cloud centroid) in km / km/s, as produced by
        relative.relative_state. Everything beyond this line is metres."""
        sim = cls(net=net, cloud=cloud, n=n, **params)
        b = NODES0 + net.n_nodes + cloud.n_pellets
        sim.pos = np.zeros((b, 3))
        sim.vel = np.zeros((b, 3))
        sim.pos[CHASER] = np.asarray(chaser_rel_r_km, float) * KM_TO_M
        sim.vel[CHASER] = np.asarray(chaser_rel_v_km, float) * KM_TO_M
        sim.pos[sim.pellets] = cloud.positions
        sim.vel[sim.pellets] = cloud.velocities
        sim.mass = np.empty(b)
        sim.mass[CHASER] = sim.chaser_mass
        sim.mass[sim.nodes] = net.masses
        sim.mass[sim.pellets] = cloud.masses
        sim._rest0 = net.rest_lengths.copy()
        sim._rest = net.rest_lengths.copy()
        # Boundary (mouth) nodes = every node touched by a perimeter link;
        # the fully-open mouth radius comes from the as-built geometry.
        sim._boundary_nodes = np.unique(net.links[net.perimeter].ravel())
        ring = net.positions[sim._boundary_nodes]
        sim._mouth_open_radius = float(
            np.linalg.norm(ring - ring.mean(axis=0), axis=1).mean()
        )
        return sim

    def cloud_centroid(self) -> np.ndarray:
        w = self.cloud.masses / self.cloud.masses.sum()
        return (w[:, None] * self.pos[self.pellets]).sum(axis=0)

    # ------------------------------------------------------------------ eject
    def eject(
        self,
        flight_time: float,
        spread_speed: float = 1.5,
        fold_scale: float = 0.15,
        aim_past: float = 1.0,
    ) -> dict:
        """Throw the net: aim the centroid at the cloud centroid with the CW
        targeting solve (the same math that aims our rendezvous burns - CW is
        linear, so it is unit-agnostic and works in metres), fold the net just
        ahead of the chaser, and give the corners divergent velocity so it
        opens in flight. Returns aiming diagnostics."""
        target_point = self.cloud_centroid()
        throw = target_point - self.pos[CHASER]
        throw_dist = np.linalg.norm(throw)
        w = throw / throw_dist  # unit throw direction
        # In-plane basis for the net sheet (perpendicular to the throw).
        helper = np.array([0.0, 0.0, 1.0])
        if abs(w @ helper) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        u = np.cross(helper, w)
        u /= np.linalg.norm(u)
        v = np.cross(w, u)

        # Fold the sheet just ahead of the chaser, facing the cloud.
        local = self.net.positions * fold_scale
        self.pos[self.nodes] = (
            self.pos[CHASER] + 1.0 * w + local[:, 0, None] * u + local[:, 1, None] * v
        )

        # Aim: what centroid velocity arrives at (slightly past) the cloud
        # centroid in flight_time under CW dynamics?
        centroid = self.pos[self.nodes].mean(axis=0)
        target = target_point + aim_past * w
        dv1, _ = two_impulse_transfer(
            centroid - target_point,
            self.vel[CHASER],
            target - target_point,
            flight_time,
            self.n,
        )
        centroid_vel = self.vel[CHASER] + dv1
        straight_line_vel = (target - centroid) / flight_time
        aim_off = centroid_vel - (self.vel[CHASER] + straight_line_vel)

        self.vel[self.nodes] = centroid_vel
        # Weighted corners fly outward in the sheet plane, dragging it open.
        for ci in self.net.corners:
            r = self.pos[NODES0 + ci] - centroid
            r_in_plane = r - (r @ w) * w
            norm = np.linalg.norm(r_in_plane)
            if norm > 0:
                self.vel[NODES0 + ci] += spread_speed * r_in_plane / norm

        # Recoil: the chaser pays momentum for the throw.
        net_p = (
            self.net.masses[:, None] * (self.vel[self.nodes] - self.vel[CHASER])
        ).sum(axis=0)
        self.vel[CHASER] -= net_p / self.chaser_mass

        # Bridle: chaser to the four corners, slack until the net flies out.
        self._bridle_links = np.array([[CHASER, NODES0 + c] for c in self.net.corners])
        self._bridle_rest = np.full(4, throw_dist + self.bridle_slack)

        self._throw_dir = w.copy()
        self._phase = FLY
        self._events["eject"] = 0.0
        return {
            "throw_distance": float(throw_dist),
            "flight_time": flight_time,
            "centroid_speed": float(np.linalg.norm(centroid_vel - self.vel[CHASER])),
            "cw_aim_off": float(np.linalg.norm(aim_off)),
            "recoil": float(np.linalg.norm(net_p) / self.chaser_mass),
        }

    # ------------------------------------------------------------------ forces
    def _accelerations(self, t: float) -> tuple[np.ndarray, float, int]:
        """Force assembly -> (accel [B,3], tether tension [N], pellet contacts)."""
        forces = np.zeros_like(self.pos)
        node_pos = self.pos[self.nodes]
        node_vel = self.vel[self.nodes]

        # Net cords (link ids are node-local).
        forces[self.nodes] += link_forces(
            node_pos, node_vel, self.net.links, self._rest, self.k_link, self.c_link
        )

        # Bridle (operates on the full body arrays).
        tension = 0.0
        if self._bridle_links is not None:
            bridle_f = link_forces(
                self.pos, self.vel, self._bridle_links, self._bridle_rest,
                self.k_bridle, self.c_bridle,
            )
            forces += bridle_f
            tension = float(np.linalg.norm(bridle_f[CHASER]))

        # Membrane contact: pellets vs the triangulated fabric surface.
        contacts = 0
        if self.enable_contact:
            contacts = membrane_contact_forces(
                forces,
                node_pos,
                node_vel,
                node_rows=np.arange(self.net.n_nodes) + NODES0,
                triangles=self.net.triangles,
                pellet_pos=self.pos[self.pellets],
                pellet_vel=self.vel[self.pellets],
                pellet_rows=np.arange(self.cloud.n_pellets) + NODES0 + self.net.n_nodes,
                pellet_radii=self.cloud.radii,
                pellet_masses=self.cloud.masses,
                omega=self.contact_omega,
                zeta=self.contact_zeta,
                tangent_zeta=self.tangent_zeta,
            )

        # Tow thrust (continuous, on the chaser only, after the settle delay).
        if (
            self._phase == TOW
            and self._tow_dir is not None
            and t >= self._events.get("tow_start", np.inf)
        ):
            forces[CHASER] += self.chaser_mass * self.tow_accel * self._tow_dir

        accel = forces / self.mass[:, None]
        accel += cw_accelerations(self.pos, self.vel, self.n)
        return accel, tension, contacts

    # ------------------------------------------------------------------ phases
    def _update_phase(self, t: float, contacts: int) -> None:
        if self._phase == FLY and contacts > 0:
            self._phase = WRAP
            self._events["first_contact"] = t

        # Drawstring trigger: NOT first contact - that fires on the nearest
        # pellet, the leading edge of a cloud that extends metres deeper, and
        # closing then bags only the front. Close when the mouth ring has
        # swept past the cloud centroid: "the cloud is in the bag",
        # geometrically.
        if self._throw_dir is not None and "drawstring_start" not in self._events:
            ring = self.pos[NODES0 + self._boundary_nodes].mean(axis=0)
            if (ring - self.cloud_centroid()) @ self._throw_dir > self.drawstring_trigger_past:
                self._events["drawstring_start"] = t

        if "drawstring_start" in self._events:
            # Shrink perimeter rest lengths toward min_frac.
            frac = 1.0 - (1.0 - self.drawstring_min_frac) * np.clip(
                (t - self._events["drawstring_start"]) / self.drawstring_close_time, 0.0, 1.0
            )
            self._rest[self.net.perimeter] = self._rest0[self.net.perimeter] * frac

        if self._phase == WRAP:
            # Sustained capture criterion.
            if self._capture_criterion_met():
                if self._capture_ok_since is None:
                    self._capture_ok_since = t
                elif t - self._capture_ok_since >= self.capture_sustain:
                    self._captured = True
                    self._events["captured"] = t
                    self._phase = TOW
                    self._events["tow_start"] = t + self.tow_delay
                    d = self.pos[CHASER] - self.cloud_centroid()
                    self._tow_dir = d / np.linalg.norm(d)
            else:
                self._capture_ok_since = None

    def _capture_criterion_met(self) -> bool:
        return (
            self._retained_frac() >= self.capture_retain_frac
            and self._mouth_radius() < self.capture_mouth_frac * self._mouth_open_radius
        )

    def _retained_frac(self) -> float:
        """Fraction of pellets inside the bag (within the net's mean radius)."""
        net_centroid = self.pos[self.nodes].mean(axis=0)
        bag_radius = np.linalg.norm(self.pos[self.nodes] - net_centroid, axis=1).mean()
        d = np.linalg.norm(self.pos[self.pellets] - net_centroid, axis=1)
        return float((d < bag_radius + self.bag_margin).mean())

    def _mouth_radius(self) -> float:
        mouth = self.pos[NODES0 + self._boundary_nodes]
        return float(np.linalg.norm(mouth - mouth.mean(axis=0), axis=1).mean())

    # ------------------------------------------------------------------ run
    def run(self, duration: float, dt: float = 0.002, record_every: int = 25) -> CaptureResult:
        """Semi-implicit Euler to t=duration, recording every record_every steps."""
        steps = int(round(duration / dt))
        ts, poss, tensions, ccounts, mouths, retained = [], [], [], [], [], []

        for s in range(steps + 1):
            t = s * dt
            accel, tension, contacts = self._accelerations(t)
            if s % record_every == 0:
                ts.append(t)
                poss.append(self.pos.copy())
                tensions.append(tension)
                ccounts.append(contacts)
                mouths.append(self._mouth_radius())
                retained.append(self._retained_frac())
            # Symplectic order: new velocity first, then position from it.
            self.vel += accel * dt
            self.pos += self.vel * dt
            self._update_phase(t, contacts)

        return CaptureResult(
            t=np.array(ts),
            pos=np.array(poss),
            tether_tension=np.array(tensions),
            contact_count=np.array(ccounts),
            mouth_radius=np.array(mouths),
            retained_frac=np.array(retained),
            events=dict(self._events),
            captured=self._captured,
            net=self.net,
            pellet_radii=self.cloud.radii.copy(),
            n_nodes=self.net.n_nodes,
        )
