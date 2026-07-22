"""Four-satellite funnel constellation. UNITS: METRES (the capture world).

The operational concept: the chaser stages ~1 km *behind* the debris cloud and
splits into four. Three satellites lead, holding a conical net's **mouth** open
in formation; the fourth trails at the **apex** as the cod-end where the catch
collects. The formation closes from behind, sweeps through the cloud like a
trawler, cinches the mouth shut, and regroups.

Why four satellites at all: in vacuum there is no drag to stream a towed net
open. A single chaser dragging a net just collapses it along the tether.
Formation flying does the job water does for a trawler - which is also why the
mouth costs fuel to hold: only pure along-track offsets are natural CW
equilibria, so a triangle held across the velocity vector must be actively
maintained, forever.

The approach is the project's recurring lesson in its sharpest form. Burning
prograde to "speed up and catch up" couples into radial motion (ẍ = +2nẏ) and
balloons you upward: +2 m/s from 1 km behind drifts +539 m up in 500 s and
still misses. Ask the CW targeting solve for a half-orbit transfer instead and
it answers with a burn that is *purely radial* - drop, go faster, let the gap
close. See ``approach_burn``.

Physics is shared with the single-chaser capture sim (capture.py):
``cw_accelerations`` and ``membrane_contact_forces`` are imported, not
reimplemented.
"""

from dataclasses import dataclass, field

import numpy as np

from orbdebris.capture import KM_TO_M, cw_accelerations, membrane_contact_forces
from orbdebris.debris import PelletCloud
from orbdebris.net import Net, link_forces
from orbdebris.relative import two_impulse_transfer

# Body array layout: rows 0..3 = satellites, then net nodes, then pellets.
N_SATS = 4
MOUTH_SATS = np.array([0, 1, 2])  # hold the mouth open, 120 deg apart
APEX_SAT = 3  # trails at the cod-end

SPLIT, APPROACH, SWEEP, CINCH, REGROUP = "SPLIT", "APPROACH", "SWEEP", "CINCH", "REGROUP"


def formation_slots(mouth_radius: float, length: float) -> np.ndarray:
    """Satellite slots in the *net's* local frame [4, 3].

    Mouth satellites sit on a circle of mouth_radius in the local x-y plane at
    z=0 (the mouth faces +z, the sweep direction); the apex satellite trails at
    -length. Matches build_funnel_net's local frame exactly.

    Note these are NOT centred on the formation's centre of mass - three
    satellites at z=0 and one at z=-length average to z=-length/4. Anything
    steering relative to the satellite centroid must subtract that mean (see
    ``centre_slots``), or it chases a target whose centroid is offset from the
    real one and thrusts against itself forever.
    """
    slots = np.zeros((N_SATS, 3))
    for k, sat in enumerate(MOUTH_SATS):
        th = 2 * np.pi * k / len(MOUTH_SATS)
        slots[sat] = [mouth_radius * np.cos(th), mouth_radius * np.sin(th), 0.0]
    slots[APEX_SAT] = [0.0, 0.0, -length]
    return slots


def centre_slots(slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift slots so they are centred on the satellite centroid.

    Returns (centred_slots, offset). Applying the same offset to the net's node
    positions keeps the rim aligned with the satellites that hold it.
    """
    offset = slots.mean(axis=0)
    return slots - offset, offset


def approach_burn(rel_r: np.ndarray, rel_v: np.ndarray, tof: float, n: float) -> np.ndarray:
    """Single burn that closes on the cloud and sweeps *through* it.

    Uses the same CW two-impulse solve that aims our rendezvous burns, but
    keeps only dv1: skipping the arrival burn is exactly what turns a
    rendezvous into a sweep-through. Target is the cloud centroid, so the
    formation transits it at whatever velocity the transfer delivers - and
    with tof = half an orbit the answer comes out purely radial.
    """
    dv1, _ = two_impulse_transfer(rel_r, rel_v, np.zeros(3), tof, n)
    return dv1


@dataclass
class FormationController:
    """Velocity-limited PD station-keeping, per satellite, in the formation's
    local frame. A rigid formation is not a natural CW motion, so this thrusts
    forever - the delta-v it burns is the honest price of holding a mouth open
    in vacuum.

    Velocity-limited rather than plain PD, and the distinction matters: a plain
    PD saturates *outward* whenever the position error is large (kp*err far
    exceeds max_accel), so it accelerates away from the slot and only brakes at
    the end - overshooting by v²/(2*a_max). Here the position loop instead
    commands a *velocity*, clamped to v_max; the inner loop regulates to it.
    The result is accelerate -> cruise -> brake, with no overshoot, and the
    deployment impulse cooperates with the controller instead of fighting it.
    """

    slots: np.ndarray  # [4, 3] target offsets from the formation centroid
    kp: float = 0.05  # [1/s] position -> commanded velocity
    kv: float = 0.5  # [1/s] velocity error -> commanded acceleration
    v_max: float = 0.15  # [m/s] cruise limit
    max_accel: float = 0.05  # [m/s^2] actuator limit
    dv_spent: np.ndarray = None

    def __post_init__(self):
        if self.dv_spent is None:
            self.dv_spent = np.zeros(N_SATS)

    def accelerations(
        self, sat_pos: np.ndarray, sat_vel: np.ndarray, basis: np.ndarray, dt: float
    ) -> np.ndarray:
        """Commanded accelerations [4, 3] (Hill frame). basis rows are the
        formation's local axes expressed in Hill coordinates."""
        centroid = sat_pos.mean(axis=0)
        centroid_vel = sat_vel.mean(axis=0)
        targets = centroid + self.slots @ basis  # local -> Hill
        err = targets - sat_pos

        v_cmd = self.kp * err
        speed = np.linalg.norm(v_cmd, axis=1)
        fast = speed > self.v_max
        v_cmd[fast] *= (self.v_max / speed[fast])[:, None]

        # Slots are fixed in the local frame, so the target velocity is the
        # centroid's velocity plus the commanded closing velocity.
        cmd = self.kv * (centroid_vel + v_cmd - sat_vel)
        mag = np.linalg.norm(cmd, axis=1)
        over = mag > self.max_accel
        cmd[over] *= (self.max_accel / mag[over])[:, None]

        self.dv_spent += np.linalg.norm(cmd, axis=1) * dt
        return cmd


@dataclass
class FunnelResult:
    t: np.ndarray
    pos: np.ndarray  # [S, B, 3]
    contact_count: np.ndarray
    mouth_radius: np.ndarray
    retained_frac: np.ndarray
    formation_error: np.ndarray  # [S] rms satellite slot error [m]
    formation_dv: np.ndarray  # [S] cumulative formation-keeping dv [m/s]
    events: dict
    captured: bool
    net: Net
    pellet_radii: np.ndarray
    n_nodes: int

    @property
    def sats(self) -> slice:
        return slice(0, N_SATS)

    @property
    def nodes(self) -> slice:
        return slice(N_SATS, N_SATS + self.n_nodes)

    @property
    def pellets(self) -> slice:
        return slice(N_SATS + self.n_nodes, None)


@dataclass
class FunnelSim:
    net: Net
    cloud: PelletCloud
    n: float
    sat_mass: float = 125.0  # each; 500 kg chaser splits into four
    mouth_radius: float = 5.0
    funnel_length: float = 12.0
    # Net cords.
    k_link: float = 400.0
    c_link: float = 6.0
    # Satellite-to-net bonds (stiff: the satellites *are* the mouth's frame).
    k_bond: float = 800.0
    c_bond: float = 30.0
    # Membrane contact (M3's gentle tuning).
    contact_omega: float = 150.0
    contact_zeta: float = 0.8
    tangent_zeta: float = 0.5
    enable_contact: bool = True
    # Formation control (velocity-limited PD; see FormationController).
    kp: float = 0.05
    kv: float = 0.5
    v_max: float = 0.15
    max_accel: float = 0.05
    # Drawstring / capture.
    drawstring_close_time: float = 4.0
    drawstring_min_frac: float = 0.05
    capture_retain_frac: float = 0.6
    capture_sustain: float = 2.0
    regroup_delay: float = 5.0

    pos: np.ndarray = field(default=None, repr=False)
    vel: np.ndarray = field(default=None, repr=False)
    mass: np.ndarray = field(default=None, repr=False)
    ctrl: FormationController = field(default=None, repr=False)
    _rest: np.ndarray = field(default=None, repr=False)
    _rest0: np.ndarray = field(default=None, repr=False)
    _bonds: np.ndarray = field(default=None, repr=False)
    _bond_rest: np.ndarray = field(default=None, repr=False)
    _basis: np.ndarray = field(default=None, repr=False)
    _slot_offset: np.ndarray = field(default=None, repr=False)
    _phase: str = SPLIT
    _events: dict = field(default_factory=dict)
    _capture_ok_since: float = None
    _captured: bool = False
    _split_dv: float = 0.0

    @property
    def sats(self) -> slice:
        return slice(0, N_SATS)

    @property
    def nodes(self) -> slice:
        return slice(N_SATS, N_SATS + self.net.n_nodes)

    @property
    def pellets(self) -> slice:
        return slice(N_SATS + self.net.n_nodes, None)

    @classmethod
    def from_hill_state(
        cls,
        net: Net,
        cloud: PelletCloud,
        n: float,
        chaser_rel_r_km: np.ndarray,
        chaser_rel_v_km: np.ndarray,
        **params,
    ) -> "FunnelSim":
        """Boundary from the mission world (km) into the capture world (m).
        The chaser starts as a single body; call split() to become four."""
        if net.mouth_nodes is None or net.apex_node is None:
            raise ValueError("FunnelSim needs a funnel net (see net.build_funnel_net).")
        if len(net.mouth_nodes) % len(MOUTH_SATS) != 0:
            raise ValueError(
                f"n_sectors ({len(net.mouth_nodes)}) must be divisible by "
                f"{len(MOUTH_SATS)} so the mouth satellites sit 120 deg apart. "
                "Otherwise one arc of the rim is left oversized and unsupported, "
                "and it sags badly inward - tension-only cords cannot push it back out."
            )
        sim = cls(net=net, cloud=cloud, n=n, **params)
        b = N_SATS + net.n_nodes + cloud.n_pellets
        sim.pos = np.zeros((b, 3))
        sim.vel = np.zeros((b, 3))
        start_r = np.asarray(chaser_rel_r_km, float) * KM_TO_M
        start_v = np.asarray(chaser_rel_v_km, float) * KM_TO_M
        sim.pos[sim.sats] = start_r
        sim.vel[sim.sats] = start_v
        sim.pos[sim.pellets] = cloud.positions
        sim.vel[sim.pellets] = cloud.velocities
        sim.mass = np.empty(b)
        sim.mass[sim.sats] = sim.sat_mass
        sim.mass[sim.nodes] = net.masses
        sim.mass[sim.pellets] = cloud.masses
        sim._rest0 = net.rest_lengths.copy()
        sim._rest = net.rest_lengths.copy()

        # Local frame: +z faces the cloud (the sweep direction).
        w = -start_r / np.linalg.norm(start_r)  # toward the cloud centroid (origin)
        helper = np.array([0.0, 0.0, 1.0])
        if abs(w @ helper) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        u = np.cross(helper, w)
        u /= np.linalg.norm(u)
        v = np.cross(w, u)
        sim._basis = np.vstack([u, v, w])  # rows: local x, y, z -> Hill

        # Centre the slots on the satellite centroid, and shift the net's local
        # frame by the same offset so the rim stays aligned with the satellites
        # holding it.
        slots, sim._slot_offset = centre_slots(
            formation_slots(sim.mouth_radius, sim.funnel_length)
        )
        sim.ctrl = FormationController(
            slots=slots, kp=sim.kp, kv=sim.kv, v_max=sim.v_max, max_accel=sim.max_accel,
        )
        return sim

    def cloud_centroid(self) -> np.ndarray:
        w = self.cloud.masses / self.cloud.masses.sum()
        return (w[:, None] * self.pos[self.pellets]).sum(axis=0)

    # ------------------------------------------------------------------ split
    def split(self, fold_scale: float = 0.04) -> dict:
        """One chaser becomes four: a momentum-conserving separation impulse
        that pushes the satellites (and the folded net with them) out toward
        their formation slots, where the controller catches and holds them.

        The separation speed is the controller's own cruise limit, so the
        impulse does the accelerating and the controller only has to brake -
        they cooperate rather than fight.
        """
        centroid = self.pos[self.sats].mean(axis=0)
        base_vel = self.vel[self.sats].mean(axis=0)
        slots_hill = self.ctrl.slots @ self._basis
        net_hill = (self.net.positions - self._slot_offset) @ self._basis

        # Start folded: everything close to the chaser's position.
        self.pos[self.sats] = centroid + slots_hill * fold_scale
        self.pos[self.nodes] = centroid + net_hill * fold_scale

        # Deploy outward at the cruise speed, scaled by how far each body must
        # travel so the whole assembly unfurls together.
        def deploy_velocity(offsets: np.ndarray) -> np.ndarray:
            dist = np.linalg.norm(offsets, axis=1)
            reach = max(dist.max(), 1e-9)
            dirs = offsets / np.maximum(dist, 1e-9)[:, None]
            return self.ctrl.v_max * (dist / reach)[:, None] * dirs

        dv_sats = deploy_velocity(slots_hill)
        dv_nodes = deploy_velocity(net_hill)

        # Conserve momentum: the separation is internal, so remove the
        # mass-weighted mean of everything it pushed.
        m_s = self.mass[self.sats][:, None]
        m_n = self.mass[self.nodes][:, None]
        total_p = (m_s * dv_sats).sum(axis=0) + (m_n * dv_nodes).sum(axis=0)
        total_m = self.mass[self.sats].sum() + self.mass[self.nodes].sum()
        drift = total_p / total_m
        dv_sats -= drift
        dv_nodes -= drift

        self.vel[self.sats] = base_vel + dv_sats
        self.vel[self.nodes] = base_vel + dv_nodes
        self._split_dv = float(np.linalg.norm(dv_sats, axis=1).mean())

        # Bond the satellites to the rim and the apex.
        mouth = self.net.mouth_nodes
        step = len(mouth) // len(MOUTH_SATS)
        bonds = [[MOUTH_SATS[k], N_SATS + int(mouth[k * step])] for k in range(len(MOUTH_SATS))]
        bonds.append([APEX_SAT, N_SATS + int(self.net.apex_node)])
        self._bonds = np.array(bonds)
        self._bond_rest = np.full(len(bonds), 0.05)  # short, stiff: sats hold the rim

        self._phase = APPROACH
        self._events["split"] = 0.0
        return {"split_dv_per_sat": self._split_dv, "slots": slots_hill}

    def burn_approach(self, tof: float) -> np.ndarray:
        """Apply the single closing burn to every body in the formation."""
        centroid = self.pos[self.sats].mean(axis=0)
        vel = self.vel[self.sats].mean(axis=0)
        dv = approach_burn(centroid - self.cloud_centroid(), vel, tof, self.n)
        self.vel[self.sats] += dv
        self.vel[self.nodes] += dv
        self._events["approach_burn"] = 0.0
        return dv

    # ------------------------------------------------------------------ forces
    def _accelerations(self, t: float, dt: float) -> tuple[np.ndarray, int]:
        forces = np.zeros_like(self.pos)
        node_pos, node_vel = self.pos[self.nodes], self.vel[self.nodes]

        forces[self.nodes] += link_forces(
            node_pos, node_vel, self.net.links, self._rest, self.k_link, self.c_link
        )
        if self._bonds is not None:
            forces += link_forces(
                self.pos, self.vel, self._bonds, self._bond_rest, self.k_bond, self.c_bond
            )

        contacts = 0
        if self.enable_contact:
            contacts = membrane_contact_forces(
                forces,
                node_pos,
                node_vel,
                node_rows=np.arange(self.net.n_nodes) + N_SATS,
                triangles=self.net.triangles,
                pellet_pos=self.pos[self.pellets],
                pellet_vel=self.vel[self.pellets],
                pellet_rows=np.arange(self.cloud.n_pellets) + N_SATS + self.net.n_nodes,
                pellet_radii=self.cloud.radii,
                pellet_masses=self.cloud.masses,
                omega=self.contact_omega,
                zeta=self.contact_zeta,
                tangent_zeta=self.tangent_zeta,
            )

        accel = forces / self.mass[:, None]
        accel += cw_accelerations(self.pos, self.vel, self.n)

        # Formation-keeping thrust (satellites only), on top of everything.
        if self._phase != SPLIT:
            accel[self.sats] += self.ctrl.accelerations(
                self.pos[self.sats], self.vel[self.sats], self._basis, dt
            )
        return accel, contacts

    # ------------------------------------------------------------------ state
    def formation_error(self) -> float:
        centroid = self.pos[self.sats].mean(axis=0)
        targets = centroid + self.ctrl.slots @ self._basis
        return float(np.linalg.norm(self.pos[self.sats] - targets, axis=1).mean())

    def mouth_radius_now(self) -> float:
        mouth = self.pos[N_SATS + self.net.mouth_nodes]
        return float(np.linalg.norm(mouth - mouth.mean(axis=0), axis=1).mean())

    def retained_frac(self) -> float:
        """Fraction of pellets inside the funnel: within the mouth radius of
        the funnel axis, and between the mouth plane and the apex."""
        axis = self._basis[2]  # local +z, pointing out of the mouth
        mouth_c = self.pos[N_SATS + self.net.mouth_nodes].mean(axis=0)
        apex = self.pos[N_SATS + self.net.apex_node]
        rel = self.pos[self.pellets] - mouth_c
        along = rel @ axis  # negative = behind the mouth plane, i.e. inside
        radial = np.linalg.norm(rel - along[:, None] * axis, axis=1)
        depth = float(np.linalg.norm(apex - mouth_c))
        inside = (along < 0.5) & (along > -depth - 1.0) & (radial < self.mouth_radius + 1.0)
        return float(inside.mean())

    def _update_phase(self, t: float, contacts: int) -> None:
        if self._phase == APPROACH and contacts > 0:
            self._phase = SWEEP
            self._events["first_contact"] = t

        # Cinch on the geometric cue that made M3 work: the mouth ring has
        # swept past the cloud centroid, so the catch is behind the rim.
        if self._phase == SWEEP and "cinch_start" not in self._events:
            mouth_c = self.pos[N_SATS + self.net.mouth_nodes].mean(axis=0)
            if (mouth_c - self.cloud_centroid()) @ self._basis[2] > 0:
                self._events["cinch_start"] = t
                self._phase = CINCH

        if "cinch_start" in self._events:
            frac = 1.0 - (1.0 - self.drawstring_min_frac) * np.clip(
                (t - self._events["cinch_start"]) / self.drawstring_close_time, 0.0, 1.0
            )
            self._rest[self.net.perimeter] = self._rest0[self.net.perimeter] * frac
            # Draw the mouth satellites in with the rim they are holding.
            self.ctrl.slots[MOUTH_SATS, :2] = (
                formation_slots(self.mouth_radius, self.funnel_length)[MOUTH_SATS, :2] * frac
            )

        if self._phase == CINCH:
            if self.retained_frac() >= self.capture_retain_frac and self.mouth_radius_now() < (
                0.4 * self.mouth_radius
            ):
                if self._capture_ok_since is None:
                    self._capture_ok_since = t
                elif t - self._capture_ok_since >= self.capture_sustain:
                    self._captured = True
                    self._events["captured"] = t
                    self._events["regroup_start"] = t + self.regroup_delay
                    self._phase = REGROUP
            else:
                self._capture_ok_since = None

        if self._phase == REGROUP and t >= self._events.get("regroup_start", np.inf):
            # Close the formation up for transit to the next target: the mouth
            # satellites converge toward the apex.
            self.ctrl.slots[MOUTH_SATS, 2] = -0.6 * self.funnel_length

    # ------------------------------------------------------------------ run
    def run(self, duration: float, dt: float = 0.002, record_every: int = 50) -> FunnelResult:
        steps = int(round(duration / dt))
        ts, poss, ccounts, mouths, retained, ferr, fdv = [], [], [], [], [], [], []

        for s in range(steps + 1):
            t = s * dt
            accel, contacts = self._accelerations(t, dt)
            if s % record_every == 0:
                ts.append(t)
                poss.append(self.pos.copy())
                ccounts.append(contacts)
                mouths.append(self.mouth_radius_now())
                retained.append(self.retained_frac())
                ferr.append(self.formation_error())
                fdv.append(float(self.ctrl.dv_spent.sum()))
            self.vel += accel * dt
            self.pos += self.vel * dt
            self._update_phase(t, contacts)

        return FunnelResult(
            t=np.array(ts),
            pos=np.array(poss),
            contact_count=np.array(ccounts),
            mouth_radius=np.array(mouths),
            retained_frac=np.array(retained),
            formation_error=np.array(ferr),
            formation_dv=np.array(fdv),
            events=dict(self._events),
            captured=self._captured,
            net=self.net,
            pellet_radii=self.cloud.radii.copy(),
            n_nodes=self.net.n_nodes,
        )
