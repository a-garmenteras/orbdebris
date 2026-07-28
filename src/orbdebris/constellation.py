"""Four-satellite funnel constellation. UNITS: METRES (the capture world).

The operational concept: the chaser stages ~1 km *behind* the debris cloud and
splits into four. Three satellites lead, holding a conical net's **mouth** open
in formation; the fourth trails at the **apex**, which carries a storage box -
the cod-end where the catch collects. The formation closes from behind and
sweeps through the cloud like a trawler. The three satellites **never let go**:
the funnel is a permanent structure that channels debris down its
energy-absorbing walls into the apex box, driven by the sweep motion and a
gentle shepherding thrust (there is no gravity to do it). Capture = debris
settled in the box; the funnel stays open, ready to tow the catch away.

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
from orbdebris.net import (
    Net,
    link_forces,
    membrane_element_forces,
    membrane_rest_data,
)
# Ledger phase labels are aliased: this module already uses SPLIT/SWEEP as
# state-machine names (line below). The strings happen to coincide today, but
# relying on that would be an invisible coupling between two unrelated
# vocabularies.
from orbdebris.propulsion import SPLIT as SPLIT_PHASE
from orbdebris.propulsion import SWEEP as SWEEP_PHASE
from orbdebris.propulsion import FORMATION, TENSION, FuelLedger
from orbdebris.relative import two_impulse_transfer

# Body array layout: rows 0..3 = satellites, then net nodes, then pellets.
N_SATS = 4
MOUTH_SATS = np.array([0, 1, 2])  # hold the mouth open, 120 deg apart
APEX_SAT = 3  # trails at the cod-end

SPLIT, APPROACH, SWEEP, SECURE = "SPLIT", "APPROACH", "SWEEP", "SECURE"


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


def funnel_demo(n_pellets: int = 30, seed: int = 11, standoff: float = 10.0):
    """The canonical, tuned M4 scenario - single source of truth for both
    scripts/run_funnel.py and the end-to-end test, so they cannot drift.

    Returns (sim, sweep_velocity, approach). ``approach`` reports the analytic
    half-orbit approach from 1 km behind (burn, tof, arrival velocity); the
    coast itself is pure CW flow and is not simulated at fine timesteps. The
    caller does ``sim.deploy_open(sweep_velocity)`` then ``sim.run(...)`` to run
    the terminal sweep.
    """
    from orbdebris.constants import GM_EARTH, R_EARTH
    from orbdebris.debris import make_pellet_cloud
    from orbdebris.net import build_tetra_net
    from orbdebris.relative import cw_propagate, mean_motion

    n = mean_motion(R_EARTH + 500.0, GM_EARTH)
    tof = np.pi / n  # half an orbit

    # Analytic approach from 1 km behind: the burn is purely radial-down, and
    # the arrival velocity (the sweep) comes back purely radial-up.
    behind = np.array([0.0, -1000.0, 0.0])
    burn = approach_burn(behind, np.zeros(3), tof, n)
    _, sweep_velocity = cw_propagate(behind, burn, tof, n)
    sweep_dir = sweep_velocity / np.linalg.norm(sweep_velocity)

    # Tetrahedral funnel: 3 flat faces + open triangular mouth, its shape fixed
    # by the 4 satellite positions (so it cannot concertina like the cone did).
    # Corners at 6.2 m match the old 4 m circular mouth's area while presenting
    # shallower ~11 deg faces. The fabric is a continuous membrane (shell
    # elements) - a cord-only net collapses and leaks.
    net = build_tetra_net(mouth_radius=6.2, length=16.0, subdiv=6, total_mass=60.0)
    cloud = make_pellet_cloud(n_pellets=n_pellets, sigma_pos=1.0, seed=seed)
    sim = FunnelSim.from_hill_state(
        net,
        cloud,
        n,
        chaser_rel_r_km=(-standoff * sweep_dir) / 1000.0,  # standoff before the cloud
        chaser_rel_v_km=np.zeros(3),
        mouth_radius=6.2,
        funnel_length=16.0,
        # Slippery chute, padded catch: inelastic normal contact (no bounce)
        # with near-frictionless tangential sliding, so debris runs all the way
        # down to the collector, whose padded walls absorb the arrival.
        contact_omega=150.0,
        contact_zeta=0.95,
        tangent_zeta=0.005,
        capture_frac=0.5,
    )
    approach = {"burn": burn, "tof": tof, "sweep_velocity": sweep_velocity}
    return sim, sweep_velocity, approach


def run_full_mission(
    standoff: float = 20.0,
    settle: float = 50.0,
    sweep: float = 150.0,  # long enough for debris to reach the apex box and secure
    dt: float = 0.0015,
    record_every: int = 130,
    seed: int = 11,
    ledger: FuelLedger | None = None,
    vehicle_names: list[str] | None = None,
):
    """One continuous run covering the whole operation, for the animation:
    split (deploy the net) -> settle (mouth opens, formation holds) -> approach
    burn -> sweep through the cloud -> collect (debris channels to the apex box)
    -> secure. The satellites hold the mouth open the entire time.

    The 47-minute half-orbit coast between the burn and arrival is not stepped
    here (it is pure CW flow); the analytic arrival velocity is injected as the
    burn that starts the sweep. Returns (sim, result, approach).
    """
    sim, sweep_velocity, approach = funnel_demo(standoff=standoff, seed=seed)
    names = vehicle_names or [f"sat{i}" for i in range(N_SATS)]
    split = sim.split()
    if ledger is not None:
        for k, name in enumerate(names):
            ledger.record(0.0, float(split["split_dv_by_sat"][k]), SPLIT_PHASE, name)
        # The cost of closing on the cloud is the *departure* burn computed
        # analytically 47 minutes earlier - purely radial-down, from 1 km
        # behind. `sweep_velocity` is the arrival state that burn produces, not
        # a second impulse: skipping the arrival burn is exactly what makes this
        # a sweep-through instead of a rendezvous. Billing |sweep_velocity|
        # would double-charge the manoeuvre.
        dv_burn = float(np.linalg.norm(approach["burn"]))
        for name in names:
            ledger.record(0.0, dv_burn, SWEEP_PHASE, name)
    sim.hold_position()  # loiter in place while the net deploys
    result = sim.run(
        settle + sweep,
        dt=dt,
        record_every=record_every,
        sweep_velocity=sweep_velocity,
        sweep_burn_time=settle,
        trim_time=0.8 * settle,  # re-trim to the settled shape before the burn
        ledger=ledger,
        vehicle_names=names,
    )
    return sim, result, approach


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
    # Of dv_spent, the part the apex thruster used to unfurl the funnel. Kept
    # as a *subset* rather than a separate budget so dv_spent stays the single
    # total (formation_dv is unchanged by this bookkeeping), while the ledger
    # can still bill unfurling and mouth-holding to different phases.
    dv_tension: np.ndarray = None

    def __post_init__(self):
        if self.dv_spent is None:
            self.dv_spent = np.zeros(N_SATS)
        if self.dv_tension is None:
            self.dv_tension = np.zeros(N_SATS)

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
    mouth_radius: np.ndarray  # [S] should stay ~constant: the mouth never closes
    stored_frac: np.ndarray  # [S] fraction of pellets settled in the apex box
    inside_funnel_frac: np.ndarray  # [S] fraction engulfed by the funnel volume
    stored_mask: np.ndarray  # [S, P] which pellets are in the box (for drawing)
    formation_error: np.ndarray  # [S] rms satellite slot error [m]
    formation_dv: np.ndarray  # [S] cumulative formation-keeping dv [m/s]
    events: dict
    captured: bool
    net: Net
    pellet_radii: np.ndarray
    n_nodes: int
    box_radius: float = 1.0  # collector aperture, for drawing

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
    mouth_radius: float = 4.0
    funnel_length: float = 16.0
    # Net cords (edge reinforcement along the mesh lines).
    k_link: float = 400.0
    c_link: float = 6.0
    # Continuous-membrane shell elements: the fabric is a surface with in-plane
    # stretch/shear/COMPRESSION stiffness (a balloon skin), not just cords.
    # This is what lets the cone hold its shape - a cord lattice goes slack
    # under compression and the funnel collapses/leaks (isolated test: cords
    # alone collect 16-18/30; with shell elements 29/30).
    k_membrane: float = 150.0  # areal stiffness (Young's modulus x thickness)
    membrane_poisson: float = 0.3
    membrane_damping: float = 2.0
    # Satellite-to-net bonds (stiff: the satellites *are* the mouth's frame).
    k_bond: float = 800.0
    c_bond: float = 30.0
    # Membrane contact, split by direction. Normal: heavily damped (inelastic -
    # absorb the perpendicular impact so debris does not bounce back out).
    # Tangential: nearly frictionless, so debris keeps sliding *along* the wall
    # all the way down to the collector. The funnel is a slippery chute, not a
    # sticky one: at tangent_zeta 0.05 debris parks strung out along the cone
    # (8/30 reach a 1 m collector, median 2.2 m); at 0.005 it slides home
    # (26/30, median 0.4 m). The energy is absorbed at the *collector* instead,
    # whose walls are padded - see box_damping.
    contact_omega: float = 150.0
    contact_zeta: float = 0.95
    tangent_zeta: float = 0.005
    enable_contact: bool = True
    # Formation control (velocity-limited PD; see FormationController).
    kp: float = 0.05
    kv: float = 0.5
    v_max: float = 0.15
    max_accel: float = 0.05
    # Apex tensioning: the apex satellite's thruster pulls *backward* along the
    # funnel axis to unfurl it completely. Without it nothing tensions the
    # funnel lengthwise - the membrane stays folded and the cone concertinas to
    # half its design length (measured 51%, folding back on itself).
    apex_tension_accel: float = 0.06  # [m/s^2] backward pull during deploy
    extension_target: float = 0.95  # stop pulling at this fraction of design
    # The apex satellite IS the collector: a compartment that opens and closes,
    # so the "stored" radius is the satellite's own size, not an arbitrary
    # volume. Debris can enter but a boundary spring pushes back any that
    # tries to drift out.
    box_radius: float = 1.0
    box_k: float = 30.0  # container-wall stiffness [N/m per unit mass-scaling]
    # Padded compartment walls: debris arriving off a near-frictionless funnel
    # still carries its sliding speed, so the collector is where that energy is
    # absorbed. Damping rate [1/s] on debris inside the compartment.
    box_damping: float = 3.0
    # Shepherding: an OPTIONAL gentle forward accel. Off by default: the sweep
    # motion alone funnels debris to the apex (a fixed-node test collects 30/30
    # with zero shepherding). Nonzero shepherding actually drives debris into
    # the walls harder and stalls it mid-funnel - kept only as a knob.
    shepherd_accel: float = 0.0  # [m/s^2] along the mouth axis
    # Capture criterion: fraction of pellets resident in the box, low relative
    # speed, sustained.
    capture_frac: float = 0.6
    settle_speed: float = 0.15  # [m/s] max relative speed to count as settled
    capture_sustain: float = 2.0

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
    _hold_target: np.ndarray = field(default=None, repr=False)
    _in_box: np.ndarray = field(default=None, repr=False)  # [P] has entered the box
    _dm_inv: np.ndarray = field(default=None, repr=False)  # membrane rest data
    _tri_area: np.ndarray = field(default=None, repr=False)
    _shepherding: bool = False
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
        if len(net.corners) < len(MOUTH_SATS):
            raise ValueError(
                f"the funnel needs at least {len(MOUTH_SATS)} corner nodes for the "
                f"mouth satellites to hold, got {len(net.corners)}."
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
        sim._in_box = np.zeros(cloud.n_pellets, dtype=bool)
        sim._dm_inv, sim._tri_area = membrane_rest_data(net.positions, net.triangles)
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
    def split(self, fold_scale: float = 0.35) -> dict:
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
        per_sat = np.linalg.norm(dv_sats, axis=1)
        self._split_dv = float(per_sat.mean())

        self._attach_bonds()
        self._phase = APPROACH
        self._events["split"] = 0.0
        # per-sat magnitudes as well as the mean: the separation impulse is not
        # equal across the four (the apex travels furthest), and the ledger
        # bills each satellite's own tank.
        return {
            "split_dv_per_sat": self._split_dv,
            "split_dv_by_sat": per_sat,
            "slots": slots_hill,
        }

    def trim_slots(self) -> None:
        """Re-trim the formation slots to the shape actually achieved.

        The membrane's equilibrium rim radius is slightly inside the as-built
        geometry (Poisson contraction of the cone), so slots at the blueprint
        radius leave the controller leaning on the structure forever - a
        steady ~0.13 m/s/s-per-satellite tug-of-war it can never win. A real
        GNC system trims to the achieved shape instead of fighting it: set
        each slot to the satellite's current offset from the centroid,
        expressed in the formation's local frame."""
        centroid = self.pos[self.sats].mean(axis=0)
        offsets_hill = self.pos[self.sats] - centroid
        self.ctrl.slots, _ = centre_slots(offsets_hill @ self._basis.T)

    def hold_position(self) -> None:
        """Station-keep the formation's whole absolute position (not just its
        shape) at where it is now, until a sweep burn releases it. Used to
        loiter behind the cloud while the net deploys, instead of letting the
        deploy transient billow the funnel forward into the cloud."""
        self._hold_target = self.pos[self.sats].mean(axis=0).copy()

    def _attach_bonds(self) -> None:
        """Bond each mouth satellite to its corner of the funnel's mouth, and
        the 4th to the apex, with short stiff tension-only links. For a
        tetrahedron the three corners *are* the satellite stations, so the four
        bonds pin the whole shape - which is the point of the tetrahedron."""
        corners = self.net.corners[: len(MOUTH_SATS)]
        bonds = [[MOUTH_SATS[k], N_SATS + int(corners[k])] for k in range(len(MOUTH_SATS))]
        bonds.append([APEX_SAT, N_SATS + int(self.net.apex_node)])
        self._bonds = np.array(bonds)
        self._bond_rest = np.full(len(bonds), 0.05)  # short, stiff: sats hold the rim

    def deploy_open(self, sweep_velocity: np.ndarray) -> None:
        """Seed the funnel already fully open at its slots, moving at
        sweep_velocity [m/s, Hill frame]. Skips the fold/deploy/settle
        transient (validated in Checkpoint 1) to focus a run on the sweep."""
        centroid = self.pos[self.sats].mean(axis=0)
        self.pos[self.sats] = centroid + self.ctrl.slots @ self._basis
        self.pos[self.nodes] = centroid + (self.net.positions - self._slot_offset) @ self._basis
        self.vel[self.sats] = sweep_velocity
        self.vel[self.nodes] = sweep_velocity
        self._attach_bonds()
        self._phase = APPROACH

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
        # Continuous membrane: in-plane shell stiffness (incl. compression),
        # the balloon-skin behaviour that keeps the cone from collapsing.
        forces[self.nodes] += membrane_element_forces(
            node_pos, node_vel, self.net.triangles, self._dm_inv, self._tri_area,
            self.k_membrane, self.membrane_poisson, self.membrane_damping,
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

        # Apex storage box: a one-way soft container. A pellet that reaches the
        # box (within box_radius of the apex node) is *latched* as collected;
        # thereafter, if it drifts back out past the boundary, a spring pushes
        # it back in. Crucially the constraint acts ONLY on already-collected
        # pellets - distant cloud debris feels nothing (otherwise the box would
        # be a tractor beam sucking in the whole cloud). Debris enters by being
        # channelled down the funnel, not pulled.
        apex_row = N_SATS + self.net.apex_node
        apex = self.pos[apex_row]
        apex_vel = self.vel[apex_row]
        pel_rows = np.arange(self.cloud.n_pellets) + N_SATS + self.net.n_nodes
        rel = self.pos[self.pellets] - apex
        d = np.linalg.norm(rel, axis=1)
        self._in_box |= d < self.box_radius  # latch on entry
        escaping = self._in_box & (d > self.box_radius)
        if np.any(escaping):
            esc = np.flatnonzero(escaping)
            u = rel[esc] / d[esc, None]  # outward radial unit
            v_out = np.einsum("ij,ij->i", self.vel[self.pellets][esc] - apex_vel, u)
            depth = d[esc] - self.box_radius
            m_esc = self.cloud.masses[esc]
            # Spring back in, plus damping of outward motion only (one-way).
            f_mag = self.box_k * m_esc * (depth + 0.3 * np.maximum(v_out, 0.0) / self.contact_omega)
            f_vec = f_mag[:, None] * u
            forces[pel_rows[esc]] -= f_vec
            # Equal-and-opposite reaction on the apex node the box is mounted
            # to: the wall pushes the pellet in, the pellet pushes the box out.
            # Without this the box is a force from nowhere and the whole
            # assembly self-accelerates - the thrusters then fight a phantom
            # forever (measured: 40 m/s of formation dv from exactly this).
            forces[apex_row] += f_vec.sum(axis=0)

        # Padded compartment: debris that has arrived is brought to rest against
        # the apex satellite. The funnel walls are deliberately slippery so
        # debris keeps sliding home; *this* is where its energy is absorbed.
        inside = self._in_box & (d < self.box_radius)
        if np.any(inside):
            ins = np.flatnonzero(inside)
            v_rel = self.vel[self.pellets][ins] - apex_vel
            f_pad = -self.box_damping * self.cloud.masses[ins, None] * v_rel
            forces[pel_rows[ins]] += f_pad
            forces[apex_row] -= f_pad.sum(axis=0)  # padding pushes back, too

        accel = forces / self.mass[:, None]
        accel += cw_accelerations(self.pos, self.vel, self.n)

        # Apex tensioning: the 4th satellite's thruster pulls backward along the
        # funnel axis, unfurling the membrane to its full design length. Runs
        # only while the funnel is short of extension_target, so it stops once
        # the funnel is taut instead of thrusting forever.
        if (
            self.apex_tension_accel > 0
            and self._phase in (APPROACH, SPLIT)
            and self.funnel_extension() < self.extension_target
        ):
            accel[APEX_SAT] -= self.apex_tension_accel * self._basis[2]
            self.ctrl.dv_spent[APEX_SAT] += self.apex_tension_accel * dt
            self.ctrl.dv_tension[APEX_SAT] += self.apex_tension_accel * dt

        # Shepherding: a gentle continuous forward thrust on the whole formation
        # once the sweep is engulfing the cloud. In the funnel's frame this is a
        # steady apex-ward pseudo-force that drives debris down the walls into
        # the box and pins it there - the job gravity does on Earth but cannot
        # do here. Applied to satellites and net together so the funnel keeps
        # its shape; its dv is counted as formation-keeping.
        if self._shepherding:
            push = self.shepherd_accel * self._basis[2]  # +mouth axis
            accel[self.sats] += push
            accel[self.nodes] += push
            self.ctrl.dv_spent += self.shepherd_accel * dt

        # Formation-keeping thrust (satellites only), on top of everything.
        if self._phase != SPLIT:
            accel[self.sats] += self.ctrl.accelerations(
                self.pos[self.sats], self.vel[self.sats], self._basis, dt
            )

        # Station-keep the whole formation's absolute position while it loiters
        # behind the cloud (before the sweep burn). The controller above only
        # holds the formation's *shape*; its centroid otherwise rides free CW
        # flow, and the net's deploy transient billows it forward into the cloud
        # before the burn. Real formation flying holds position relative to the
        # target, so we add a centroid-position hold, released when the sweep
        # burn fires. The dv it costs is counted like any station-keeping.
        if self._hold_target is not None:
            c = self.pos[self.sats].mean(axis=0)
            c_vel = self.vel[self.sats].mean(axis=0)
            cmd = self.ctrl.kp * (self._hold_target - c) - self.ctrl.kv * c_vel
            mag = np.linalg.norm(cmd)
            if mag > self.ctrl.max_accel:
                cmd *= self.ctrl.max_accel / mag
            accel[self.sats] += cmd
            self.ctrl.dv_spent += np.linalg.norm(cmd) * dt / N_SATS
        return accel, contacts

    # ------------------------------------------------------------------ state
    def formation_error(self) -> float:
        centroid = self.pos[self.sats].mean(axis=0)
        targets = centroid + self.ctrl.slots @ self._basis
        return float(np.linalg.norm(self.pos[self.sats] - targets, axis=1).mean())

    def mouth_radius_now(self) -> float:
        mouth = self.pos[N_SATS + self.net.mouth_nodes]
        return float(np.linalg.norm(mouth - mouth.mean(axis=0), axis=1).mean())

    def mouth_corner_radius(self) -> float:
        """Mean distance of the three held *corners* from the mouth centroid -
        the meaningful "is the mouth open?" measure for a triangular mouth.
        (``mouth_radius_now`` averages the whole perimeter, which for a triangle
        mixes corners at R with edge midpoints at R/2, so a fully open
        triangular mouth reads only ~0.75 R there.)"""
        mouth = self.pos[N_SATS + self.net.mouth_nodes]
        corners = self.pos[N_SATS + self.net.corners[: len(MOUTH_SATS)]]
        return float(np.linalg.norm(corners - mouth.mean(axis=0), axis=1).mean())

    def funnel_extension(self) -> float:
        """Deployed axial length / design length. The diagnostic that would
        have caught the concertina fold instantly: the folded cone read 0.51
        while its radii all looked correct."""
        mouth_c = self.pos[N_SATS + self.net.mouth_nodes].mean(axis=0)
        apex = self.pos[N_SATS + self.net.apex_node]
        axial = abs((mouth_c - apex) @ self._basis[2])
        return float(axial / self.funnel_length)

    def stored_mask(self) -> np.ndarray:
        """Boolean [P]: pellets collected in the apex box - they have entered it
        (the one-way latch) and are settled (low speed relative to the apex)."""
        apex_row = N_SATS + self.net.apex_node
        v_rel = np.linalg.norm(self.vel[self.pellets] - self.vel[apex_row], axis=1)
        d = np.linalg.norm(self.pos[self.pellets] - self.pos[apex_row], axis=1)
        return self._in_box & (d < self.box_radius + 0.5) & (v_rel < self.settle_speed)

    def stored_frac(self) -> float:
        """Fraction of pellets collected and settled in the apex box."""
        return float(self.stored_mask().mean())

    def inside_funnel_frac(self) -> float:
        """Diagnostic: fraction of pellets within the funnel's bounding volume
        (within one funnel-length of the net centroid). Reads how many have
        been engulfed, whether or not they have reached the box yet."""
        net_c = self.pos[self.nodes].mean(axis=0)
        d = np.linalg.norm(self.pos[self.pellets] - net_c, axis=1)
        return float((d < self.funnel_length).mean())

    def in_box_frac(self) -> float:
        """Fraction of pellets that have *entered* the box (the one-way latch),
        settled or not. Drives the decision to stop shepherding."""
        return float(self._in_box.mean())

    def _update_phase(self, t: float, contacts: int) -> None:
        if self._phase == APPROACH and contacts > 0:
            self._phase = SWEEP
            self._events["first_contact"] = t

        # Optional shepherding (off by default; the sweep alone funnels the
        # debris). If enabled, engage once the mouth has swept past the cloud
        # centroid and stop once enough debris is in the box.
        if (
            self.shepherd_accel > 0
            and self._phase == SWEEP
            and not self._shepherding
            and "shepherd_stop" not in self._events
        ):
            mouth_c = self.pos[N_SATS + self.net.mouth_nodes].mean(axis=0)
            if (mouth_c - self.cloud_centroid()) @ self._basis[2] > 0:
                self._shepherding = True
                self._events["shepherd_start"] = t
        if self._shepherding and self.in_box_frac() >= self.capture_frac:
            self._shepherding = False
            self._events["shepherd_stop"] = t

        # Secure when a majority of the debris is stored in the box and settled,
        # sustained. The funnel stays open the whole time - it is the structure
        # holding the catch, ready to tow away. The satellites never let go.
        if self._phase == SWEEP and not self._shepherding:
            if self.stored_frac() >= self.capture_frac:
                if self._capture_ok_since is None:
                    self._capture_ok_since = t
                elif t - self._capture_ok_since >= self.capture_sustain:
                    self._captured = True
                    self._events["secured"] = t
                    self._phase = SECURE
            else:
                self._capture_ok_since = None

    # ------------------------------------------------------------------ run
    def run(
        self,
        duration: float,
        dt: float = 0.002,
        record_every: int = 50,
        sweep_velocity: np.ndarray | None = None,
        sweep_burn_time: float = 0.0,
        trim_time: float | None = None,
        ledger: FuelLedger | None = None,
        vehicle_names: list[str] | None = None,
    ) -> FunnelResult:
        """Integrate to t=duration.

        sweep_velocity: if given, applied to the whole formation (satellites +
        net) at sweep_burn_time - this is the arrival velocity of the analytic
        approach, injected as the maneuver that starts the sweep. It lets one
        continuous run cover deploy -> settle -> sweep -> collect -> secure (see
        run_full_mission), which is what the animation shows.

        ledger: optional FuelLedger. Formation-keeping is *continuous* thrust,
        so the controller keeps its fast per-step accumulator (correct, and at
        dt~1.5 ms we are not writing 100k ledger rows) and this **flushes the
        delta** into the ledger once per recorded frame - one entry per
        satellite, split into FORMATION and TENSION. Fine granularity where it
        is free, coarse where it is not: the same scale separation the analytic
        approach / stepped sweep split already uses.
        """
        steps = int(round(duration / dt))
        names = vehicle_names or [f"sat{i}" for i in range(N_SATS)]
        last_form = self.ctrl.dv_spent - self.ctrl.dv_tension
        last_tens = self.ctrl.dv_tension.copy()
        ts, poss, ccounts, mouths, stored, inside, smask, ferr, fdv = (
            [], [], [], [], [], [], [], [], []
        )
        swept = sweep_velocity is None
        trimmed = trim_time is None

        for s in range(steps + 1):
            t = s * dt
            if not trimmed and t >= trim_time:
                self.trim_slots()  # stop leaning on the settled structure
                self._events["trim"] = t
                trimmed = True
            if not swept and t >= sweep_burn_time:
                self.vel[self.sats] += sweep_velocity
                self.vel[self.nodes] += sweep_velocity
                self._hold_target = None  # release the loiter hold; start sweeping
                self._events["approach_burn"] = t
                swept = True
            accel, contacts = self._accelerations(t, dt)
            if s % record_every == 0:
                ts.append(t)
                poss.append(self.pos.copy())
                ccounts.append(contacts)
                mouths.append(self.mouth_radius_now())
                stored.append(self.stored_frac())
                inside.append(self.inside_funnel_frac())
                smask.append(self.stored_mask())
                ferr.append(self.formation_error())
                fdv.append(float(self.ctrl.dv_spent.sum()))
                if ledger is not None:
                    form = self.ctrl.dv_spent - self.ctrl.dv_tension
                    for k, name in enumerate(names):
                        ledger.record(t, float(form[k] - last_form[k]), FORMATION, name)
                        ledger.record(
                            t, float(self.ctrl.dv_tension[k] - last_tens[k]), TENSION, name
                        )
                    last_form = form.copy()
                    last_tens = self.ctrl.dv_tension.copy()
            self.vel += accel * dt
            self.pos += self.vel * dt
            self._update_phase(t, contacts)

        # Final flush: `steps` rarely divides evenly by record_every, so the
        # last few steps' thrust would otherwise be dropped. The ledger holds
        # the true total; formation_dv[-1] is the sampled one and sits a hair
        # below it.
        if ledger is not None:
            form = self.ctrl.dv_spent - self.ctrl.dv_tension
            for k, name in enumerate(names):
                ledger.record(duration, float(form[k] - last_form[k]), FORMATION, name)
                ledger.record(
                    duration, float(self.ctrl.dv_tension[k] - last_tens[k]), TENSION, name
                )

        return FunnelResult(
            t=np.array(ts),
            pos=np.array(poss),
            contact_count=np.array(ccounts),
            mouth_radius=np.array(mouths),
            stored_frac=np.array(stored),
            inside_funnel_frac=np.array(inside),
            stored_mask=np.array(smask),
            formation_error=np.array(ferr),
            formation_dv=np.array(fdv),
            events=dict(self._events),
            captured=self._captured,
            net=self.net,
            pellet_radii=self.cloud.radii.copy(),
            n_nodes=self.net.n_nodes,
            box_radius=self.box_radius,
        )
