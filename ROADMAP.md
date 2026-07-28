# Roadmap

Living document — reorder, split, or drop items freely. "Not done yet" is
not "behind." Milestones roughly ascend in complexity; we are not committed
to this order.

- [x] **Milestone 0 — Project setup.** Modern Python tooling (uv, pytest,
      ruff), verified end-to-end.
- [x] **Milestone 1 — Single-satellite, single-debris toy loop.** Propagate
      one satellite and one debris object in a simplified orbit (two-body),
      visualize the trajectories, and have the satellite execute one
      deliberate maneuver toward the debris. Proves the propagate → decide →
      actuate → visualize loop works end to end. *Result: naive pursuit
      correctly **failed** — burning at a target ahead raises the orbit,
      lengthens the period, and drops you further behind (a: 6878 → 6941 km,
      separation 1798 → 1905 km). Motivated M2.*
- [x] **Milestone 2 — Relative-motion rendezvous (terminal).** Clohessy-
      Wiltshire two-impulse guidance closes from ~20 km behind and holds a
      1 km along-track lead, self-correcting against the nonlinear truth
      model. Held 0.98 km on 4 burns.
- [x] **Milestone 2b — Long-range phasing + handoff.** Chaser starts on an
      arbitrary coplanar orbit (different altitude/phase, thousands of km
      out): drifts unpowered to the cheap phasing window, runs a Lambert
      transfer to a 20 km standoff, velocity-matches, then hands off to the
      M2 terminal policy. Cut phasing cost 2780 → 65 m/s (vs a 56 m/s
      Hohmann floor) by waiting ~62 h.
- [x] **Milestone 2c — Whole-process animation.** Three-panel GIF
      (`uv run python scripts/make_animation.py`): ECI global (fixed), ECI
      tracking (camera follows the pair, auto-zoom), and the Hill frame
      (auto-zoom). Both bodies get a dotted ballistic forecast that snaps at
      each burn, plus distinct markers and vanishing trails. Phase labels
      state when the camera is stroboscopic vs real time. The tracking panel
      degenerates at close range into two dots on one line — an accidental but
      good demonstration of *why* the Hill frame exists: it subtracts the
      ~7.6 km/s of common motion that hides the relative geometry.
- [x] **Milestone 3 — Capture dynamics (single-chaser baseline).** The contact
      engine: a hand-rolled mass-spring membrane net, a cloud of 1–10 cm
      pellets, and a Hill-frame capture sim. From a 30 m standoff the chaser
      tosses a CW-aimed net, the drawstring closes, and it tows the bag away
      (`uv run python scripts/run_capture.py`): 30 pellets, captured at
      t=29 s, 77% retained flat through 80 s of tow, centroid moved 57 m.
      Proves the physics the constellation (M4) builds on. No PyBullet — see
      decisions below.
- [ ] **Milestone 3b — Launch-window targeting.** Given a spaceport lat/long,
      pick the launch time: the target's orbital plane sweeps over the site
      twice a day, and launching off-window costs plane-error delta-v at
      ~130 m/s per degree (v·dθ at 7.6 km/s) — far dominating phasing costs.
      Scope: insertion-state-from-(site, time) abstraction (no ascent
      modeling), RAAN as a scenario variable, 24 h launch-time sweep showing
      the two daily alignment notches. Subsumes the deferred plane-change
      item; the Lambert solver is already 3D, so scenario plumbing is most of
      the work. Phase timing stays with the drift-first planner (already
      near-free); the launch window's real payoff is plane alignment.
- [x] **Milestone 4 — Four-satellite funnel constellation.** The operational
      concept, working end to end (`uv run python scripts/run_funnel.py`;
      animation `scripts/animate_funnel.py`): the chaser stages 1 km *behind*
      the cloud, splits into four; three lead in formation holding a conical
      membrane funnel's **mouth** open (this is why a constellation is needed —
      in vacuum there is no drag to stream a towed net open, so formation flying
      does the job water does for a trawler); the fourth trails at the **apex**,
      carrying the storage box. The formation station-keeps while the funnel
      deploys, burns to close from behind, and sweeps through the cloud; debris
      glances off the energy-absorbing walls and is channelled down into the
      apex box. **The three satellites never let go — the mouth stays open the
      whole time.** Reuses the M3 engine (contact, pellets, CW) via a shared
      force core; new work is the conical membrane, the velocity-limited PD
      formation controller, the approach, and the one-way storage box.
      The membrane is a **tetrahedron** (3 flat faces, open triangular mouth)
      whose corners are the three mouth satellites, and the apex satellite *is*
      the collector - a compartment sized like the satellite itself.
      *Result:* 21/30 debris stored in the 1 m collector, funnel at 99% of its
      design length, mouth held open throughout, 3.2 m/s formation-keeping for
      the whole operation.
      *Physics note:* burning prograde to "speed up and catch from behind" is
      the M1 trap in CW clothing — `ẍ = +2nẏ` balloons you radially (a 2 m/s
      prograde burn drifts +539 m up in 500 s and still misses). The CW solve
      for a half-orbit transfer answers with a burn that is *purely radial-
      down*, arriving *purely radial-up* — the constellation catches the cloud
      by sweeping vertically, never chasing along-track.
      *Superseded design:* the first version closed a drawstring purse-net,
      which required the mouth satellites to release the rim. The author replaced it
      with the funnel-to-storage concept below — more physical, and it removes
      the cinch/release/regroup machinery entirely.
- [ ] **Milestone 5 — Fuel & power constraints.** Resource budgets feeding
      back into guidance decisions. Three stages:
  - [x] **5a — One ledger, then kilograms** (`src/orbdebris/propulsion.py`,
        `uv run python scripts/fuel_budget.py`). Every guidance module now
        records into a single `FuelLedger`, and Tsiolkovsky turns the total into
        propellant mass. *Result for the reference M2b→M4 mission:* **87.7 m/s,
        19.1 kg** of a 100 kg hydrazine load (Isp 220 s). Breakdown: getting
        there dominates — PHASING 65.3 m/s (74%) and the terminal 20 km→1 km
        transfer 17.6 m/s (20%) — while the entire four-satellite capture
        (split + sweep + tensioning + formation-keeping) is 4.7 m/s (5%).
  - [ ] **5b — Urgency sets the deadline.** Physically-derived urgency
        (collision flux × mass × residual lifetime) choosing `max_mission_time`,
        producing the cost-of-urgency curve.
  - [ ] **5c — Multi-target campaign.** One tank, N targets: the case where a
        budget actually *binds* and changes which debris you go after. Also the
        discrete decision problem M6 would learn on.
- [ ] **Milestone 6 — Autonomy / training.** Revisit classical vs. learned
      (RL) control now that the fundamentals and a working environment
      exist.

## Known simplifications (honest limits, future work)

- **Capture:** no net self-collision (fabric folds pass through themselves —
  this is why a shallow pocket leaks pellets and the deep 7x7 one does not),
  no real friction (tangential velocity damping stands in), no pellet-pellet
  collisions (dilute cloud), and gravity inside the capture region is the
  linear CW approximation.
- **Constellation:** the satellite-to-funnel attachment is a stiff tension-only
  bond, not a modelled boom; the apex storage box is a soft one-way constraint,
  not a modelled hatch or valve; the membrane's wrinkle model is an area-ratio
  gate, not true buckling.
- **Collection is 21/30, not 30/30.** The remaining nine are debris that missed
  the mouth on the pass or are still sliding down the walls when the run ends -
  not a modelling failure. Longer collect time and sweep aim are the levers.
  (The under-deployment that used to cap this is fixed: see the tensioning
  decision above.)
- **The collector is a soft one-way constraint plus padding**, not a modelled
  hatch with a door that opens and closes on command.
- **Orbits:** two-body only — no J2, no drag, no SRP. This is why
  station-keeping costs ~0.1 m/s/year here; real LEO station-keeping is
  dominated by drag make-up, and differential ballistic coefficients between
  chaser and debris would continuously regenerate the drift we null.
- **Scenarios:** coplanar (see M3b), non-tumbling bodies.

## Architecture decisions

- **Propagation:** dedicated propagator — numpy + `scipy.integrate.solve_ivp`
  (double precision) for two-body absolute orbits; Clohessy-Wiltshire (Hill's)
  equations for relative motion once satellites are close. PyBullet (or
  similar contact-physics engine) deferred to the capture-dynamics milestone
  only — not used for orbital-scale propagation (float32 precision + no
  inverse-square gravity model make it a poor fit there). Decided 2026-07-15.
- **Autonomy:** start with closed-loop classical guidance (recompute a burn
  each step from current relative state). RL deferred until after classical
  rendezvous + capture work — that's when its value (coordination, robustness
  to uncertainty) actually shows up, and by then we'll have a validated
  environment and a natural reward signal. Decided 2026-07-15.
- **Layered guidance stack:** Lambert for the global transfer (exact under
  two-body — it assumes the same dynamics our propagator integrates) and CW
  for the local terminal phase (a linearization, valid only within ~tens of
  km, kept honest by re-solving from the measured state). Long-range and
  terminal concerns stay separate, joined by a velocity-matched handoff at a
  20 km standoff. Decided 2026-07-16.
- **Phasing is drift-first:** fixing phase with fuel is ~40x more expensive
  than fixing it with patience, so the planner drifts unpowered to the
  natural window by default. `max_mission_time` is the knob that trades back
  toward speed. Scope is coplanar; plane changes deferred. Decided 2026-07-16.
- **Station-keeping controls drift rate, not position:** the Hill axes are not
  equivalent — an along-track offset is free (same orbit, same period, it just
  sits there), while a radial offset changes the period and drives *secular*
  along-track drift at 3*pi*da per orbit. So HOLD uses a per-axis deadband
  (tight radial, loose along-track) and prefers a small along-track "drift-null"
  impulse enforcing the CW no-drift condition `ydot = -2*n*x` over a two-impulse
  reposition. Repositioning buys back an along-track error that costs nothing;
  nulling the drift costs ~0.0008 m/s versus ~0.08 m/s. Measured over 50 orbits:
  station-keeping fell from 0.156 to 0.0002 m/s/day (~845x) *and* the radial
  hold tightened from ±26 m to ±0.6 m. Decided 2026-07-16.
- **Capture engine is hand-rolled, not PyBullet:** a mass-spring membrane in
  the Hill frame, integrated with semi-implicit (symplectic) Euler — the
  standard cloth-sim choice, stable for stiff springs where explicit Euler
  pumps energy in. Transparency over free contact features; the float32
  objection from kickoff does *not* apply at Hill-frame scales, so PyBullet
  stays available if friction-dominated wrapping ever demands it. Cords are
  **tension-only** (a rope that could push would make the net a trampoline and
  bounce debris off). Capture units are **metres**, converted at
  `CaptureSim.from_hill_state` — cloth/contact literature is SI. New engine
  cross-validated against the old: a free body in the capture integrator
  reproduces `relative.cw_propagate` to <1 cm over 200 s. Decided 2026-07-17.
- **The net needs a membrane, not just cords:** the target population is
  1–10 cm pellets and the mesh has ~1 m holes — a cord lattice is a sieve. The
  fabric surface is the grid's triangulation, with sphere-vs-triangle contact
  and reactions split barycentrically to the spanning nodes. The debris is a
  *cloud* whose centroid is exactly the point M2b's rendezvous targeted.
  Per-pellet contact stiffness `k_i = m_i*omega²` keeps a 2 g pellet and a
  1.4 kg pellet on the same contact timescale (a fixed k would put the light
  one outside the stable timestep). Decided 2026-07-17.
- **Capture is gentle and geometric, learned from three failures** (M3's
  single-chaser tossed net): (1) fast contact is near-elastic — 2.6 m/s batted
  pellets to ±200 m; approach at ~1.2 m/s with heavily damped (inelastic)
  contact so the membrane herds. (2) Trigger the drawstring on *geometry*
  (mouth ring has swept past the cloud centroid), not first contact — first
  contact fires on the nearest pellet and bags only the cloud's leading edge.
  (3) The bridle arrest is impulsive and flings the catch back out: soften and
  heavily damp it, and cinch the mouth to 5% so cm pellets cannot slip the gap.
  Retention went 7% -> 17% -> 77%. Decided 2026-07-17.
- **Contact damping is deliberately anisotropic:** high *normal* damping
  (inelastic — absorb the perpendicular impact, no bounce) with low
  *tangential* damping (glancing debris keeps sliding). Measured on a flat
  membrane: a 45° hit rebounds <10% normally while retaining ~85% of its
  tangential slide. This asymmetry is what makes a funnel channel debris rather
  than bat it away, and it is the M3 "grippy" tuning inverted. Decided
  2026-07-23.
- **Formation-keeping is a real, ongoing cost:** a rigid formation held across
  the velocity vector is *not* a natural CW motion (only pure along-track
  offsets are), so the three mouth satellites thrust continuously to hold the
  funnel open — **23 m/s/day** at a quiescent trimmed hold (measured properly in
  M5a; the earlier ~8 m/s/day conflated a per-run number with a rate) and 2.9
  m/s over a capture. Reporting that cost is the point, not a bug: it is the
  fuel answer to "why not just fly a net?" For scale, the *single* chaser
  drift-nulling at its 1 km hold spends 0.003 m/s/day — some 7500x less, so the
  formation is the expensive thing, not the orbit. The controller is a
  **velocity-limited PD** (position error commands a
  clamped cruise velocity; an inner loop regulates to it), because a plain PD
  saturates *outward* at large error and overshoots by v²/(2·a_max). Its slots
  must stay centred on the satellite centroid — an un-centred target is an
  unreachable fixed point and the controller thrusts forever (this bug bit
  three times: initial formation, regroup, and is why `centre_slots` exists).
  Decided 2026-07-22.
- **Capture is a funnel to a storage box, not a closing purse-net** (the author's
  redesign, supersedes the cinch): the three satellites *never let go*. The
  funnel is a permanent structure whose energy-absorbing inner walls channel
  debris to a **one-way storage box** on the apex satellite. Glancing impacts
  shed their small normal-velocity component and slide on; only near-
  perpendicular hits would bounce, which a shallow cone (~14° half-angle) and
  gentle post-rendezvous closing speeds avoid. This deletes the cinch,
  drawstring, rim-release and collapse-regroup machinery entirely.
  *Realism basis:* energy-absorbing impact fabrics are mature (Whipple/Nextel/
  Kevlar shields; Stardust captured comet dust at ~6 km/s in aerogel), and our
  mission makes it easy — rendezvous brings relative speeds to cm/s–1 m/s, so
  a 10 g pellet at 1 m/s carries ~5e-5 J. Note there is **no gravity** to pull
  debris down the funnel: migration is *sweep-driven* (the funnel scoops the
  cloud like a trawl), which is why an added "shepherding" thrust made things
  *worse* — it drove debris into the walls and stalled it mid-funnel (isolated
  test: 30/30 collected with zero shepherding, 17/30 with it). Decided
  2026-07-23.
- **The fabric must be a continuous membrane, not a cord lattice:** a
  tension-only cord goes slack under compression, so a cord-only funnel
  collapses and leaks (16–18/30 isolated, 1–4/30 in the full sim). Adding
  **constant-strain shell elements** (St-Venant–Kirchhoff, rotation-invariant
  Green strain) gives the surface in-plane stretch/shear/**compression**
  stiffness — a balloon skin — and collection jumps to 29–30/30 at *any*
  stiffness tested: it is the character of the stiffness that matters, not its
  magnitude. The rim also goes round (4.7–5.3 m all the way) instead of sagging
  to 1.5 m between supports. Two corrections raw St-VK needs: a **wrinkle gate**
  (folded fabric goes limp instead of storing absurd elastic energy — without
  it, deploying from a fold detonates) with a **small floor** standing in for
  bending stiffness (without it, folds lock closed forever). Decided 2026-07-23.
- **Every modelled force must be an action-reaction pair:** the storage box
  first pushed pellets inward with no reaction on anything — a force from
  nowhere that self-accelerated the assembly while the thrusters burned ~40 m/s
  chasing the phantom. The reaction now lands on the apex node the box is
  mounted to, and a test asserts total non-CW force is zero. Related: the box
  must only retain debris that has *entered* it (a one-way latch); constraining
  everything outside its radius made it a tractor beam that sucked in the whole
  cloud from 20 m away. Decided 2026-07-23.
- **The funnel is a tetrahedron, not a cone** (the author, from watching the
  animation): 3 flat triangular faces + an open triangular mouth, its corners
  held by the three mouth satellites. Its shape is then *exactly* the
  tetrahedron of the four satellite positions, so commanding the formation is
  commanding the geometry - and the concertina fold below becomes structurally
  impossible (a taut membrane can sag inward, never fold back axially). Mouth
  edges are straight lines between held corners, so the circular rim's
  sag-between-supports is gone, and the `n_sectors % 3` guard disappears
  because 3 corners is native. Sizing: corners at 6.2 m give the triangular
  mouth the same ~50 m^2 area as the old 4 m circle (a triangle covers only
  ~41% of its circumscribed circle) - and because the *inradius* (R/2) sets the
  face inclination, the area-matched tetrahedron presents ~11 deg faces where
  the cone presented 14: shallower, more glancing, exactly the capture
  condition. Decided 2026-07-23.
- **The funnel must be tensioned lengthwise** (the apex satellite's thruster
  pulls backward): nothing else stretches it. Without it the cone
  **concertina'd** - folding back on itself at mid-length, reaching 51% of its
  design length with its smallest ring 1.9 m *behind* the apex, while every
  radius still looked correct. Working half-angle was 25 deg instead of 14.
  With tensioning: 99% extension for ~1.1 m/s. `funnel_extension()` is the
  diagnostic that would have caught it instantly, and it is now a regression
  test. Note the order matters - tension *before* `trim_slots()`, or the trim
  accepts the folded shape as its target. Decided 2026-07-23.
- **Slippery chute, padded catch** (the author's fix for delivery): a funnel
  *concentrates* debris but, with no gravity, cannot *deliver* it - each pellet
  slides until the walls absorb its motion and then parks wherever it stopped,
  strung out along the cone (measured: 1/30 inside a 1 m collector, debris
  spread 0.8-14 m along a 16 m funnel). Pushing them apex-ward only wedges them
  into the narrowing walls. The fix is to move the dissipation: make the
  membrane nearly frictionless *tangentially* so debris keeps sliding home
  (tangent_zeta 0.05 -> 0.005: 8/30 -> 26/30 within 1 m in isolation, median
  distance 2.2 -> 0.4 m), and pad the *collector* so arriving debris is damped
  to rest there. Normal damping stays high throughout - that is what stops
  bouncing. Decided 2026-07-23.
- **Trim the controller to the structure it holds:** the membrane's settled rim
  radius sits slightly inside the as-built blueprint, so slots at the blueprint
  radius leave the controller leaning on the structure forever (~40 m/s of
  tug-of-war it can never win). `trim_slots()` re-trims each slot to the
  achieved geometry after deployment — 40 → 8.7 m/s. Real GNC trims to the
  shape it got, not the one on the drawing. Decided 2026-07-23.
- **Fuel accounting is one ledger of entries, never totals** (M5a): delta-v is
  *created* in three incompatible shapes — discrete ECI impulses in km/s
  (`simulate`), continuous Hill-frame thrust integrated per step in m/s
  (`FormationController`), and a planner estimate never reconciled against what
  was flown (`TransferPlan`). The fix is a *boundary*, not a rewrite: each
  source converts once when it hands delta-v to `FuelLedger`, canonical unit
  m/s. Totals are derived by query, so the headline number and the breakdown
  cannot disagree. Caveat recorded in the module: summing |dv| is right for
  propellant but is not the vector sum and is not invertible — no trajectory can
  be reconstructed from the ledger. Two source gaps were closed on the way:
  `ScenarioResult` recorded burn *times* but not magnitudes, so the terminal
  rendezvous cost (17.6 m/s, a fifth of the mission) was literally unmeasurable;
  and `TERMINAL` used to bundle the 20 km→1 km transfer with station-keeping,
  hiding that one is paid once and the other forever. Decided 2026-07-24.
- **Continuous thrust flushes, it does not stream:** the formation controller
  keeps its fast per-step accumulator (at dt≈1.5 ms we are not writing 100k
  ledger rows) and flushes the delta once per recorded frame, plus a final flush
  so the last partial interval is not dropped. Fine granularity where it is
  free, coarse where it is not — the same scale separation as analytic coast /
  stepped sweep. A test asserts FORMATION + TENSION reconstructs the
  controller's own total exactly. Decided 2026-07-24.
- **Firing order usually doesn't matter, except twice:** for one vehicle at one
  Isp, propellant from the *summed* delta-v equals replaying burns one at a time
  — the mass ratios telescope (`m0/m1 · m1/m2 = m0/m2`). It fails at the
  **split** (500 kg chaser → 4×125 kg: phasing is paid at the heavy mass,
  everything after at the light one) and for **per-phase attribution** (a
  subset's propellant depends on the mass when those burns fired). So entries
  are ordered and attributed, per-phase propellant comes from replay, and a test
  pins the telescoping identity so nobody "optimises" the replay away and then
  trips over the split. Decided 2026-07-24.
- **Never extrapolate a rate from a transient run:** the first loiter-crossover
  estimate read **1235 m/s/day**, taken by dividing a capture's formation-keeping
  by its 200 s duration — but that window is almost entirely deploy transient,
  slot re-trim, sweep burn and debris impact, all one-off. A dedicated quiescent
  hold gives **23 m/s/day**. And the same measurement re-exposed the trim bug
  from a new angle: *without* `trim_slots()` the quiescent rate is 199 m/s/day
  (8.5x too high), because a short run hides a constant-rate error inside its
  transient while a steady hold makes it the only thing left. Decided
  2026-07-24.
- **Power is a sanity check, not a subsystem** (M5a, decided with the author): the
  acceleration caps we invented correspond to 6.25 N (formation-keeping) and
  7.5 N (apex tensioning) on a 125 kg satellite — inside the range of real
  monopropellant thrusters (1–22 N classes are standard), though near the top of
  it, i.e. we assumed a generously actuated satellite. Nothing needed changing;
  it needed checking. **Electric propulsion is explicitly deferred**: at Isp
  ~1600 s it would slash propellant mass, but its ~0.1 N thrust cannot produce
  an impulsive burn, and every guidance module here (Lambert, CW two-impulse,
  the mission state machine) assumes impulses. Adopting it is a re-architecture,
  not a parameter change. Decided 2026-07-24.
- **Compute scale-separation, again:** the 47-minute half-orbit approach is
  pure CW flow, computed analytically; only the ~60 s terminal sweep runs in
  the fine contact sim. Same split the mission scale used (drift analytic,
  capture stepped). Decided 2026-07-22.
- **Two propagators, on purpose:** `dynamics.propagate` (numerical, general,
  the simulation's truth) and `kepler.kepler_propagate` (analytic universal-
  variable, same physics in closed form) for the thousands of coasts inside
  the planner's search. They are cross-validated against each other in
  `tests/test_kepler.py`. Decided 2026-07-16.
