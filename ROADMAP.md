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
      net's **mouth** open (this is why a constellation is needed — in vacuum
      there is no drag to stream a towed net open, so formation flying does the
      job water does for a trawler); the fourth trails at the **apex** cod-end.
      The formation station-keeps while the net deploys, burns to close from
      behind, sweeps through the cloud, purses the mouth shut, captures, and
      collapses to a compact formation to regroup. Reuses the M3 engine
      (membrane, pellets, contact, drawstring) via a shared force core; new work
      is the conical net, the velocity-limited PD formation controller, and the
      approach. *Result:* 28/30 pellets bagged, ~2–4 m/s formation-keeping over
      the whole operation.
      *Physics note:* burning prograde to "speed up and catch from behind" is
      the M1 trap in CW clothing — `ẍ = +2nẏ` balloons you radially (a 2 m/s
      prograde burn drifts +539 m up in 500 s and still misses). The CW solve
      for a half-orbit transfer answers with a burn that is *purely radial-
      down*, arriving *purely radial-up* — the constellation catches the cloud
      by sweeping vertically, never chasing along-track.
- [ ] **Milestone 5 — Fuel & power constraints.** Resource budgets feeding
      back into guidance decisions. *Groundwork exists: the planner's
      `max_mission_time` deadline already selects a point on the fuel/time
      Pareto front (see `scripts/phasing_tradeoff.py`). This milestone turns
      that into an explicit urgency/priority policy — how fast a given debris
      must come down vs. what its removal costs in fuel.*
- [ ] **Milestone 6 — Autonomy / training.** Revisit classical vs. learned
      (RL) control now that the fundamentals and a working environment
      exist.

## Known simplifications (honest limits, future work)

- **Capture:** no net self-collision (fabric folds pass through themselves —
  this is why a shallow pocket leaks pellets and the deep 7x7 one does not),
  no real friction (tangential velocity damping stands in), no pellet-pellet
  collisions (dilute cloud), and gravity inside the capture region is the
  linear CW approximation.
- **Constellation:** the satellite-to-net attachment is a stiff tension-only
  bond, not a modelled winch/boom; the "cod-end" is where pellets gather, not a
  sealed container; the regroup target is a fixed compact cluster, not an
  optimised transit formation.
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
- **Capture is gentle and geometric, learned from three failures:** (1) fast
  contact is near-elastic — 2.6 m/s batted pellets to ±200 m; approach at
  ~1.2 m/s with heavily damped (inelastic) contact so the membrane herds.
  (2) Trigger the drawstring on *geometry* (mouth ring has swept past the
  cloud centroid), not first contact — first contact fires on the nearest
  pellet and bags only the cloud's leading edge. (3) The bridle arrest is
  impulsive and flings the catch back out: soften and heavily damp it, and
  cinch the mouth to 5% so cm pellets cannot slip the gap. Retention went
  7% -> 17% -> 77%. Decided 2026-07-17.
- **Formation-keeping is a real, ongoing cost:** a rigid formation held across
  the velocity vector is *not* a natural CW motion (only pure along-track
  offsets are), so the three mouth satellites thrust continuously to hold the
  funnel open — ~8 m/s/day at hold, ~2–4 m/s over a capture. Reporting that
  cost is the point, not a bug: it is the fuel answer to "why not just fly a
  net?" The controller is a **velocity-limited PD** (position error commands a
  clamped cruise velocity; an inner loop regulates to it), because a plain PD
  saturates *outward* at large error and overshoots by v²/(2·a_max). Its slots
  must stay centred on the satellite centroid — an un-centred target is an
  unreachable fixed point and the controller thrusts forever (this bug bit
  three times: initial formation, regroup, and is why `centre_slots` exists).
  Decided 2026-07-22.
- **The cinch releases the satellites; it does not haul them:** closing the
  mouth by flying the three satellites inward whips the membrane and flings the
  catch back out (the M3 arrest lesson again). Instead the satellites release
  the rim and a gentle drawstring purses it shut — 67% → 90% retention. And the
  net deploys under a whole-formation **position hold**, or its deploy transient
  billows the funnel forward into the cloud before the approach burn. Decided
  2026-07-22.
- **Compute scale-separation, again:** the 47-minute half-orbit approach is
  pure CW flow, computed analytically; only the ~60 s terminal sweep runs in
  the fine contact sim. Same split the mission scale used (drift analytic,
  capture stepped). Decided 2026-07-22.
- **Two propagators, on purpose:** `dynamics.propagate` (numerical, general,
  the simulation's truth) and `kepler.kepler_propagate` (analytic universal-
  variable, same physics in closed form) for the thousands of coasts inside
  the planner's search. They are cross-validated against each other in
  `tests/test_kepler.py`. Decided 2026-07-16.
