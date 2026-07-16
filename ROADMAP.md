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
- [ ] **Milestone 3 — Capture dynamics.** Short-range contact/net physics
      (likely where a contact-physics engine like PyBullet enters), separate
      from the orbital-scale propagator.
- [ ] **Milestone 4 — Three-satellite constellation.** Multiple chasers,
      coordination, shared net.
- [ ] **Milestone 5 — Fuel & power constraints.** Resource budgets feeding
      back into guidance decisions. *Groundwork exists: the planner's
      `max_mission_time` deadline already selects a point on the fuel/time
      Pareto front (see `scripts/phasing_tradeoff.py`). This milestone turns
      that into an explicit urgency/priority policy — how fast a given debris
      must come down vs. what its removal costs in fuel.*
- [ ] **Milestone 6 — Autonomy / training.** Revisit classical vs. learned
      (RL) control now that the fundamentals and a working environment
      exist.

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
- **Two propagators, on purpose:** `dynamics.propagate` (numerical, general,
  the simulation's truth) and `kepler.kepler_propagate` (analytic universal-
  variable, same physics in closed form) for the thousands of coasts inside
  the planner's search. They are cross-validated against each other in
  `tests/test_kepler.py`. Decided 2026-07-16.
