"""Physical constants, in km / s units (the orbital-mechanics convention)."""

GM_EARTH = 398_600.4418  # km^3 / s^2
R_EARTH = 6_378.137  # km, equatorial radius

# Unit bridge. Orbital code works in km/s; the contact engine and the fuel
# ledger work in SI metres. Conversions happen at module boundaries, never
# scattered mid-computation - see capture.CaptureSim.from_hill_state and
# propulsion.FuelLedger.
KM_TO_M = 1000.0

# Standard gravity: the constant that turns specific impulse (seconds) into
# exhaust velocity (m/s). Not a local gravitational acceleration - it is a
# defined unit-conversion constant, which is why Isp in seconds works
# identically in orbit as on the ground.
G0 = 9.806_65  # m / s^2
