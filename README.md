# orbdebris

A simplified Earth-orbit simulator, built incrementally, for developing and
eventually training an autonomous small-to-medium space-debris collection
system. See [ROADMAP.md](ROADMAP.md) for where this is headed and
[CLAUDE.md](CLAUDE.md) for the working agreement.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management — a single Rust-based tool that replaces the
pip + venv + pip-tools combo, resolving and installing dependencies fast and
reproducibly via a lockfile (`uv.lock`).

```
uv sync        # creates .venv/ and installs exact locked dependencies
uv run pytest  # run the test suite
uv run ruff check .   # lint
```

`uv run <cmd>` runs a command inside the project's virtual environment
without you needing to manually activate it.

## Running things

```
uv run orbdebris                              # reference mission: drift -> Lambert -> CW hold
uv run python scripts/make_animation.py       # render mission.gif (three-panel animation, ~2 min)
uv run python scripts/phasing_tradeoff.py     # fuel-vs-time Pareto study -> phasing_tradeoff.png
```

The reference mission starts the chaser 100 km below the debris and 45°
behind, waits ~62 h for the cheap phasing window (~65 m/s total, vs ~2,780 m/s
if you refuse to wait), transfers, and holds 1 km ahead of the debris.
Scenario knobs live in `build_scenario()` in `src/orbdebris/__init__.py`;
see `ROADMAP.md` for milestones and the architecture-decision log.

Note: this machine needs `native-tls = true` (already set in
`[tool.uv]` in `pyproject.toml`) for uv to reach PyPI through a local
intercepting certificate authority.
