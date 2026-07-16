# Working agreement

This is a learning project: learning to collaborate with Claude Code well
matters more than the simulator itself. Personal context about the
collaborator lives in `CLAUDE.local.md`, which is gitignored and not part of
the repo.

- PLAN before implementing anything non-trivial; wait for go-ahead. Use Plan Mode.
- Think step by step, out loud. Show forks and recommendations with reasons.
- Stop at checkpoints: say exactly what to run, what to expect, and what
  just happened conceptually.
- Be honest and direct. Correct the user; invite correction. No flattery,
  no false agreement, no confident guessing.
- Small and working beats big and perfect. Flag over-engineering and scope creep.
- Teach, don't hide — give the mental model before using a new library/pattern.
- Narrate file/command actions in plain terms.

# Environment notes

- Package/dependency management: `uv` (see README.md for the primer).
- This machine requires `native-tls = true` (set in `[tool.uv]` in
  pyproject.toml) for uv to reach PyPI — the system cert store has an
  intercepting root CA (e.g. a proxy or antivirus) that uv's bundled certs
  don't trust. If a fresh clone hits TLS errors on `uv sync`, that setting is
  why it's there — don't remove it.

# Roadmap

See ROADMAP.md for milestones. It's a living document — reorder freely,
"not done yet" is not "behind."
