# Codex Session Operations

## Workspace

/home/horus/robotics_thesis/xr1_force_exp

## Initial architecture audit

Run:

./scripts/launch_codex_audit.sh new

This launches Codex with a read-only sandbox.

## Resume an existing session

Run:

./scripts/launch_codex_audit.sh resume

Select the relevant conversation in Codex's session picker.

Do not assume the most recent session is always the correct one.

## Session logging

Each session creates a timestamped directory under:

logs/codex/

The directory contains:

- before.txt
- terminal.typescript
- after.txt
- session_summary.txt

The terminal transcript contains the displayed terminal session.
It can contain ANSI control sequences and sensitive information.

Do not publish or commit raw session transcripts without review.

## Persistent handoff

Before ending a substantial session, update:

context/PROJECT_STATE.md
context/SESSION_HANDOFF.md

Record:

- Completed work
- Exact Git branch and commit
- Files changed
- Tests run and observed outcomes
- Unresolved problems
- Next approved action

Do not claim a test passed unless it actually ran.

## Working directory

Codex launches from the workspace root.

This ensures the workspace-level AGENTS.md is in scope.

## Reference source protection

The ForceVLA repository is reference-only.

Audit sessions use a read-only sandbox.

Development permissions will be configured separately,
after experimental Git worktrees are established.

## Session recovery hierarchy

1. Native Codex session resume
2. PROJECT_STATE.md and SESSION_HANDOFF.md
3. Git status and experiment reports
4. Terminal transcripts for supplementary evidence

A new session should not repeat a full repository audit
if prior validated findings already exist.

## Current restriction

Do not start architecture implementation yet.
