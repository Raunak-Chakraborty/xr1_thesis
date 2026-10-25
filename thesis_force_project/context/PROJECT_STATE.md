# XR-1 Force Experiment — Current State

## Current phase

Phase 0: Validating Codex logging, synchronization, and recovery.

Git repository and experimental worktree setup are complete.

## Authoritative repository

https://github.com/Raunak-Chakraborty/xr1_thesis

## Verified XR-1 baseline

afe739d1c9aac9218036a13fee9f9eb182747159

## ForceVLA reference

ef5ba4f5522d241fe72b559d2ccc331405f56b67

ForceVLA is a read-only source reference.

## Completed

- Codex CLI installed and authenticated.
- GitHub CLI installed and authenticated.
- XR-1 source cloned and verified.
- ForceVLA source cloned and verified.
- Project directory structure established.
- AGENTS.md and project documentation initialized.
- Read-only Codex audit launcher created.
- Four independent experimental worktrees created.
- Four experimental branches published to GitHub.
- Git push.default configured as nothing.
- Separate context branch/worktree created locally.

## Experimental branches

thesis/xr1-force/baseline
thesis/xr1-force/state
thesis/xr1-force/encoder
thesis/xr1-force/moe

All initially derive from the same verified XR-1 commit.

## Protected branches

main
thesis/doosan-xr1

Do not modify these branches during force experiments.

## Context branch

thesis/xr1-force/context

Purpose: Version-controlled project instructions,
decisions, handoffs, and reproducibility records.

The context branch was published to GitHub at commit d5eb18afe646f65deb288cb817eee7a02b6f6734.

## Completed infrastructure since initial publication

- Context branch committed and published to GitHub.
- Shared AGENTS.md and context links installed in all four experiments.
- Experiment-specific Codex launcher created.
- Timestamped terminal and Git-state logging configured.
- Context synchronization utility created and status-tested.
- No architecture implementation has started.

## Remaining infrastructure tasks

1. Publish updated Codex tooling and documentation.
2. Test the experiment launcher with a read-only Codex session.
3. Verify session logs and instruction loading.
4. Verify a fresh session can recover project state.
5. Confirm the development workflow before changing model code.

## Research status

No force-integration implementation has started.

No force architecture has been trained or evaluated.

## Next approved task

Review and publish updated project-context documentation and
Codex tooling. Then perform a read-only session-recovery test.

Do not begin architecture implementation yet.

Do not begin architecture implementation yet.
