# XR-1 Force Experiment — Current State

## Current phase

Phase 0: Finalizing Codex development infrastructure.

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

The context branch has not yet been published.

## Remaining infrastructure tasks

1. Commit and publish project context.
2. Establish safe synchronization of shared context.
3. Configure Codex development launchers.
4. Verify logging and fresh-session recovery.
5. Verify instruction scope in experimental worktrees.

## Research status

No force-integration implementation has started.

No force architecture has been trained or evaluated.

## Next approved task

Finish version-controlling the project context.

Do not begin architecture implementation yet.
