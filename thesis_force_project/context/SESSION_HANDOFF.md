# XR-1 Force Experiment — Session Handoff

## Objective

Develop and compare force-aware Xiaomi-Robotics-1
architectures for Doosan M1013 contact-rich assembly.

## Workspace

/home/horus/robotics_thesis/xr1_force_exp

## Current status

Repository setup and experimental branch publication
are complete.

Codex launchers and context synchronization are configured.
Interactive session logging and recovery remain untested.

No architecture implementation has started.

## Authoritative repository

https://github.com/Raunak-Chakraborty/xr1_thesis

## Source baseline

XR-1:
afe739d1c9aac9218036a13fee9f9eb182747159

ForceVLA:
ef5ba4f5522d241fe72b559d2ccc331405f56b67

## Experimental worktrees

worktrees/xr1_force_baseline
  thesis/xr1-force/baseline

worktrees/xr1_force_state
  thesis/xr1-force/state

worktrees/xr1_force_encoder
  thesis/xr1-force/encoder

worktrees/xr1_force_moe
  thesis/xr1-force/moe

All four experimental branches are published on GitHub.

## Project context worktree

worktrees/xr1_force_context

Branch:
thesis/xr1-force/context

This branch is reserved for shared project documentation.

Published GitHub commit:
d5eb18afe646f65deb288cb817eee7a02b6f6734

## Protected repositories and branches

Never modify:

- GitHub main branch.
- thesis/doosan-xr1 baseline branch.
- ForceVLA reference source.

## Next tasks

1. Review and publish updated shared documentation and tooling.
2. Run the first read-only experiment Codex session.
3. Verify terminal logging and Git-state snapshots.
4. Verify project instructions load in the experimental worktree.
5. Test new-session recovery using the shared handoff.
6. Approve the first XR-1/ForceVLA architecture audit.

## Current Codex tooling

Read-only audit launcher:
scripts/launch_codex_audit.sh

Experimental launcher:
scripts/launch_codex_experiment.sh

Context synchronization:
scripts/sync_project_context.sh

Session logs:
logs/codex/

No development session has been executed yet.

## Session recovery

Read:

1. AGENTS.md
2. context/PROJECT_STATE.md
3. context/SESSION_HANDOFF.md
4. context/REPOSITORY_BASELINES.md
5. context/DECISIONS.md

Check Git status before making changes.

Use the latest verified handoff rather than
repeating completed repository setup.

## Important

No model implementation, training, or real-robot
testing is authorized by this handoff.
