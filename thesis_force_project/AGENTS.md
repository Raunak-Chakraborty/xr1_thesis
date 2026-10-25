# XR-1 Force Experiment — Codex Instructions

## Project objective

Investigate force as a first-class observation modality in
Xiaomi-Robotics-1 for contact-rich Doosan M1013 assembly.

The reference architecture is ForceVLA, which augments pi0
with force-aware multimodal fusion.

This project concerns Xiaomi-Robotics-1, not the unrelated
Open-X-Humanoid model also named XR-1.

## Workspace

Project root:
/home/horus/robotics_thesis/xr1_force_exp

Development repository:
xr1_thesis/

Read-only ForceVLA reference:
references/forcevla_thesis/

Independent experimental worktrees:
worktrees/

Project context:
context/

Experiment configurations:
experiments/

Reports:
reports/

Logs:
logs/

Scripts:
scripts/

## Permanent constraints

1. Preserve the original Xiaomi XR-1 architecture unless a
   specific modification is required and approved.

2. Preserve the existing Doosan state/action contracts.
   Doosan semantic actions are 7D and XR-1 uses 60D
   padded state/action interfaces.

3. Preserve the original force-free XR-1 baseline.

4. Do not change the ROS 2 teleoperation pipeline,
   robot controllers, or real-robot safety mechanisms.

5. Treat ForceVLA as read-only reference source code.

6. Do not install ForceVLA dependencies or download
   large checkpoints or datasets without approval.

7. Do not modify existing environments outside this
   project without approval.

8. Do not commit, push, merge, rebase, reset, or delete
   Git branches without explicit user approval.

9. Never silently overwrite existing files or experiment
   results.

10. Do not claim complete checkpoint compatibility
    based only on synthetic or CPU tests.

11. Keep development outputs within this project root.
    Existing Codex authentication and session storage
    under ~/.codex are managed by Codex itself.

12. Never store credentials, authentication tokens,
    private keys, or other secrets in project logs.

## Experimental architectures

A. Baseline: XR-1 without force.
B. State: 6D wrench through existing state conditioning.
C. Encoder: independent force encoder and DiT conditioning.
D. MoE: ForceVLA-inspired force-aware expert fusion.

Do not assume these architectures are already implemented.

## Session startup protocol

At the start of every new session:

1. Read context/PROJECT_STATE.md.
2. Read context/SESSION_HANDOFF.md.
3. Read context/REPOSITORY_BASELINES.md.
4. Read context/DECISIONS.md when relevant.
5. Inspect Git status for the active repository.
6. Identify the next approved task.

Do not repeat a complete repository audit unless necessary.

## Session completion protocol

After every substantial task:

1. Record the files changed.
2. Record tests executed and their actual results.
3. Record the exact Git branch and commit.
4. Update context/PROJECT_STATE.md.
5. Update context/SESSION_HANDOFF.md.
6. Record new architectural decisions in
   context/DECISIONS.md.
7. Identify unresolved issues and the next recommended step.

Distinguish verified results from assumptions.

Do not report tests as passing unless they were executed.

## Development methodology

Prefer small, reversible changes.

Use CPU-compatible tests and synthetic inputs for
initial laptop validation.

Defer full-model training, checkpoint loading, and
real-robot evaluation to appropriately equipped systems.

Never initiate physical robot motion from this project
without explicit authorization.
