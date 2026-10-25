# XR-1 Force Experiment Plan

## Phase 0 — Infrastructure

Set up Git repositories, persistent context,
Codex logging, and session recovery.

## Phase 1 — Architecture audit

Inspect Xiaomi XR-1 and ForceVLA source code.

Determine exact model integration points.

## Phase 2 — Common force interface

Define force modes and preserve baseline compatibility.

## Phase 3 — Implementation

A: Existing state pathway.
B: Independent encoder.
C: Force-aware MoE.

Develop in isolated Git worktrees.

## Phase 4 — Laptop validation

Run CPU-compatible unit tests, tensor tests,
and gradient tests.

## Phase 5 — GPU feasibility

Verify full checkpoint loading, forward/backward,
VRAM consumption, and throughput.

## Phase 6 — Fine-tuning

Use controlled comparisons with matched data,
training budgets, and evaluation procedures.

## Phase 7 — Real-world evaluation

Evaluate vertical and angled insertion using
the Doosan workstation.

## Rule

Do not proceed to expensive experimentation
without passing appropriate earlier gates.
