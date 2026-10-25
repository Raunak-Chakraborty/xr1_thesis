# Architectural and Project Decisions

## D001 — Base model

Use Xiaomi-Robotics-1 5B.

Repository:
https://github.com/Raunak-Chakraborty/xr1_thesis

Do not confuse it with Open-X-Humanoid XR-1.

## D002 — Force reference

Use:
https://github.com/Raunak-Chakraborty/forcevla_thesis

Reference only. No installation required.

## D003 — Development environment

Develop with Codex on Ubuntu laptop horus@war.

Use suitable GPU infrastructure for full-model tests
and fine-tuning.

Use the Doosan workstation for eventual robot tests.

## D004 — Architecture experiments

Compare:

- Force-free baseline.
- Force in state.
- Independent force encoder.
- Force-aware MoE.

Do not assume which will perform best.

## D005 — Reproducibility

Preserve source commits, training configurations,
normalization statistics, test evidence, and
checkpoint provenance.

## Future decisions

Append new entries with date, evidence,
decision, and rationale.
