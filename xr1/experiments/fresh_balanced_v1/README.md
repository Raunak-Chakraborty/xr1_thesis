# Doosan XR-1 fresh-pretrained four-arm controlled campaign v2

Four **independent 5,000-step** fresh-pretrained fine-tuning arms, run sequentially
in one H100 allocation. No arm restores 48,129-step Doosan trainable tensors,
old optimizer moments, or scheduler states. All use the same verified released
XR-1-5B pretrained source, with separate optimizers initialized at step zero.

| Arm | Optimizer | Trainable final blocks | Samples |
|---|---|---:|---|
| natural_adamw_l1 | AdamW LR 1e-5 | 1 in each vision, language, DiT stack | Original frozen 120,000 draws |
| balanced_adamw_l1 | AdamW LR 1e-5 | 1 | Balanced 120,000 draws |
| balanced_muon_l1 | Muon LR 7.5e-5 + AdamW fallback LR 1e-5 | 1 | Same balanced schedule |
| balanced_muon_l2 | Same optimizer recipe | 2 | Same balanced schedule |

Train-only sampler: retain 50% original draws; route 15% of draws to per-episode
high translation, 15% to high rotation, 10% around the **opening** event,
and 10% to quiet/fine-alignment windows. The experiment starts with a held peg;
**closing actions are neither available nor required**. No validation/test
examples are sampled. For each original episode, its number of draws is kept
constant, and candidate pools are formed using the training trajectories only.

All arms: 255 training episodes, batch 24, horizon 30, 5,000 optimizer steps,
**500-step warmup**, cosine floor at 0.25 of base LR, seed 42,
BF16 forward with FP32 trainables, gradient clipping 1.0, weight decay 0.01,
motion-focus loss first5=2.0, next5=1.25, final20=0.75. The gripper loss
itself remains unweighted to isolate data sampling/optimizer/depth effects.
ModelCheckpoint saves full optimizer/scheduler states every 500 steps.

At 1,000 / 3,000 / 5,000 steps, evaluate the identical fixed 136 windows across
34 held-out episodes and 34 pre-release windows. Save per-window evidence,
release precision proxies and recall (including false-open fraction), and
aggregate motion, translation, rotation and gripper errors. Finalize with
**34-episode paired bootstrap** comparisons between A/B, B/C and C/D at each
milestone. Confidence intervals are exploratory (no multiple-test correction).

This is a **screening experiment**, not proof of robot task success. Compare
against the existing AdamW-500 / Muon-500 Doosan-adapted baselines on the
workstation before any real-robot deployment. Choose best checkpoints on
held-out performance, not merely the final step. A low-speed, monitored trial
with workspace, force and gripper safety checks is still required.

Resource request: one H100, 12 CPUs, 96 GB RAM, **13 hours**. Based on Job
5649588 (7,000 additional steps + validation in 3h27m), 20,000 steps require
~9h50m before the extra 2-block training cost, model startup, 12 validations,
and safety margin. Estimated ~10.5-12 hours; request 13 hours to avoid an
avoidable timeout. This is not a guaranteed runtime.

Source and schedules are committed in `thesis/xr1-fresh-balanced-v1`; only
train-only sampling schedules are rebuilt by the v2 workstation update. This
branch may be pushed before CLAIX deployment, without starting any GPU job.
