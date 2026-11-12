# Doosan XR-1 fresh-pretrained balanced-sampling screen (v1)

Four independent 2,000-step training arms in one sequential H100 allocation.
None restores the 48,129-step Doosan model's trainable weights, optimizer or scheduler:
all use the exact original released XR-1 base checkpoint pinned by the Doosan
`resolved_compact_config.json`, then create fresh optimizers.

| Profile | Optimizer | Trainable VLM/DiT final blocks | Sampling |
|---|---|---:|---|
| natural_adamw_l1 | torch AdamW, LR 1e-5 | 1 | natural frozen reference |
| balanced_adamw_l1 | torch AdamW, LR 1e-5 | 1 | event-balanced |
| balanced_muon_l1 | pytorch-optimizer 4.1.0 hybrid Muon LR 7.5e-5; AdamW fallback 1e-5 | 1 | same event-balanced |
| balanced_muon_l2 | same Muon configuration | 2 | same event-balanced |

Natural vs balanced AdamW isolates the data mixture; balanced AdamW vs balanced
Muon isolates optimizer configuration; Muon last1 vs last2 isolates capacity.

All arms: batch 24, horizon 30, last-layer FP32 master weights, BF16 compute,
AdamW betas 0.9/0.95, gradient clip 1.0, weight decay 0.01 from approved
motion experiment, 200-step warmup and cosine floor 0.25, seed 42,
`motion_focus_loss`: [first5=2, next5=1.25, remaining20=0.75].

The sampling manifest is generated using 255 TRAIN episode JSON trajectories and
an immutable 120k sample frozen schedule. It preserves exact task-group draw
counts and uses per-episode motion magnitude quantiles and release boundaries.
The two `.npy` schedules are generated once on the workstation and committed,
then verified against SHA256 on CLAIX. No validation/test data is used for
sampling. Validation remains 34 held-out episodes / 136 fixed windows, with
an additional 34 release-focused windows at the final step.

Checkpoints every 500 steps, validation at 1000 and 2000. The run preserves
complete optimizer/scheduler state and generates a small integrity-verified
scientific archive (excluding large checkpoints and raw media).

Provisional HPC time request: 07:00:00 for 8,000 steps, compared to measured
03:27:21 for 7,000 steps including two validations. Extra startup time,
additional validations and the 2-block arm are included in the margin.

**Scientific limitation:** offline MAE/release recall does not measure successful
real assembly. The best model must undergo controlled low-speed, force-limited
physical trials with a static Behavior Tree safety supervisor.

No existing training branch, model or data is modified.
