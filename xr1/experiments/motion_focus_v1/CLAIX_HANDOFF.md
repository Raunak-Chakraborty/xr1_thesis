# XR-1 Motion-Focused Muon/Optuna V1 — CLAIX Handoff

Status: DEVELOPMENT IN PROGRESS. NOT TRAINING-READY.

## Provenance

Base source: 0106f1a0e0dfbcbf18e9732f4924b8ccd9e937f0
Parent model: successful last-one checkpoint, step 48129
Target branch: thesis/xr1-motion-muon-optuna-v1
Action format: 30 x 60, first seven channels physically meaningful.

## Implemented

- Opt-in deterministic motion-focused training sample schedule.
- 120000 samples, designed for batch 24 and 5000 total optimizer steps.
- 255 training episodes; train and validation splits remain unchanged.
- Terminal action motion masking.
- Isolated translation-label anomaly masking.
- Conservative down-weighting of identified intermediate pauses.
- Opt-in first-5/next-5/remaining-20 flow-MSE emphasis.
- Last-one trainable parameter routing to Muon/AdamW groups.
- Fourteen component unit tests passed on the workstation.

## IMPORTANT: Remaining requirements before H100 training

1. Validate and install the experimental Muon implementation.
   The helper currently requires pytorch-optimizer==4.1.0.
   Real Muon update and optimizer-state restore have NOT been tested.

2. Verify actual last-one parameter names against the routing rules.

3. Create an experimental Lightning runner and optimizer/scheduler
   integration. Do not modify the proven production AdamW contract.

4. Restore the parent checkpoint's TRAINABLE MODEL WEIGHTS only,
   using the exact original pretrained base and verified provenance.
   Start a NEW optimizer and scheduler, not the parent AdamW state.
   Do not use the old checkpoint as an ordinary same-run resume.

5. Integrate Optuna with a small, bounded experimental search.
   Record objective values and checkpoint identity for every trial.

6. Verify the actual data batch, loss, gradients, masks, optimizer
   state, checkpoint saving and resume in a short H100 smoke test.

7. Ensure DDP sampler order, step accounting and experiment budget
   match the frozen schedule. The schedule is stored as global frame
   indices; the existing sampler may reorder the indices.

8. Use the frozen validation split for model selection.
   Keep the test split untouched.

## Known constraints

The existing production checkpoint format checks AdamW metadata.
It must not be silently reused as a Muon checkpoint.

The existing production training launcher is tied to the earlier
48k-step run and its replay contract. A separate experimental
launcher/configuration is necessary.

All current experimental changes are opt-in.
Training has not been started on this branch.

Do not copy model weights or raw datasets into Git.
