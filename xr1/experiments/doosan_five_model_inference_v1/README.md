# Doosan XR-1 Five-Model Offline Inference v1

This experiment preserves the successfully executed workstation
evaluation from 2026-11-11: five_models_20261111_203437.

## Models

1. Original last-one FP32 compact model, step 48,129
2. Original last-two FP32 compact model, step 47,501
3. Motion-focused AdamW, 500 additional optimizer steps
4. Motion-focused conservative Muon/AdamW, 500 additional steps
5. Motion-focused faster Muon/AdamW, 500 additional steps

The final three share the same original last-one starting checkpoint.

## Evaluation contract

- 34 held-out validation episodes
- 476 inference predictions per model
- 2,380 predictions in total
- 136 fixed CLAIX validation windows
- 272 motion/rotation/quiet repeated predictions
- 68 gripper-release repeated predictions
- Instruction-only inference prompts without teacher-forced action tokens
- Paired episode-level bootstrap comparisons
- Validation split only; test split never accessed
- No physical robot commands or training updates

## Files

- run_five_model_eval.sh: exact successful workstation launcher
- exec/: five generated Python modules from the successful run
- evidence/selection.json: frozen windows and seeds
- evidence/summary.json: aggregate metrics and statistical results
- evidence/per_prediction_metrics.csv: paired prediction-level metrics
- evidence/REPORT.txt: human-readable scientific report
- SOURCE_SHA256SUMS.txt: source and evidence integrity

## Reproduction requirements

The launcher is workstation-specific, not a portable standalone
application. It requires the approved local XR-1 pretrained weights,
last-one and last-two checkpoints, three experimental checkpoints,
original Job 1 evidence, validation dataset and media, previous
workstation inference source, and xr1_inference_py311 environment.

The launcher regenerates its Python modules at runtime. The copies in
exec/ are retained as verified implementation snapshots.

Model weights, camera recordings, caches, datasets, and the full
prediction archive are deliberately excluded from this Git branch.

## Interpretation

These results measure offline action-prediction accuracy, not physical
assembly success. The small first-five-step advantage of conservative
Muon over AdamW was inconclusive in paired comparisons. AdamW had
better translation accuracy and gripper-release recall at 500 steps.

Both optimizers should be retained for controlled continuation tests.
