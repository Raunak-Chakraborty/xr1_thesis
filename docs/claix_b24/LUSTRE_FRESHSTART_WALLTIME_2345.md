# Experimental XR-1 wall-time correction

The intended fresh-start experimental CLAIX allocation is
**23:45:00**, not 20:00:00.

This change adjusts only the experimental allocation contract.

- Experimental Slurm wall time: 23:45:00.
- Experimental maximum allocation: 85500 seconds.
- Experimental runtime check: requires exactly 23:45:00.
- Safe-stop margin: unchanged at 1800 seconds.
- Approximate safe-stop point: 23:15:00 after allocation start.
- Strict production wall time: unchanged at 20:00:00.
- GPU: one actual NVIDIA H100, Slurm GRES gpu:hopper:1.
- CPU allocation: 12.
- RAM: 96 GiB.
- Account: thes2452.
- Partition: c23g.

The scientific training configuration is unchanged.

The experiment still initializes from original pretrained
Xiaomi XR-1 weights at optimizer step zero.

Training horizon remains 48284 steps, with verified
checkpoints every 250 steps and same-run resume support.

No historical checkpoint is used to initialize training.

This code change does not constitute GPU qualification.
The exact 23:45:00 request must be accepted by the actual
CLAIX scheduler before submission.
