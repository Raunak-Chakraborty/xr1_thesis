# Doosan / XR-1 Data Contract

Status: Preliminary; requires source verification.

## Robot

Doosan M1013 with SCHUNK gripper.

## Observations

- RGB camera observations.
- Robot proprioception.
- 6D estimated force/torque wrench.
- Language instruction.

## Existing compatibility targets

- XR-1 state interface: 60D.
- XR-1 action interface: 60D.
- Doosan semantic actions: 7D.
- Force-aware state: preserve existing adapter contract.

## Important sensor consideration

Estimated external wrench contains nuisance forces
from the robot, cables, and mounted equipment.

Do not assume raw wrench equals contact wrench.

## Pending verification

- Exact feature ordering.
- Units and coordinate frames.
- Synchronization.
- Normalization.
- Validity masks.
- Training/inference consistency.
