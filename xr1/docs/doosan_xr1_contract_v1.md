# Doosan XR-1 Contract v1

Contract ID: `doosan_xr1_contract_v1`

This contract adapts Xiaomi-Robotics-1 to the single-arm Doosan M1013
while preserving XR-1's pretrained 60D state and 60D action interfaces.

The v1 training baseline is force-free. A force-aware extension is
reserved without changing the physical action space.

## 1. Fixed XR-1 internal dimensions

- state dimension: 60
- action dimension: 60
- action horizon: 30

The pretrained XR-1 state/action projector dimensions remain unchanged.

## 2. Cameras

Exactly two physical RGB streams are used:

1. `external_camera_2`
   - Intel RealSense D435i
   - fixed external workspace view

2. `tcp_camera`
   - Intel RealSense D405
   - wrist/TCP view

No synthetic third physical camera is introduced.

## 3. Force-free semantic observation

There are 19 informative physical state values:

- TCP position: 3
- absolute TCP rotation vector: 3
- measured gripper opening: 1
- Doosan joint positions: 6
- Doosan joint velocities: 6

Total informative state width: 19.

Physical conventions:

- TCP position: robot-base frame, metres
- TCP orientation: absolute base-to-TCP SO(3) rotation vector, radians
- gripper: normalized measured opening, 0 closed, 1 open
- joint positions: radians
- joint velocities: radians/second

## 4. XR-1 60D state packing

The first eight slots preserve XR-1's single-arm joint/gripper layout:

- 0..5: Doosan J1..J6 positions
- 6: zero compatibility slot for absent seventh joint
- 7: measured normalized gripper opening

Additional thesis proprioception:

- 8..10: TCP position x,y,z
- 11..13: absolute TCP rotation vector rx,ry,rz
- 14..19: Doosan J1..J6 velocities
- 20..59: zero padding in the force-free baseline

Thus the model receives 19 informative values while retaining the
native 60D XR-1 interface.

## 5. Reserved force-aware extension

The future force-aware profile keeps all force-free fields unchanged
and additionally uses:

- 20..25: external TCP wrench [Fx,Fy,Fz,Tx,Ty,Tz]
- 26..59: zero padding

This creates 25 informative state values:

19D proprioception + 6D wrench.

Force/torque values are observations only. They are never action
channels.

## 6. Semantic Doosan action

The physical semantic action remains exactly seven channels:

- 0..2: measured delta TCP translation
- 3..5: measured relative TCP rotation vector
- 6: absolute normalized gripper target

Translation:

`delta_p = p(t+1) - p(t)`

expressed in robot-base coordinates and metres.

Rotation:

`delta_R = R(t+1) @ R(t).T`

with `Log(delta_R)` represented as a 3D rotation vector in radians.

Gripper:

- 0 = fully closed
- 1 = fully open

This action definition intentionally matches the existing ForceVLA
Doosan semantic action contract.

### Exactly-once action-label rule

`doosan-forcevla-data-tools` is the sole owner of measured Cartesian
action-label derivation from consecutive synchronized TCP poses.

For processed row `t`:

`delta_p[t] = p[t+1] - p[t]`

and the relative rotation is computed once from `R[t]` and `R[t+1]`.

Therefore `actions.semantic` already contains a measured 7D action
delta. The XR-1 dataset/runtime path must never subtract neighboring
TCP poses or neighboring action rows again. A second difference would
change the target into an acceleration-like quantity and is forbidden.

The absolute gripper target is likewise consumed directly and must not
be converted to a gripper delta.

## 7. XR-1 action packing

XR-1 model-facing action remains 60D:

- 0..6: semantic Doosan action
- 7..59: zero / masked

Only dimensions 0..6 contribute to the Doosan action loss.

The output adapter slices dimensions 0..6 from XR-1 predictions.

## 8. Action horizon

XR-1 retains its native 30-step action horizon.

The converter emits one already-differenced 7D semantic action for
each valid consecutive synchronized transition. The XR-1 dataset
adapter selects future semantic-action rows and performs only
30-step windowing and repeat-last padding.

Horizon construction must not recompute any TCP delta.

## 9. Normalization

- Dataset files retain physical semantic units.
- State and action statistics are computed from training episodes only.
- Zero-padding dimensions remain zero.
- Inactive action dimensions remain masked.
- Normalization is owned by the XR-1 training/deployment path rather
  than permanently baked into raw converted data.

## 10. Relationship to ForceVLA

The same physical demonstrations may support both models.

ForceVLA v2 semantic state:

`[tcp_pos_3, tcp_rotvec_3, gripper_1,
  joint_pos_6, joint_vel_6, wrench_6]`

XR-1 force-free uses the same 19 proprioceptive quantities but packs
them into XR-1's 60D compatibility state.

XR-1 force-aware later adds the same six wrench quantities.

The semantic 7D physical action remains identical between ForceVLA
and XR-1 so architecture comparisons are not confounded by different
robot action definitions.

## 11. Ownership boundary

The XR-1 repository owns:

- 19D/25D semantic-to-60D state packing
- 7D-to-60D action packing
- 30-step windowing/padding of already-computed semantic actions
- action masking
- two-camera prompt/model adapter
- inference output slicing
- XR-1 dimensional validation

`doosan-forcevla-data-tools` will later own:

- MCAP decoding
- timestamp synchronization
- physical-unit extraction
- video materialization
- episode/frame indexing
- XR-1 exchange-data export generation
- exactly-once 7D measured-action derivation from consecutive TCP poses
- training statistics generation
- conversion provenance

The existing ForceVLA export remains unchanged.
