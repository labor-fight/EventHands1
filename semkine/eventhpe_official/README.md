# EventHPE official code (vendored subset)

Source: https://github.com/JimmyZou/EventHPE, commit 2af451f08205e442d92aa6bcd7c47285a59c5445 (MIT licence, `LICENSE`).
Zou et al., "EventHPE: Event-based 3D Human Pose and Shape Estimation", ICCV 2021.

Used by `semkine/cmp_ehpe.py` (CMP arm `ehpe`, docs/EXPERIMENT_RECORDS.md#cmp-amend7-20261110).

| file | from | content | edits |
|---|---|---|---|
| `model.py` | `event_pose_estimation/model.py` | `TemporalEncoder`, `Regressor`, `resnet50_encoder` | classes verbatim; `EventTrackNet.resnet50_encoder` moved to a module-level function (`self` dropped) |
| `geometry.py` | `event_pose_estimation/geometry.py`, `SMPL.py` | `projection_torch`, `rotation_matrix_to_angle_axis` (+ quaternion helpers), `rot6d_to_rotmat` (+ helpers), `delta_rotmat_to_rotmat`, `batch_rodrigues`, `quat2mat` | `delta_rotmat_to_rotmat`: the joint count 24 is inferred (`-1`) |
| `loss_funcs.py` | `event_pose_estimation/loss_funcs.py` | `compute_losses`, `compute_flow_loss` | joint count 24 and vertex count 6890 inferred from the tensors; the flow-validity threshold (official `> 4` px) is an argument with that default; one `.view` -> `.reshape` |

Not vendored: `EventTrackNet.forward` and `SMPL` (restated for MANO in `semkine/cmp_ehpe.py`), `VIBERegressor`, the
dataloaders, `flow_net/` (the CMP port uses the EvHandPose FlowNet of stage `epf`), metrics, rendering.
