# Prospective measured-plane observer history diagnostic

Run one fresh paired observer diagnostic on the persisted extended-budget
development trajectory, observations 0 through 3837 inclusive. Stop earlier
if the candidate visual observer or its unchanged floor-registration layer
fails. Preserve that negative outcome. No retry, resume or replacement run is
part of this definition.

Original: `DualCameraVisualMotion`. Candidate: `MeasuredPlaneVisualMotion`.
Both start empty at frame zero, consume identical public primary/auxiliary RGB,
depth and body/fast-gyro packets, and maintain separate causal histories.
Both use an independent instance of the original
`MeasuredFloorTransportRegistration`. The original observer's complete evidence
must reproduce the saved evidence at every consumed frame; its floor evidence
must also reproduce before the known frame-3837 rejection. The terminal original
height-gate failure must reproduce if that frame is reached.

The candidate refines each originally qualified image pair using all its
accepted correspondences and the reference/current measured floor planes.
Every original inlier must retain the original two-image residual/reprojection
checks. Original reference/increment displacement and gyro gates remain, as do
the temporal conflict thresholds and ten-frame bridge limit. A qualified
image/plane estimate conflict is terminal even if inherited pair selection
would otherwise try another reference or camera. Missing count/extent support
uses the original qualified image fit without admitting a missing plane;
incoherent plane evidence is terminal. Initial floor support must be present.

Current floor candidate extraction uses the preceding accepted visual rotation
and the quiet initial specific-force direction; current depth supplies the
new normal. Gyro remains a rotation-consistency monitor. Store only planes for
the eight retained references and previous frame. Reference ownership, acquisition
clocks, depth hashes and selected refinement receipts remain explicit. No pose
reset, old-reference correction, global floor-gate increase or historical map
rewrite is permitted.

This is an observer diagnostic on a fixed, already executed trajectory. It
intentionally permits different estimates over the full recorded history.
It selects no commands, loads no learned model, replays no mapper or planner,
and establishes no hypothetical candidate navigation outcome after estimates
diverge. A later controller comparison must stop at the first changed command;
a later native experiment must execute the candidate's actual commands.

Inputs are provisional persisted development artifacts: the original native
worker may still be performing final checks. Bind the fixed launch, audit,
readout, closed decision stream, all consumed raw packet files, and all sources
before and after this diagnostic. Do not equate that binding with final native
completion admission or replace the existing native queue's verifier.

Admission requires the completed single-pass replay's exact verification
receipt and ended owner, at least 40 GiB available RAM, and 40 GiB reserved disk
plus a 512 MiB diagnostic-output allowance. One CPU replay may coexist with the
existing single native worker. CPU OpenCV/BLAS use one thread; OpenCL is disabled.
The output is an exclusive attempt under the fixed navigation artifact root.

Before launch, 31 focused component, observer and complete synthetic
observation-to-planning tests passed in 9.49 s. The original controller and
queued native sources are unchanged. The complete synthetic test exercises
actual image inference, memory, residual tracking and one fixed synthetic-model
selection; it is not a trained-model or native-navigation result.
