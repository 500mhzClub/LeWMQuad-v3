# Bounded rotation-to-geometry adapter V1

This is a saved-output engineering diagnostic, not another estimator, physical
trial, independent validation or navigation policy. Preserve the four frozen
pose histories and all 708 original coverage rejections from longer-motion
explicit-coverage-status V1. No model is rerun or tuned.

For a finite matrix accepted by the unchanged estimator's proper-rotation
predicate, compute Q=U V^T from its SVD. Require Q to pass the unchanged geometry
1e-12 proper-rotation predicate. Reject reflections or grossly invalid matrices;
do not turn arbitrary matrices into admissible poses. Do not alter stored R,
translation, keyframe references, gyro history or sensor uncertainty.

For every point x in a body shape, ||(R-Q)x|| <= ||R-Q||F ||x||. Bound ||x|| by
the farthest corner of the measured-joint body-frame support box. Add this
per-shape correction to the explicitly supplied physical error hypothesis;
unknown physical errors cannot become zero. Include an explicit floating-point
guard for the finite matrix calculation and outward rounding on scalar additions.
This is a bound on a numerical change, not on error relative to the true robot.

Query the same initial measured surface using Q and these increased allowances,
without changing the floor predicate or its camera/plane assumptions. This
diagnostic supplies zero ADDITIONAL physical shape error, exactly as its
predecessor did; retain that unvalidated limitation in every output. Compare
the adapted coverage to original statuses on every saved accepted state, and
then to native-pose coverage after the adapted predictions are saved. Record
all rejections, flips, first/full coverage frames and correction maxima.
Agreement with native diagnostics is not calibrated safe-clearance evidence.

Also diagnose startup from the existing frame-zero measured surface and actual
sensed joint configuration. Project the entire conditional floor-footprint
enclosures through the ACTUAL fixed camera and range limits; report why they
cannot be observed, without replacing RGB/depth or assuming unseen floor.
The current packet has gyro, specific force and joints, not a support/contact
modality. The current scene builder explicitly disables robot rendering; any
future downward/wider camera protocol must account for self-occlusion rather
than inherit transparent-robot floor observations. No hypothetical camera image
is generated or admitted by this diagnostic.

Bind this protocol, adapter, diagnostic runner and focused tests plus predecessor
launch/result/audit sources and inputs before execution. Exclusive output:
`.generated/go2_bounded_rotation_geometry_adapter_v1_attempt_001`.
No source edits after launch, silent retry, inherited-output overwrite, sealed
access, physics, model training, hardware actuation or command permission.

Next use the resulting common numerical interface when diagnosing task-relevant
body/surface and prospective actuation error. Freeze any fitting rule before NEW
reserved validation. Resolve sensor-valid startup initialization, then complete
closed-loop navigation/memory and matched JEPA/multistep/layout/seed/hardware
evidence. This adapter cannot close those scientific requirements.
