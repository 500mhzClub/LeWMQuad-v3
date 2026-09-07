# Single-sample RGBD and actual visual-surface reference V1

This separately recorded successor addresses the source-backed failures in
[RGBD V1](go2_rgbd_observation_development_v1_result_2026-09-05.md). That study's
ten failed metric checks, raw data, sources and audit remain unchanged. This is
not a rescoring or retry of that root. Its new measurement claim is depth to
actual rendered surfaces, with physical/visual geometry disagreement explicit.

## Changes fixed before execution

Render RGB using its existing path, then depth-only without a physics step or
camera-pose change between calls. Verify the unchanged clock and camera transform.
Read actual OpenGL draw framebuffer, sample count/buffers, multisample-enable
state and pixel scale after depth rendering; require the single-sample target,
zero samples/buffers, multisampling disabled and scale1. Keep image calibration,
policy depth clipping .2–5 m, masks, causal reader/history and body sensors unchanged.
No segmentation, world pose, analytic wall queries or ground truth enters policy
depth. The floor identity described below is evaluation-only.

Record the native floor's visual mesh vertices/faces, actual visual and collision
poses, plane data and collision flag. Verify the two upward triangles covering
the1000-m square, allowing equivalent shared/unshared vertices, at visual z=-.005 m
and collision plane z=0. This offset comes from inspected installed mesh source,
not fitted outcomes; ten exact native renderer/mesh/entity source paths are bound.
Do not shift the scene, alter physical contacts, modify package files or compensate
policy depth using evaluator geometry.

The prospective visual-depth check uses the actual source-and-readback-verified
visual plane at-.005 m, unchanged boxes, native pixel centres, stride8,
2-cm box-edge exclusion, .22–4.98 m eligibility and5-mm maximum error tolerance.
At least1000 eligible rays,20 visible panel rays or zero panel/100 occluder rays,
and background-invalidity requirements remain unchanged. Also recompute and
report the original physical-floor-reference checks on these new captures;
passing visual geometry does not make the physical floor coincident or certify
clearance. Do not relabel those distinct claims.

## Fixed population, audit and next task

Two visible/occluded constructions and seed2026100400 remain identical to V1,
with new scene IDs/root only. Each settles1.5 s then captures five pairs at
1.5–1.9 s using zero commands:950 native physics/fast rows,95 ordinary rows.
The collection control body, gait and sensor wrappers are unchanged. Preserve
all observations and metric failures; no threshold fitting or retries in this root.

Before launch test separate calls/common clock, rejection of multisampled or
time-advanced capture, native-floor geometry including holes/false poses, distinct
visual/physical evaluation and unchanged collector/policy path. Full raw audit
retains RGB/body/fast/contact/gain/camera/mask checks and adds actual framebuffer
and floor identities, exact visual and separate physical-reference reduction.
All predecessor bindings are verified. Audit fidelity and sensor metrics are
separate outcomes; a failed metric does not become PASS because artifacts replay.

If the interface succeeds, next implement observed local surfaces/openings,
relative-motion observability and articulated turning-clearance state and connect
them to the existing continuous discovery-and-return controller. No further
stationary repetition should substitute for that work. Ideal simulated depth
is not installed or calibrated hardware; unseen volume, noise/latency/holes,
independent mazes, memory benefit and matched JEPA planning comparisons remain.
