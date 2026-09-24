# Fixed RGBD interface V1: actual-render metric and occlusion check

This is the single initial range-interface check required by the
[observed-geometry whole-task plan](go2_observed_geometry_whole_task_next_steps_2026-09-05.md).
It does not replace continuous navigation, qualify hardware, or establish JEPA
benefit. Keep the completed0/4 whole-task pilot and all predecessors unchanged.

## Observation contract and native source

Use the existing640x480 camera, horizontal FOV78.323 degrees and body mount
[.326,0,.043] m, zero roll/pitch/yaw. The scene builder converts horizontal FOV
to Genesis's vertical convention; native intrinsics and clipping are checked
on every capture. Call the actual rasterizer once with RGB and depth enabled,
no segmentation or normals. No analytic scene query generates policy depth.
The installed camera and pyrender source are exact-hash-bound at five explicit
paths in the runner; no native package export or broad materialization occurs.

Installed `Camera.render_pointcloud` unprojects with `(u+.5-cx)/fx`,
`(v+.5-cy)/fy`, and z=depth. The underlying depth-buffer reader flips image rows
and linearizes OpenGL z using near/far. Thus values are optical-axis metres,
not Euclidean ray length. The prospective separate sensor clips to .20–5 m,
with nonfinite/out-of-range pixels invalid and zero-filled. The renderer itself
retains its original .05–200 m clipping. Missing depth never implies free space.

RGB/body packets remain unchanged. A separate depth packet binds exact current
RGB pixels, episode, acquisition/availability/decision clocks and calibration.
A four-frame history requires10-Hz cadence; faults latch. Pixel unprojection
returns body-frame observed surface points and unknown pixels, never a whole
volume or clearance certificate. Initial acquisition is ideal zero-latency
simulation, explicitly hardware_calibrated=false. A real depth camera is not
assumed installed on Go2; deployment requires an actual sensor and calibration,
or this modality remains simulation-only. Keep RGB-only as a separate baseline.

## Fixed physical population and outcome

Two new trials reuse the visible and occluded marker construction specifications
under fresh IDs and paired seed2026100400. Keep all physical arena walls and
marker/occluder collision boxes. Remove only unused legacy route annotations;
use the verified geometry-free physical session, unchanged gait and sensor paths.
Settle1.5 s, then capture five colocated RGB/depth packets at1.5–1.9 s with only
zero commands. Each complete trial has950 native2-ms rows and95 ordinary sensor
rows, plus all native-rate gyro samples. No navigation controller is exercised.

Evaluate native depth against independent evaluation-only nearest intersections
with physical boxes and ground, at every eighth native pixel centre. Exclude
hits within2 cm of a box edge and expected optical depths outside .22–4.98 m
before evaluating the .20–5 m sensor. Require at least1,000 eligible rays and
maximum absolute error<=5 mm. The visible case must include at least20 interior
panel rays; the occluded case must include zero panel rays and at least100
occluder rays. The visible case must include background rays, all invalid at the
sensor's5 m limit; any background rays in the occluded case must also be invalid.
The close occluder covers the nominal camera frustum, so no background population
is required there. This distinction was checked analytically before rendering;
these criteria are fixed before actual collection and failures stay failures.

## Evidence, tests and continuation

Persist unmodified native depth separately from sanitized policy depth. Keep
actual RGB/body/fast histories, physics, contacts, gains, static identities,
camera transforms/intrinsics, source/native hashes and exact detector outputs.
Full raw audit reuses the predecessor RGB/body audit on its exact artifact subset
and separately checks every added artifact, fast measurement/history, depth
clock/calibration/mask/pixel binding and independent ray-reference metric.
Audit PASS and successful sensor metrics are different claims.

Before launch, tests must reject wrong units/pose/labels/clock/calibration,
out-of-range confident depth, protected/escaping paths and RGB/depth rewrites.
Check metric projection, optical-depth versus Euclidean-range confusion,
occlusion/background reference, unchanged collector body and sensor wrappers.
All ancestors remain bound. One fresh fixed root, no overwrite/retry or outcome-
based threshold fitting. After this interface check, implement observed opening
and clearance state and feed the existing continuous mission; do not add a
series of stationary probes in place of navigation. Noise, timing, occlusion
coverage, independent layouts, JEPA controls and real-platform evidence remain.
