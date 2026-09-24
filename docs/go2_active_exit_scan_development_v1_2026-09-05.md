# Actual RGB/body active exit scan — fixed development protocol

Prepared before this study's first physical run. This is a new bounded development
experiment, not a retry, qualification or promotion of a predecessor.

Question: can a Go2 actively reveal local openings using only actual camera and
body history, and can it complete that scan within a junction without contact?
This precedes traversal/arrival association. No JEPA comparison or navigation
claim is made in this study.

## Frozen population and execution

Sixteen fresh local specimens: dead end, corner, tee and cross, fully crossed
with 0.9/1.2 m width and initial heading −0.15/+0.15 rad. Exact geometry and seeds
2026092500–2026092515 come from `scan_scenes()` and are written to launch.json.
These are familiar motif families, not independent held-out mazes. Geometry is
used for physical construction and evaluation only. Initial centering is an
initialization condition; no runtime pose or prepopulated map is supplied.

CPU Genesis, fixed gait and effective kp=20, kv=0.5. Settle 1.5 s at zero.
One continuous initial-body gyro reference tracks absolute targets π/2, π,
3π/2 and 2π; P=1.5, yaw command clipped ±0.35 rad/s, no commanded translation.
Each target requires heading error ≤0.08 rad and measured projected yaw rate
≤0.1 rad/s continuously for 0.3 s. Global scan deadline 30 s; terminal command
zero plus 0.5 s release. Native disallowed contact/body instability terminates
physics immediately, including during settling or release. Sensor faults
latch a failed scan and request the bounded zero-command release. No retry,
resampling, gate/threshold adaptation, rejected specimen replacement or
discarding failures. Infrastructure failures stop the batch and retain evidence.

## Runtime observation boundary

Every 100 ms control decision uses its actual native RGB packet and co-timed
body/control history. Transported specific-force feedback supplies a nominal
ground hypothesis (unchanged 2 s time constant and gates). Its uncertainty is
NOT calibrated; mean-force feedback was slightly better in the preceding
reused-route aggregate, so this choice does not assert estimator superiority.

Fixed palette-positive, bottom-connected floor pixels sampled at native stride8
are nominally ground-projected. A proposal needs ≥3 radial bands in 1.0–1.6 m
body-XY range, in each of ≥4 contiguous 2° bearing bins. No angular gap filling.
Negative evidence remains UNKNOWN. No metric clearance, body-volume viability,
stable place, exit identity, arrival, beacon or graph edge is inferred. This is
a renderer-palette baseline with a flat-wall maze prior, not general RGB vision.

Primary observations are the initial view plus the four dwell-completed target
views, including return. All available control frames are a separately reported
secondary population. All failed scans remain in all sixteen denominators.
No oracle pose/geometry enters controller, ground estimator or RGB proposals.

## Frozen evaluation and integrity

Physical scan success requires controller completion, four target views, no
physical/sensor stop or contact, final true relative heading error ≤0.12 rad,
maximum true XY drift ≤0.15 m, all 250 release samples, final 100 release samples
at planar speed ≤0.1 m/s and yaw speed ≤0.25 rad/s, and terminal body height ≥0.2 m,
absolute roll/pitch ≤0.5 rad. These are assay criteria, not a safety certificate.

Evaluation-only: transform each proposal center bearing with that frame's true
body rotation and intersect a ray from actual body XY with the central-cell
boundary at ±pitch/2. Count opening-directed versus closed-side-directed proposals;
origins outside the central cell, near-vertical rays and exact corner ties are
unscorable, not successful. Count unique known open sides covered per specimen,
including missed sides in failures. Report initial, selected-view and all-frame
coverage separately. A ray aimed through an opening does not establish that a
finite robot fits or can reach it; duplicate proposals do not create new exits.

Bind recursive ignore-aware ordinary source/test/protocol dependencies, fixed
gait and predecessor identity evidence before launch and verify after. Save every
native contact, physical sample, camera transform, RGB image, causal sensor
history, live ground/scan/proposal output and executed command. A source-bound
audit independently rebuilds sensors from raw physics, checks clocks/gains/slew/
contact termination/camera mount and reproduces every live decision, then
recomputes opening/physical outcomes. No renderer rerun or model selection.

Exact root: `.generated/go2_active_exit_scan_development_v1_attempt_001`.
If observation coverage is promising, the next distinct experiment must test
candidate-directed traversal and actual arrival evidence. If scans or proposals
fail, retain that result and identify whether motion, visibility, palette,
projection or angular support caused the failure before specifying a successor.
