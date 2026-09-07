# Fixed physical RGB marker acquisition probe V1

Scope: one fresh six-case stationary Go2 development acquisition study. This is
the beacon-perception interface for the planned whole-task prototype, not a maze,
return, learned-policy, JEPA or hardware qualification. No protected benchmark
access, predecessor edits, retries or threshold fitting on these frames.

## Fixed population and physical execution

Six cases in order: positive, absent, occluded, red_only, reversed, separated.
All use seed 2026100200, the same 5 m arena, actual Go2 body camera and the frozen
checkpoint gait with corrected gains. Four enclosing collision walls are retained
identically. The beacon pattern is two physical .08 x .25 x .50 m panels centered
at x=1.5 m, z=.55 m, y=+.14 m (red) and -.14 m (blue). Default inherited landmark
red/blue materials are used without texture edits. Collision remains enabled.
Absent removes only panels. Occluded adds an opaque .08 x 1.2 x 1.4 m wall at
x=1.05 m, y=0, z=.7 m. Red_only omits blue; reversed swaps left/right colors;
separated moves panel centers to y=+/-.60 m. Boxes use the existing wall-box
geometry carrier solely for construction and contact identity. No teacher route
is executed and none of that geometry enters the detector.

Settle under zero command for 1.5 s; capture at 1.5, 1.6, 1.7, 1.8 and 1.9 s.
Execute only four further zero-command 100-ms ticks (950 total 2-ms samples).
There is no repositioning, camera adjustment, follow-up stimulus or optional
capture on negative detection. Native contact/body limits stop physics immediately;
such a case is incomplete and cannot be reported as a clean perception probe.
Preserve partial outputs on infrastructure or sensor failure and stop the study;
do not restart this output root.

## Runtime definition fixed before actual captures

The declared marker identity is red-left/blue-right adjacent rectangular panels.
Current native RGB only: signed channel arithmetic; dominant channel >=16 and
at least twice each other channel. Four-connected components require width>=4,
height>=8, area>=64 and rectangular fill>=.75. A red/blue pair requires red left
of blue, nonnegative gap <= half the larger width, width ratio .5–2, height ratio
.7–1/.7 and vertical overlap >=.7 of the larger height. These are explicit
integration design constants, not calibrated semantic probabilities or distances.

Three consecutive 100-ms positive RGB observations register one identity. Absence
or a frame gap resets the current streak. Repeated detections do not count more
beacons; exact timestamp replay is idempotent, rewritten or noncausal input fails
closed. The detector receives no scene name, geometry, visibility/proximity flag,
beacon count label or destination image. Identical pattern copies are intentionally
the same observable identity; hidden semantic labels cannot distinguish them.

Expected development responses, not runtime inputs: positive has five positive
frames and a first discovery on frame 2; other cases have no detection/discovery.
Report mismatches unchanged, including partial/false detections and native stops.
No minimum success is inferred from completed observations or tuned afterwards.

## Evidence and verification

Bind sources, this protocol, inherited input/gait identities and package versions
before launch. Preserve actual RGB, camera extrinsics, policy-only histories,
raw physics, native contacts, physical static-object identity, actuator readback,
per-frame detector/discovery results and process logs. Verify ordinary source
imports with ignore-aware discovery; no source export or sealed access.

Audit all six cases: exact artifact/source bindings, 950-step zero-command tape,
native contact flags, sensor reconstruction/history causality, RGB pixel hashes
and fixed body-camera extrinsics, marker output replay and independent expected
case accounting. Physical object records must agree with the declared sizes,
materials and installed static collision entities. An audit PASS establishes
faithful evidence, not that expected scientific responses all passed.

Afterward proceed to continuous search/beacon/return integration with the existing
episodic hypothesis memory and a matched memory ablation. Preserve the current
narrow-maze false-arrival and scan-clearance failures. This stationary panel alone
cannot show initially hidden beacon discovery during exploration, reliable return,
appearance/dynamics generalization, JEPA contribution or real-platform transfer.
