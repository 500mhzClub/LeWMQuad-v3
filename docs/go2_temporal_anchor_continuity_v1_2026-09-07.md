# Temporal anchor continuity V1: bounded measured increments

## Scientific purpose and distinction

The completed [measured-pose initializer comparison](go2_measured_pose_correspondence_result_2026-09-07.md)
returns473/7,433 poses versus7,400 for the original. End that initializer branch.
The current work instead separates two observation baselines: retained-keyframe
association and an immediately preceding-frame RGB-D measurement. It retains
the original descriptor matching and rigid-registration gates, not either failed
flow matcher. It is an engineered motion-estimation candidate, not a learned
policy, JEPA contribution or demonstrated navigation recovery.

Existing contact work was reviewed before implementation. The low-friction
contact hypothesis loses815/1,126 observations, while frozen visual observers
remain available on that separate short tape. Contact availability therefore
cannot be assumed during visual loss, and adding uncalibrated contact weights
would not solve the missing evidence. No contact or inertial translation is
introduced here.

## Fixed runtime semantics

`lewm/temporal_anchor_continuity_development.py` implements:

1. Validate the current RGB-D/gyro packet and exactly100ms separation from the
   previous accepted observation. Preserve the fixed initial coordinate frame.
   Run the unchanged original retained-anchor selection, with at most8 anchors.
2. Measure current pose from the immediately preceding accepted RGB-D frame,
   using the same original descriptor matcher, gyro-conditioned registration
   and per-step gates. When the anchor is that same previous frame, reuse the
   one measurement; do not count it as independent evidence. Otherwise there
   are at most9 candidate registrations per observation.
3. If both measurements qualify, require agreement within the existing recent-
   reference diagnostic thresholds:2cm and0.10rad. Select the anchor result,
   not a fitted blend. Reject contradiction. Agreement is correlated and cannot
   identify common depth/gyro bias or calibrate absolute pose uncertainty.
4. If the anchor qualifies but the incremental measurement does not, retain the
   anchor result and report the missing increment. Conflicting qualified anchor
   alternatives remain terminal and may not be overridden by an increment.
5. Only when retained anchors are unavailable may a qualified current increment
   supply a bridge pose. It is a newly measured displacement, not a commanded
   displacement, velocity extrapolation or last-pose hold. Keep all old anchors;
   never promote bridge-only poses into the retained-anchor bank. Retain one
   previous-frame reference for the next measured increment.
6. Permit at most10 consecutive bridge observations. This one-second operational
   allowance is fixed before recorded execution; it is not a derived error or
   safety bound. Record bridge length, cumulative measured bridge path and last
   previous-frame provenance. When an old anchor qualifies again, check current
   incremental agreement and use that measured anchor solution without changing
   coordinate origin. Report the preceding bridge span/path rather than erasing
   its history. Retained anchors may then be promoted by the original rules.
7. Missing both observations, measurement contradiction, exhausted bridge budget
   or invalid sensor/time/calibration/identity data gives a terminal failure.
   No restart, pose reset, unmeasured fallback or automatic motion permission.
   Historical estimates are not current between measurement timestamps.

The10-frame budget and2cm consistency check are engineering hypotheses, not
validated deployment limits. No new fitted covariance or error radius is claimed.
The external6cm arrival criterion, physical guards and recorded failures remain
unchanged. This module is not integrated into a physical controller.

## Completed tests and their limits

- Initial runtime16112:20 tests passed in8.05s.
- Expanded integration46656:79 tests passed in17.77s, covering22 new tests plus
  the frozen multi-reference, visual-led and measured-pose test files.
- Final runtime43166:23 tests passed in8.79s after adding an explicit common-
  depth-bias counterexample. These are not one combined80-test invocation.
- Four actual rendered multi-frame geometry cases preserve the original anchor
  poses and reference selections exactly, with no bridge required. They include
  different depths, signs, translation axes and combined rotation/translation
  with a nearer panel. Fixture truth produces images/depth only.
- Explicitly injected anchor unavailability exercises measured increments,
  retained-anchor preservation,5-frame rejoining, finite budget and terminal
  behavior. These injections test control flow; they are not naturally occurring
  tracking recoveries or independent scene evidence.
- Fault tests cover contradictory measurements/anchors, missing measurements,
  clocks, identity, image binding, calibration, blank imagery, privileged fields,
  bounded reference storage and returned-copy isolation. Zero commanded velocity
  with actually moving rendered images still yields measured bridge translation.
- A static-scene test deliberately biases reported depth by2mm per observation.
  The model returns more than3mm of false translation while anchor/increment
  disagreement is below1mm. This is a retained limitation, not a passing accuracy
  claim: both observations can share the same sensor error.

Source SHA-256:
`e0af3433738d2e0cac84d414bfe797222e5096b01762304b40f1c11ca0bec910`.
Runtime test SHA-256:
`696ae40199db26863c830653f724710d615cfc52a8d05d5efa1568cbedf8b77b`.

## Next executable experiment—not yet launched

Implement `scripts/replay_go2_temporal_anchor_continuity_v1.py` and focused runner
tests. Inherit and verify the completed696-source replay and its raw/native/
original-witness bindings. Bind the new runtime, tests, runner, runner tests and
this protocol before execution. Do not alter the existing771-source collector,
786-source planned learning study,696-source completed replay or predecessors.
Run the expanded explicit regression before any recorded comparison.

The single planned output is:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_temporal_anchor_continuity_v1_attempt_001`.

Use the existing development interpreter, PYTHONDONTWRITEBYTECODE=1,
PYTHONHASHSEED=0, PYTHONPATH=.:lewm_genesis:lewm_worlds and OMP/MKL/OPENBLAS/OpenCV
threads1. No GPU, physics, learning or hardware actuation. Preserve the2GiB
output allowance,40GiB reserve, exclusive durable artifacts and bounded rows/
metadata. A terminal execution fault retains partial artifacts with no retry,
resume or replacement attempt.

Compare original and temporal-anchor observers on all six inner-arrival/intent
recordings:7,433 frames,14,866 arm-observations including terminal no-ops. Reproduce
every original pose/selection/failure against its authenticated witness. Do not
rerun either failed flow method, tune bridge duration on exposed failures or
select different methods by trajectory.

Persist all six complete sensor streams before reading native coordinates.
Each candidate row must retain continuity evidence and reference selection,
including conflicts, missing increments, bridge spans and successful rejoining.
Evaluate exact native timestamps/samples, pose and incremental errors, all
availability categories, first failures and shared-frame comparisons. Report
the accuracy and availability of bridged frames separately, not just overall
means. Preserve every anchor/increment contradiction even when the original
would have continued. Rejoining is a measured anchor correction, not a verified
place identity or global loop closure.

Timings include the extra current-frame registration but exclude packet loading/
control unless explicitly measured. Do not call terminal no-ops a speedup.
Verify source/input/output bindings and independently reconstruct row counts,
provenance, errors and summaries after completion.

## Decision and the full unfinished goal

A positive reused-data result only motivates independent scene, sensor-error,
longer-duration drift and complete latency challenges, followed by a separately
frozen fresh closed-loop test. A negative result must remain negative; do not
search bridge budgets or weaker consensus gates to recover it. Neither result
qualifies deployment or changes the latest0/3 room returns.

Continue the original independent-layout collector. Only all12 successful
receipts permit the reviewed36-fit matched JEPA/supervised/input-ablation study.
Then test useful predictive training and online rollout separately, with the
same sensors, gait, local executor, action vocabulary, memory and budgets.
Reuse the existing memory event interfaces for physically executed branch
choice/backtracking/home association after local execution is dependable.
Independent-maze and training-seed evidence, deployment-valid calibrated sensing,
self-occlusion, unpaused real-time operation and bounded hardware evidence remain
required. This continuity implementation does not complete those aims.
