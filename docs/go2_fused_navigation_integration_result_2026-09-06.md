# Fused state reaches navigation; appearance-specific exit failure addressed

Previous goal turn: **NO_PROGRESS**, a user-requested status inspection. This
continuation makes source changes, executes new sensor-consumer diagnostics and
independently reproduces their results. No new physics or hardware was executed
in this continuation; the preceding fresh physical shadow experiment is reported
in [its result](go2_rgbd_shadow_motion_development_v1_result_2026-09-06.md).

## Implemented

`lewm/rgbd_fused_navigation_development.py` connects the full discovery/return
controller to one RGBDInertialRayMemory owner. Region targets, floor observations,
heading, approach guidance and clearance queries use its admitted pose/evidence.
The original depth motion/rank remains explicit; no fabricated full-depth
translation is supplied. Braking reads labelled fused endpoint velocity in current
body axes. Single-use region consumption and shared attitude views avoid duplicate
integration. Scans have an operator-local rotation reference without resetting
the global gyro, memory, prior sensitivity or accumulated error. Actual terminal
tail observations have a separate sensor-only ingestion method; a failed estimator
never resumes and the mission cannot restart.

Synthetic end-to-end tests reach traversal with real depth/point/gyro processing,
including raw rank2 translation remaining null, and match the standalone fusion
exactly. Faults and unchanged uncertainty-budget exhaustion latch the controller
and memory. These tests do not establish physical execution or braking safety.

## Recorded-data evidence changed the next action

The initial recorded-data diagnostic V1 failed at its first comparison: Python
tuple-valued prior fields were compared directly with saved JSON lists. Read-only
diagnosis found only `initial_velocity_prior` differed; JSON-normalized values
were exactly equal. That failed launch/output/source remains unchanged. Separate
V2 corrects only serialization, with no tolerance or field omission.

V2 consumes four frames per appearance and stops at 1.8s with
`FAILED_INITIAL_NO_EXIT`. All fusion states match the frozen shadow output.
Every RGB floor mask is empty: the predecessor proposal detector requires greenish
floor chroma, while the independently textured scene is grayscale. Better tracking
alone did not produce usable navigation proposals.

New `lewm/depth_supported_exit_candidates_development.py` proposes extensions from
actual optical-depth surface points and the observed floor plane, using local
planar support and the same angular/radial proposal criteria. Invalid rays stay
unknown; vertical walls and different-height horizontal surfaces do not become
floor. It does not extrapolate a plane into missing depth. RGB still supplies
tracking, marker detection and appearance memory. This is a sensor-based geometric
proposal, not learned semantics, traversability or a JEPA contribution.

`lewm/depth_proposal_navigation_development.py` uses that provider for initial,
traversal and scan proposals. AST-equivalence tests prove that the two copied
decision methods change only their proposal-provider calls. No predecessor/global
function is patched. Existing turn-volume, target, timeout, arrival and return
rules remain unqualified development rules.

The new recorded-depth-proposal diagnostic gives:

| Appearance | Frames consumed | Last controller state | Nonzero proposed commands |
|---|---:|---|---:|
| Neutral | 29, including failed frame | Sensor-budget failure at4.3s during traversal | 6 |
| Repeated | 54 | Traversal, still running at tape end | 32 |
| Distinctive | 54 | Traversal, still running at tape end | 32 |

The blank-scene budget failure is preserved. All admitted fused states and raw
depth observations remain exactly equal to the physical tape's frozen shadow
outputs. **None of these new commands was executed.** Applied commands remain the
old fixed tape. This shows working interfaces and addresses the observed palette
failure; it does not show closed-loop motion, arrival, marker discovery or return.

## Verification and identities

- Initial focused suite: 11passed/1failed because a negative fixture mistakenly
  expected a new in-range depth perturbation to be rejected. Changed that test to
  an explicitly out-of-range depth; no runtime threshold changed.
- 81focused integration/regression tests passed (handle65391).
- Full suite1921tests/155explicitfiles passed161.69s (7882).
- Five serialization tests passed (87558);12depth-proposal tests passed (9648).
- Final full suite **1938tests/157explicitfiles passed167.32s** (20318).
- Separate saved-sensor auditor reconstructed all149logged frames across both
  successful interface diagnostics, including the neutral failed frame; all
  decisions, failures, floor counts and admitted fusion outputs match exactly
  (36074). All known handles are terminal. No tested-source edits occurred during
  its live regression run; the separately executed auditor is identified below.

Latest output:
`.generated/go2_depth_proposal_navigation_interface_development_v1_attempt_001`.

- `launch.json`: `d39e01184f1bc3526b0a32859ce2b6e9f2fa7f9b650643b1c47952ed8c324ea3`
- `result.json`: `434700298aab6afc7001d9247d3e708e0a3ff90628c51a3ef26ab0a75adb0e03`
- `interface_audit.json`: `72f408c118e028ea09ae9c97f4adff2486a60486dfe9a2e71e66cd76b898f79f`
- Auditor `scripts/audit_go2_fused_navigation_interfaces_development_v1.py`:
  `ab9655e8ab3704bf3b466ef2ae69a616169efff6e70717b0537162d49755dc09`.

Launch verifies464sources/5396inputs and the inherited native/OpenCV bindings;
all launched predecessors and new sources remain frozen. The audit binds both
diagnostic launch/result pairs and their six arm artifacts. The prior physical
582-artifact set is inherited input, not a new physical run. No sealed access,
new training, hardware, source export or benchmark promotion.

## Remaining scientific gaps

Whole-maze completion remains **0/2**. Neither JEPA benefit nor learned high-level
navigation has been established. The learned low-level gait is unchanged. Actual
fresh missions, prospective body/foot/braking checks, correlated long-duration
relative-pose transport, verified place/return association, noisy sensor validity,
real-time performance and independent matched learning/rollout/memory/hardware
comparisons remain required.

The next action is the [fresh-mission implementation plan](go2_fused_navigation_fresh_mission_next_steps_2026-09-06.md),
not another recolored arena replay. Source inspection found the physical initializer
hard-codes the short-motion arena, and the appearance builder renders all boxes
grayscale, including any marker panels. A proper fresh maze adapter must preserve
the discovery marker and isolate evaluator-only geometry from controller sensing.
