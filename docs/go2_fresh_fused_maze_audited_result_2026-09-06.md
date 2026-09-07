# Fresh closed-loop maze: audited execution, failed mission

The first fresh sensor-controlled attempt advanced 2.137 m along its path, but
failed before completing its first local traversal. There was no hidden-marker
discovery, return, verified route edge, learned navigation policy, or JEPA test.
This is one development attempt, not generalization evidence. Earlier whole-task
0/2 and every predecessor negative result remain unchanged; do not pool unlike
protocols as matched trials. The scientific goal remains unachieved.

## Recorded and independently checked

Output: `.generated/go2_fresh_fused_maze_development_v1_attempt_001`.
Frozen protocol: [fresh mission V1](go2_fresh_fused_maze_development_v1_2026-09-06.md).

| Evidence | Result |
| --- | --- |
| Actual command selection | One uninterrupted RGB-D/body/joint/gyro controller; not a stimulus tape |
| Path / final displacement | 2.136708 m / 2.068983 m |
| Time from settled anchor | 21.8 s to terminal decision, plus 0.5 s zero tail |
| Captures and execution | 219 decisions, 218 active ticks, 5 tail ticks, 224 RGB-D frames, 11,900 physics samples |
| First nonzero request | 3.7 s absolute simulation time, decision 22 |
| Terminal status | `FAILED_LOCAL_FAILED_UNOBSERVED_TURN_VOLUME` |
| Contacts and supervision | No recorded disallowed contact, nonfoot-ground violation, native stop, or sensor fault |
| Stopping tail | 1.416 mm displacement; final 200 ms maximum XY speed 2.076 mm/s and yaw speed 0.000975 rad/s |
| Mission outcome | No marker discovery, no return, no trusted graph edge |
| Sequential control-cycle wall time | 203.350–350.152 ms, median 291.333 ms; all 218 measured cycles exceed 100 ms |

The underlying gait is learned. The navigation controller is engineered. Native
simulation guards externally supervise execution; they are not deployment-valid
policy inputs or a hardware safety certificate. A run without contact does not
validate the prospective body/foot sweep or braking model.

The recorded cycle timings overlapped a CPU regression run; they are not an
isolated deployment benchmark. No deadline compliance has yet been demonstrated.

The separate auditor reconstructed every controller decision and tail observation
exactly from saved sensor packets. It checked raw body/gyro generation, clocks,
camera mounting, native-depth conversion and geometric alignment, contact
attribution, all command/sample intervals, actuator/foot identities, setup support,
every additional native guard row, marker visibility and physical outcome metrics.
The terminal ray query also matched the independent reference implementation.
All 471 source, 5,402 input, native/OpenCV and 722 output bindings passed before
and after. The acquisition and original source were not changed or resumed.

Audit implementation:
`scripts/audit_go2_fresh_fused_maze_development_v1.py` and
`scripts/fresh_maze_turn_conflict_audit_development.py`.
Audit result: `raw_artifact_audit.json` within the output root.

Identities (SHA-256):

- Launch: `54c9e50bba03f6e74dce97d2338d96176b3436269066e1ad7deca94db5e8ab30`.
- Result: `5ede0552ccd31681f6cbb12ba441005dbf9e221289cba36c50546d2c31faa1f9`.
- Independent audit: `8275b9edb0cc5a201f07f3f10b86db51f3738390e1fdfcd9ac35686f95ecb064`.

## Causal failure: padded turn samples, floor support and pose envelopes

At terminal time, 267/932 nominal turn samples were unknown and all 267 carried
near-surface conflicts. The controller retained 39 keyframes plus its current
view. Thirty-one stored views supplied 5,688 conflicting sample/view pairs.
Every near-hit pixel in these pairs reconstructed to the physical floor within
1 mm in an evaluator-only camera/world calculation. None reconstructed to a
non-floor surface. This classification was not sent to the controller.

Of the 267 conflicted samples, 258 had the old generic ground-support role and
nine did not. All 5,688 conflicting pairs exceeded the old 60 mm
query-height-plus-pose-radius support allowance. Some also touched incomplete
windows or pixels without complete four-pixel floor cells; these are overlapping
failure counts, not separate physical obstacles. The current view itself supplied
no conflicts. Removing transport radii in a diagnostic-only calculation left
zero conflicts and zero unknown samples, with the same poses and sensor frames.
That calculation grants no motion authority and is not a validated repair.

The mechanism is explicit in the source:

1. `nominal_turn_volume` collapses observed articulated postures into padded
   gravity-height disks, losing individual physical primitive/contact identity.
2. Historical ray queries add current and stored global pose-error scales.
   They do not cancel shared anchor errors or otherwise model their correlation.
3. A floor return becomes a near-surface veto when an entire projected uncertain
   sample window fails the separate fixed support predicate. Any view's veto
   overrides positive support from all other views.

This is a conservative evidence failure, not evidence that the physical robot
could not turn. Nor does the absence of wall hits prove the future gait is safe.
The bottom nominal band is about -0.3534 m in gravity-aligned body coordinates;
its padding is not a physical foot penetration allowance. A box/point envelope
intersecting floor cannot be interpreted identically to an unpadded body collision.

Supplemental read-only pose comparison used the saved physics trace, initial
pose at sample 749, and each decision's sample index. For each decision it
compared fused position with `R_start.T @ (p_actual - p_start)`. Maximum error
was 0.535 mm; terminal error was 0.240 mm. Maximum attitude discrepancy was
0.000736 rad. Maximum interior depth discrepancy was 0.094 mm. These unusually
small errors concern ideal simulated sensing, not real-camera calibration.
The terminal uncalibrated position scale was 22.147 mm, below its unchanged
80 mm budget. Every motion interval had depth rank three: RGB point tracking
was available but was not needed to fill a weak depth direction in this run.
Therefore this mission does not isolate an RGB-fusion benefit either.

## What this establishes—and what it does not

A fresh eight-cell physical maze and marker-preserving appearance now work with
the actual sensor/control/physics loop and an independently replayable stopping
tail. The first local failure is reproducible and has a concrete geometric cause.
The 1,955-test pre-audit regression passed; 15 new synthetic audit corruption and
projection tests also passed. The combined regression then passed all 1,970 tests
in 159 explicitly selected files (163.24 s). Software checks do not replace
mission success.

Outstanding requirements include physically meaningful prospective action
evidence, justified sensor/relative-pose errors, successful branch exploration
and return memory, real-time execution, matched predictive-training and genuine
multistep-rollout comparisons, independent layouts/seeds/robustness and bounded
hardware evidence. [The next plan](go2_floor_factored_navigation_next_steps_2026-09-06.md)
keeps those requirements intact.
