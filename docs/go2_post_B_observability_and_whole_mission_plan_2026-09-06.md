# Post-B plan: recover motion observability and test the whole task

Update: the [fixed RGB-D correspondence diagnostic](go2_rgbd_correspondence_motion_diagnostic_v1_result_2026-09-06.md)
is implemented, tested and executed. All80A/B pairs were rejected; B's13weak
pairs contain zero keypoints. Independent replay reproduced every result.
The current scene disables textures; the optional texture branch also changes
collision primitive type. Follow the [controlled visual-information plan](go2_rgb_information_and_whole_mission_next_steps_2026-09-06.md)
next, retaining textureless/repeated-texture controls and unchanged physics,
then fresh motion/full-mission integration. Do not retune V1 to accept these
images or present rendering changes as a successful rerun of B. The numbered
requirements below remain the broader execution plan, with its first diagnostic
now completed negatively.

The objective is unchanged: defensible RGB-plus-deployment-valid-sensor JEPA
Go2 navigation in novel mazes, complete discovery/marker/return, memory benefit,
matched predictive-training and genuine online-rollout comparisons, independent
layouts/seeds/robustness and bounded hardware evidence when access permits.

[B's retained result](go2_action_motion_validation_development_v1_result_2026-09-06.md)
changes the next action. Its frozen supervised response improves body translation
but not articulated error versus fixed posture. More importantly, motion
observability is lost after a turn; the system exhausts its unchanged 80-mm
proxy budget. It must not be promoted to reliable local execution or maze success.

## 1. Use RGB to test the missing motion constraint

Implement a new, explicit causal RGB-D correspondence observer, initially as a
recorded diagnostic on A/B. Do not modify their frozen depth observer, reset the
failed controller, feed native pose into estimation, or rerun either trace.
Use co-timed calibrated RGB/depth and gyro with original hashes, episode and
availability clocks. RGB feature tracks with depth lifting can provide point
correspondences within a plane; plane-normal depth matching alone does not
observe translation tangent to that plane. This is a hypothesis to test on the
actual images, not an automatic full-rank label.

Use forward/backward tracking, valid depth, reprojection/3-D consistency and
spatially distributed matches. Specify all thresholds before inspecting the
diagnostic outcome; report every frame, rank, match count, residual and rejection.
Test planar textured translation, blank/repeated texture, wrong correspondence,
occlusion, depth discontinuity, clock mismatch, wrong camera/episode and movement
with zero target. Distinguish current/previous body and optical conventions.
Local OpenCV is available; no dependency installation or external data is needed.

Compare the frozen depth-only estimator with the new sensor-only estimate on the
exact B weak interval (4.1–5.3 s), native truth evaluation-only. These are now
development diagnostics, not independent validation. Test a subsequent fresh
trajectory/layout before claiming improved estimation. If RGB cannot supply
reliable constraints, retain missing directions and use an observation maneuver
or justify deployment-valid additional sensing. Do not add simulator contacts,
native pose or scene maps as onboard inputs.

## 2. Integrate continuous full-mission execution, not another fixed schedule

Build a new whole-task adapter around the same persistent observer/memory,
factored non-floor/ground/visibility evidence and existing task state machine.
Include an uncertainty-reserve monitor and explicit look/turn actions before
exhaustion, using sensed geometry rather than the successful sign from A as a
universal turn policy. A return to full-rank relative motion does not relocalize
old memory or remove accumulated drift; validated visual revisit constraints
must retain their provenance and uncertain cross-reference transforms.

Preserve the original complete discovery, marker and home-return criteria,
visited-place memory, backtracking, fault states and real stopping tail. Do not
count a few traversals as a mission. Native contact/geometry stay evaluation-only.
Use a fresh source/output and explicitly declare static-world/start/support
conditions for that mission. Never silently enlarge or extend A/B's local
[-1.25,1.25]^3, 8-s calibration prior to cover the maze. New observed space must
be supported by sensor evidence; unknown and contradictory space remain explicit.

Retain the frozen response and both persistence baselines; separate nominal
prediction from measured error and empirical margins from guarantees. A/B do
not establish continuous swept-volume, foot-support or hardware error bounds.
If new motion-model work is necessary, collect balanced command/turn/braking
coverage in development, freeze its rules, and evaluate on genuinely new data.
Do not refit to B and score B as independent evidence. Do not make a universal
safety certificate a substitute objective for testing complete simulated missions
under clearly stated, evaluation-checked development conditions.

Run the complete development mission to a real terminal outcome, diagnose its
actual failures and then expand to independent layouts and starts. The existing
whole-task paths remain 0/2 until a new full task is actually completed.

## 3. Make timing and scientific comparisons part of mission implementation

B's moving-loop mean cost is about 224 ms, including about 97 ms of controller
work. Profile shared depth preparation, redundant current-posture comparators
and repeated prediction-input validation. Preserve exact sensor evidence and
decision outputs on recorded reference cases; measure full moving loops after
optimization. If changing cadence, jointly rederive histories, command holds,
braking and uncertainty propagation—simulation-time 10 Hz is not wall-clock
deployment readiness.

With the same sensors, data, controller, capacity, training/compute budget and
seeds, compare geometry-only, supervised prediction and JEPA prediction. Isolate
no lookahead, one-step and true multistep candidate-action rollout separately
from training objective. Predicted states must propagate without future packets
and affect command selection. The present diagnostic forecasts do neither
navigation selection nor a JEPA contribution test.

Include memory-disabled/enabled runs with verified marker discovery and return.
Measure whole-task success, collisions/falls, false home declarations, path/time
and available effort proxies, interventions, uncertainty stops and complete
latency. Use independent layout-level replicates and intervals, not overlapping
forecast windows as independent trials. Challenge texture, sensor noise/bias/
dropout, starts and support/dynamics mismatch. Preserve old negative comparisons.

## 4. Deployment and completion

Resolve real sensor calibration/latency, gait/model warnings and operator safety
arrangements before a conservatively bounded physical Go2 experiment. No current
simulation result establishes hardware readiness. Keep final evaluation in an
external custodian-owned setting, inaccessible to model-facing development.

The goal is complete only with evidence for the full task and contribution
claims above. B's failed schedule, passing unit tests, a fitted response model
or a successful observation maneuver cannot satisfy that completion audit.
