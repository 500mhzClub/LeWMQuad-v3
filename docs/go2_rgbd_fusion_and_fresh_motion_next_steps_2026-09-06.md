# From recovered RGB information to continuous novel-maze navigation

Latest: [full-controller fusion and measured-depth proposal integration are
implemented and audited](go2_fused_navigation_integration_result_2026-09-06.md).
1938tests pass. The recorded consumer reaches traversal on both textured tapes;
no new controller command has been physically executed. Use the
[fresh complete-mission plan](go2_fused_navigation_fresh_mission_next_steps_2026-09-06.md)
for the next action; the sections below retain the scientific requirements and
chronology. Do not redo completed fusion/appearance/interface implementation.

Progress: [fresh physical shadow motion and independent raw audit are complete](go2_rgbd_shadow_motion_development_v1_result_2026-09-06.md).
Both textured arms track all54frames; neutral stops accepting observations at4.3s.
All three fixed physical tapes complete; matched physics is bit-identical.
1909tests passed. This is one small-motion layout, not a navigation outcome or
uncertainty calibration. Next connect the fused sensor owner to the complete
discovery/return controller, including braking velocity, operator-local attitude
views and uninterrupted stopping-tail ingestion. Sections1/2's implementation
and first shadow collection are done; independent robustness/error validation
and section3's full-task execution remain outstanding.

The [appearance assay](go2_appearance_information_result_2026-09-06.md) is complete.
It justifies testing RGB-D constraints in fusion; it does not justify changing
the old80mm proxy budget, editing B, assuming known motion or claiming a policy.

## 1. Implement one explicit fusion owner

Add a new development sensor-fusion path; do not edit frozen
SetupVelocityIntegrator, MotionState, original memory or their launched sources.
Consume each calibrated RGB/depth/body/fast-gyro packet once at its original
epoch. Keep the same relative orientation history through startup, walking,
turning, braking and return. Never initialize translation from commands or truth.

Retain separate raw plane-depth and point-correspondence observations. If W
spans the plane-depth weak directions in previous-body coordinates, use
P=W.T@W and combine the measured plane projection with P times accepted RGB-D
translation. Explicitly test the agreement of (I-P) times RGB-D translation
with the plane observation before combining. If inconsistent, retain a conflict
and stop/observe; do not choose whichever measurement permits movement.
On rejected RGB, retain the original weak-space inertial propagation rather
than making missing displacement zero. Preserve the original reported depth rank
and annotate the additional conditional point constraint separately.

Both estimates share depth, gyro and images: do not treat them as independent
measurements and multiply away uncertainty. Transport previous global error and
initial-velocity sensitivity; good local displacement cannot remove past position
error. Maintain explicit conditional/uncalibrated status. Do not replace the
old uncertainty with the observed1.25mm maximum. Error assumptions must include
camera calibration, depth, rotation/lever-arm and mismatches—not just residuals.
Keep geometry queries conservative about missing or conflicting evidence.

Tests: full/partial plane ranks, complementary point constraints, common-sensor
correlation handling, projected disagreement, RGB rejection/occlusion/aliasing,
wrong sensor identities/clocks, packet mutation, fault latch, no truth or command
motion input, uninterrupted memory epoch and no accumulated-error reset. A
separate diagnostic can check interface replay, but cannot rescore B as a success.

## 2. Fresh physical motion, not another B rendering comparison

Implement a new collision-preserving visual adapter for the robot scene using
the tested MeshSet and optical readback paths. Keep original primitive collision
geometry, floor height, robot URDF, friction, gait gains and native solver
identities. Native readback must verify the *robot* and physical arena as well as
the new visual meshes; the static assay did not establish robot equivalence.
Camera placement must now follow the physically executed robot, never B poses.

Before launch, freeze a new layout/initial heading, independent appearance seed,
and fixed bidirectional forward/turn/braking sequence. Include motion not present
in A/B and sustained weak-plane intervals. Retain textureless and repeated
conditions with matched geometry, command schedule and reset conditions; do not
select seeds after seeing tracking outcomes. Record RGB-D/gyro/body/joints,
actual command acknowledgements, physical contacts/falls, zero tails and complete
outer-loop timing. Any native pose/contact guard is evaluator supervision, not
deployment sensing; declare interventions and do not conceal them as autonomy.

First run new fusion in shadow under an explicitly bounded acquisition envelope,
with a separate sensor-only consumer. Preserve every rejection and any divergence;
only then enable it for a new complete-task execution. Score displacement in
both observed and weak directions, accumulated drift, false acceptances, dropout
growth and conflict handling. Retain frozen response/fixed-joint/qdot baselines;
none already has a validated universal foot-sweep bound. Freeze any error model
on development data before separate motion/layout validation. Independent runs,
not overlapping windows, are the units for claims of generalization.

## 3. Complete the actual task

Connect new fusion to a single persistent observer/memory and the existing full
discovery/marker/return controller. Keep raw depth-rank provenance and factored
non-floor/ground/unknown checks. Reserve uncertainty for observation actions and
stopping; do not wait until a budget is exhausted to seek a new view. A supplied
start region may cover startup only—it cannot stand in for an observed maze map.
Require junction decisions, explored-frontier/backtracking bookkeeping, marker
evidence, verified return and actual stopping tail without pose/memory resets.

Leave the calibration arena. Record complete mission outcomes on independently
generated layouts, all collisions/falls, false home declarations, interventions,
uncertainty stops, length/time/effort and latency. Keep the original0/2 untouched.
Fix duplicated compute with reference-equivalence checks or explicitly redesign
sensor/control cadence; simulation timestamps are not real-time evidence.

For long missions, distinguish accumulated global pose uncertainty from relative
transport between views sharing an anchor. Validate explicit common-error
cancellation/relative-pose transport and eventual loop-closure constraints with
false-match challenges. Do not reset global pose uncertainty merely to keep ray
queries inside a local budget; local ray memory is not trusted place identity.

## 4. Test the scientific contribution, then the real platform

Once a common complete-task system functions, hold physical/sensor/controller
interfaces and compute budgets fixed across geometry-only, supervised and JEPA
representations. Separate predictive-training benefit from genuine online action
sequence rollout using no/one/multistep ablations, and separate those from memory
using memory-off/on comparisons. Current engineered high-level control plus a
learned response model is not a learned navigation policy; label it correctly.
Use independent training/development/evaluation layouts, multiple seeds and
texture/lighting/depth/IMU/occlusion robustness. Do not grant every added module
credit for an improvement from richer sensor information alone.

Bounded hardware work requires actual access, calibrated synchronized real RGB
and the declared other sensors, resolved gait/URDF warnings, latency measurements,
operator supervision and stop authority. No hardware success is inferred from
this simulator. Final-test custody stays external and sealed material remains
inaccessible. Completion still requires evidence for the full scientific goal.
