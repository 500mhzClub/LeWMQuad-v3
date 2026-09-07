# Next: resolve floor/pose evidence, then complete the scientific mission

## Current next action: finish continuous return, then maze science

The [coupled feedback assay](go2_coupled_room_return_result_2026-09-06.md) is
complete and independently audited: **0/3 returns**, with both nominal runs
reaching the final home stage and **12/13 native-passing local holds**. Signed
quarter- and half-turns pass in both nominal runs. The remaining failures expose
single-reference visual support, clipped return-heading intent and bounded
planning limits; they are not full navigation success. Full regression passes
2,476 tests across 195 files.

Follow the [continuous-return completion plan](go2_continuous_return_completion_next_steps_2026-09-06.md).
Keep the current failure records and scoring criteria. Test bounded multiple
visual references without lowering gates, preserve travel heading for clipped
return subgoals, and then test actual fresh closed-loop returns. A separate
[pulse-timed JEPA interface](go2_pulse_timed_jepa_interface_2026-09-06.md) now
represents partial action blocks truthfully, but remains untrained and unused
by the robot. Live branch/marker mazes, useful memory, matched JEPA and rollout
studies on independent layouts/seeds, real sensing/timing and hardware remain.

## Preceding joint pose/action feedback plan

The [continuous room-return assay](go2_room_return_pulse_result_2026-09-06.md)
is complete and independently audited: **0/3 full returns**, 4/5 native-passing
local holds, 4,448 exact runtime replays and 4,478 passing depth checks.
Actual continuous raw-sensor ownership and multi-leg execution now work in
simulation, but turn-induced translation defeats the separable position/yaw
controller. A wrapped orientation goal also loses the sign of a requested
half-turn. Preserve the low-friction visually accepted/native-failing hold.

Follow the [joint pose/action plan](go2_coupled_pose_action_planning_next_steps_2026-09-06.md).
An offline empirical pulse rollout baseline now composes XY and yaw and
distinguishes orientation from signed winding; it is not JEPA or a tested
physical controller. Full regression passes 2,443 tests across 193 files.
Next implement one-pulse-at-a-time joint feedback and freeze a new physical
multi-leg return study, then live observed-branch/marker maze missions and the
matched memory/predictive-training/online-rollout comparisons. The ultimate
scientific goal, sensing/timing limits and hardware requirements remain intact.

## Preceding continuous execution and useful online memory integration

The [continuous pulse executor and episodic bridge](go2_continuous_pulse_memory_integration_result_2026-09-06.md)
are implemented, with sensor-anchored goals, uninterrupted frame/fault/budget
lifetime, live visit/attempt handoffs and a distinct .45yaw session boundary.
All864prior local decisions replay exactly;2,420tests pass across191 files. No new physics or memory benefit
is established. Next is raw-sensor runtime ownership, a fresh bound initializer/
collector and actual multi-leg/maze-turn/return execution, with independent
scoring and explicit fixed-lookahead/metric-return limitations.

The [goal-region pulse successor](go2_goal_region_pulse_servo_result_2026-09-06.md)
completed3/3 nominal local move/turn/hold tasks with independent full-second
physics checks; the low-friction challenge remains0/1. Raw audit passes all864
decisions and874depth checks;2,388tests pass. Nearby starts in one controlled
floor scene are not broad reliability or novel-maze evidence. The previous
[pulse-feedback0/4 result](go2_pulse_feedback_servo_result_2026-09-06.md) remains.

Follow the [continuous-execution/online-memory plan](go2_local_execution_to_online_memory_next_steps_2026-09-06.md):
parameterized sensor-anchored goals without pose resets, live visit/attempt
memory, actual maze-scale turns/returns and full connected-maze missions.
Resolve the old maze-session .35yaw cap versus measured .45bank command in a
distinct declared interface. Then matched memory/predictive-training/online-
rollout studies on independent layouts/seeds, realistic sensing/timing and
bounded hardware. Do not substitute further repetitions of the fixed local
task for the full scientific aim. Robustness and optical/terrain gaps remain.

## Preceding bounded pulse feedback and complete local execution plan

The [command-pulse characterization](go2_command_pulse_response_result_2026-09-06.md)
completed all four episodes and 64 pulse/braking events without native stops.
Raw replay passed for 1,548 decisions; 2,348 tests passed. Bank commands produce
useful but variable net responses, and a short forward pulse can still end
backwards/sideways under low friction. Stopping may undo or extend rotation.
Follow the [pulse-feedback execution plan](go2_pulse_feedback_execution_next_steps_2026-09-06.md):
finite supported drive pulses, action-specific sensor feedback and actual-goal
hold verification, frozen before several fresh complete-task trials. Then
persistent place/branch memory, full maze missions and matched JEPA predictive/
online-rollout/memory studies remain. Collection completion is not task success.

## Preceding command-support and response collection plan

The [bounded goal-hold trial](go2_goal_hold_and_gait_command_support_result_2026-09-06.md)
is complete: 0/2 full sequence successes, raw audit PASS, 2,327 tests passed.
Nominal stalls at a tighter internal correction margin despite entering actual
final pose tolerances; low friction reaches forward braking too late. The saved
gait training bank contains discrete commands, and ALL nonzero commands in the
new trial lie outside that bank. This command-support mismatch changes the
immediate priority. Follow the
[command-supported execution plan](go2_command_supported_execution_next_steps_2026-09-06.md):
bounded supported/small-command pulse and braking-response collection, then
finite correction/actual-goal verification and fresh physical validation. Do not
keep tightening margins or treat numerical command ranges as trained support.

## Preceding action-conditioned stopping and goal-holding plan

The [course-aware successor and supervised response fit](go2_course_response_and_control_result_2026-09-06.md)
completed. The fitted predictor does not transfer reliably between friction
conditions. Both actual target sequences fail again: nominal reaches final
braking but drifts beyond the unchanged yaw tolerance; low friction times out.
All 488 new controller decisions replay exactly, raw audit passes, and all
2,303 tests pass. No complete local sequence or JEPA/maze benefit is established.

Follow the [stopping/goal-hold plan](go2_action_conditioned_stopping_next_steps_2026-09-06.md):
bounded sensor-only terminal correction/verification and action-conditioned
forward/turn/brake prediction, frozen before new physical validation. Do not
relax old criteria or assume retrospective course predicts a changed action.
Establish repeatable local execution in a declared condition, then progress
memory/full-maze/JEPA matched comparisons while honestly retaining robustness,
optical/sensing, timing and hardware gaps. Do not replace the objective with an
endless series of local filters.

## Preceding heading-based visual-servo failures

The [bounded visual servo](go2_bounded_visual_servo_result_2026-09-06.md) executed
two fresh-start simulated trials: 348 sensor-selected decisions, 19,800 physics
samples and 368 RGB-D captures. Raw acquisition and exact command replay passed;
both target sequences failed. Nominal timed out during the turn, and low friction
exceeded the forward excursion bound. Body lateral motion and lower-than-commanded
forward response expose limitations of the simple heading-based controller.

Follow the [course-aware successor plan](go2_course_aware_visual_control_next_steps_2026-09-06.md),
not another replay or an unchanged retry. Use causal measured course/action
response, preserve stopping and failure records, freeze before genuinely fresh
validation, and retain controlled-floor/ideal-camera limitations. Then complete
local execution, memory/whole maze and matched JEPA predictive/multistep/memory
studies across independent layouts/seeds, timing, robustness and hardware.

## Preceding visual-led interface integration

The [visual-led interface](go2_visual_led_motion_integration_result_2026-09-06.md)
is implemented and replayed. All 904 visual records exactly match the frozen
baseline; all 3,600 intermediate queries retain explicitly historical pose.
Contact is optional evidence, not a new gate or fitted pose weight.

Follow the [bounded visual-servo plan](go2_bounded_visual_servo_next_steps_2026-09-06.md):
make actual commands depend on visual pose under explicit controlled-floor,
ideal-camera simulation conditions, with native stop-only supervision and new
physical initial states. This sequencing refinement enables an honest engineering
stage without falsely clearing the outstanding camera/terrain/hardware limits.
The next action is not another same-tape interface or accuracy comparison.
Continue to reliable local execution, memory, complete novel-maze mission and
matched JEPA predictive/multistep/memory comparisons across independent layouts
and seeds, timing, robustness and bounded hardware.

## Preceding frozen visual replay

The [frozen RGB-D friction replay](go2_friction_frozen_rgbd_dropout_result_2026-09-06.md)
completed with all 904 model observations available and an independent audit
PASS. Both models retain all 184 low-friction contact-dropout camera intervals.
Joint RGB-D has lower mean incremental error, while gyro-conditioned RGB-D has
less accumulated drift; neither is universally superior or a learned policy.
Follow the [visual-led integration plan](go2_visual_led_motion_integration_next_steps_2026-09-06.md):
separate pose, optional contact diagnostics and permission; preserve unavailable
channels and uncalibrated bounds, then genuinely fresh bounded physical
validation and short sensor-only execution. Do not repeat the completed replay
or add contact weights without evidence. Memory/full-maze/JEPA comparisons,
camera validity and deployment evidence remain outstanding.

## Preceding friction challenge and contact-dropout plan

The [fresh friction challenge](go2_support_friction_challenge_result_2026-09-06.md)
completed and passed its raw audit:24,000physics samples,452RGB-D views and2,252
live support predictions. Low friction loses contact consensus in815/1126frames
(541/600forward), versus2nominally; yaw and stopping also change substantially.
The nominal physics prefix through14.5s exactly repeats earlier fitting data
despite a new seed, so do not count it as independent validation. All2,255tests
pass, not sensor/terrain qualification.

Follow the [contact-dropout fusion plan](go2_contact_dropout_fusion_next_steps_2026-09-06.md):
first frozen RGB-D replay on both recordings including missing-contact intervals,
then separately named dropout-aware fusion and prospective action/braking with
genuinely fresh physical validation. Keep contact odometry availability distinct
from support, locomotion stability and clearance. Optical and hardware-channel
gaps remain; preserve the full execution/memory/JEPA/multistep/maze/seed goal.

## Preceding causal support fitting

The [causal support-kinematics study](go2_causal_support_kinematics_result_2026-09-06.md)
completed2,926fitting observations and an independent23,408-vector derivative
audit. Rolling correction improves mean forward velocity error5.52→2.81mm/s,
but maximum error worsens64.52→67.60mm/s;10observations lack consensus. All2,249
tests pass, not physical bounds. Follow the
[fresh nominal/lower-friction challenge preparation](go2_support_motion_challenge_next_steps_2026-09-06.md)
with both hypotheses frozen. No gate has been released; camera and feasible
hardware-sensor gaps remain. Continue to sensor-only short execution, memory
and the full matched JEPA/rollout/maze/seed/timing/hardware goal, not more
same-tape threshold adjustments.

## Preceding support-sensor reconstruction

The [ideal foot-load reconstruction](go2_ideal_foot_load_reconstruction_result_2026-09-06.md)
completed all30000fitting samples and an independent120000-vector audit. Four
feet have persistent above-threshold resultant loads at startup. This is a NEW
hypothetical three-axis sensor, not stock-Go2 force calibration or ground support.
The tape contains only ground foot loads and has no recorded torque channel.
Follow the [concrete causal integration plan](go2_support_kinematic_integration_next_steps_2026-09-06.md):
load/joint/IMU consistency, explicit stance/rolling/slip assumptions, sensor-only
predictions before native scoring, then fresh local execution evidence. No gate
was waived; camera defects and the deployment gap remain. Preserve the full
JEPA/memory/multistep/independent-layout/seed/hardware goal.

## Preceding camera/support contract findings

The [self-visible startup camera study](go2_startup_self_visible_camera_result_2026-09-06.md)
has now rendered all six fixed views and their controls. The outboard pair sees
420/675 sampled footprint points, not complete coverage. The independent ray
audit has 32 discrepancies; a separate culling diagnosis explains 30, leaving
two unresolved. All sampled front-mount rays hit robot geometry before the
renderer near clip. No candidate is qualified for policy integration.

Next implement the separately typed support-sensor diagnostic and explicit
current-support / future-footfall / body-sweep obligations described in that
result. Current loaded support does not prove continuous floor. Complete
underbody floor imagery is not a necessary navigation objective either: replace
that conservative diagnostic only with justified task-level evidence, not an
unseen-floor exemption. Resolve aperture/near-clip and residual raster issues
before integrating a new camera. Then sensor-supported start/forward/turn/brake,
memory and full matched JEPA mission evidence remain the execution priorities.

## Preceding numerical interface resolution

The [bounded numerical rotation adapter](go2_bounded_rotation_geometry_adapter_result_2026-09-06.md)
is complete and audited: all 708 previously rejected gyro coverage queries now
have conditional results under the unchanged geometry predicate, with explicit
numerical displacement allowances <=1.173e-12 m. No stored pose or previously
available coverage decision changed. All four histories now obtain full floor
coverage after travel; the earlier gyro failure was not a JEPA/joint-estimation
advantage. Physical uncertainty remains uncalibrated.

Startup is now the priority: all 27 initial body-floor footprint enclosures lie
before the camera's .2-m optical range boundary. The existing sensed channels
have no support/contact evidence, and the scene builder hides the robot. Follow
the [new startup-support sensing plan](go2_startup_support_sensing_next_steps_2026-09-06.md):
new self-occlusion-aware camera study, separately calibrated candidate support
sensor, then task-relevant body/surface and prospective forward/turn/brake error.
Do not insert a blanket unseen-floor assumption, fake foot loads from evaluator
labels or use the supervised motion prelude as deployment initialization.

Keep the conventional gait beneath the scientific contribution, as the original
SAINTS document intends. Return to short sensor-only closed-loop execution,
memory and the full matched JEPA/rollout/novel-maze/seed/hardware evidence as soon
as these concrete input/action interfaces are established. Goal remains active.

## Preceding longer motion and frozen pose transfer

The [fresh physical collection](go2_longer_observed_floor_motion_result_2026-09-06.md)
and [frozen pose/coverage transfer](go2_longer_motion_frozen_pose_result_2026-09-06.md)
are complete and audited. Both physical trials travelled over 2.1 m in the
forward segment. Initial observed floor covers all 27 actual body/leg shapes
from native frame 262 onward. Joint RGB-D tracks all 586 frames and obtains
sensor-estimated full coverage in 320/322 frames, with maximum position errors
23.162/20.861 mm. This resolves the prior observation shortage, not safe gait.

Nominal gyro pose is more accurate (16.654/17.381 mm), but 337/371 coverage queries
reject its composed rotations at the unchanged 1e-12 geometry tolerance. The
separate direct gyro remains within 5.73e-14; repeated keyframe composition grows
the numerical defect to 3.86e-12. The original coordinator failure is preserved;
the explicit-status successor records unknown coverage without model changes.
Do not call this interface failure a scientific joint-estimation advantage.

Next resolve the rotation representation/composition contract with quantified
numerical differences, not a silently enlarged gate. Then diagnose fitting-only
relative body/surface and prospective gait/brake error, freeze a fitting procedure
and obtain fresh reserved validation. The observed validation is no longer an
untouched selection set. Global position-error maxima are not clearance radii.

Also resolve startup observability: these successes follow about 26 s of
externally supervised travel before full floor coverage. A deployment policy
cannot assume that prelude. Evaluate a new downward/wider RGB-D sensing protocol
or validated contact/proprioceptive local support evidence; never expand the
evaluator-only setup region into a hidden map or assume unseen floor. Then
integrate short closed-loop motion and memory before the full mission and matched
JEPA/multistep/memory/layout/seed/timing/hardware stages below. Goal remains active.

## Preceding joint RGB-D rigid-pose fitting result

The [joint/gyro fitting comparison and independent pose audit](go2_joint_rgbd_rigid_pose_result_2026-09-06.md)
are complete. Joint admits all 336 nominal and both gyro-bias frames at 6.904-mm
maximum position error; the matched biased gyro modes reach 88.021/105.863 mm,
with the positive-bias control stopping at frame 275. Nominal gyro is better
(5.340 mm), so this is a bias-tolerance result, not universal accuracy improvement.
Gyro only monitors joint pose; no gyro bias estimator or learned policy was added.
Both error bounds remain unknown. The audit reconstructs all 317,801 accepted
point pairs and 227 promotions; correspondence identity is still not proved.

Fresh [longer physical collection V1](go2_longer_observed_floor_motion_development_v1_2026-09-06.md)
is now the execution step: 40-s forward stimulus, new fit/validation seeds,
unchanged bounded commands and external native supervision. Nominal joint/gyro
models and the comparison are fixed before new validation exposure. Audit raw
acquisition, actual travel/full-body observed-floor coverage and stopping tails;
then score the frozen estimators without retuning. Preserve incomplete trials.
This is not navigation continuing after a failed clearance gate.

Afterward derive and independently validate relative body/surface uncertainty
and prospective gait/braking; integrate the common floor/non-floor consumer,
complete actual discovery/backtracking/return and the full matched JEPA,
multistep, memory, layout/seed, timing and hardware requirements below. Longer
data acquisition alone does not qualify any command or achieve the goal.

## Preceding support-aware RGB-D V1

The [support-aware replay and audit](go2_support_aware_rgbd_pose_result_2026-09-06.md)
completed all336 fitting observations in nominal and both gyro-bias conditions,
with121 actual accepted reference promotions across the five members. Matching
and six-cell acceptance were unchanged; global history was retained. Nominal
maximum position error was5.143mm. All2,123 tests passed.

This resolves the recorded reference-continuity failure, not uncertainty or
navigation. Gyro bias produced105–107mm drift; local gyro bounds failed even
nominally, and robust global radii grew to0.46–0.58m. The noise member stopped
before registration when one perturbed valid pixel exceeded5m; retain that
malformed-packet failure, and model noise before range filtering in a separately
declared future experiment. The earlier full-body floor-coverage failure remains.

Next freeze joint RGB-D rotation/translation versus the unchanged support-aware
gyro-conditioned baseline, with robust correspondences, geometric conditioning
and gyro-disagreement diagnostics. Test bias correction and revise/validate error
hypotheses explicitly; do not just enlarge the angular allowance or reset global
history. Then obtain longer actual observed-floor motion, independently validate
prospective gait/brakes and complete the mission/JEPA/multistep/memory stages below.

## Preceding direct-keyframe evidence

The [fitting-only keyframe experiment](go2_keyframe_rgbd_pose_result_2026-09-06.md)
is complete, with 2,109 regression tests passed. Nominal admitted 40 frames versus
36 for the unchanged old observer, but all members failed before any actual
keyframe promotion. Non-blank failures were caused by reference grid support
falling to five cells versus the six-cell gate, not insufficient inlier count
or an unstable fit. More importantly, 537/16,361 accepted point pairs violate
the conditional static-point/lifting/localization assumptions; gyro-bias members
also violate the angular allowance. No controller integration is justified.

Next use a NEW model/protocol to test promotion from accepted observations based
on observed support/conditioning margin, with global pose/error history retained.
Diagnose the point-assumption failures and evaluate robust outlier treatment and
joint RGB-D rotation/translation against gyro-conditioned registration. Keep V1
immutable with its original rejection criteria/failures, and freeze selection before
new validation. Merely lowering the grid requirement or inflating a radius to
fit recorded maxima is not the next step. Continue with the required longer
physical collection and action/mission/JEPA stages below; this comparison did
not solve their missing evidence.

## Preceding sustained-motion evidence and acquisition requirement

The [audited sustained-motion collection](go2_sustained_observed_floor_motion_result_2026-09-06.md)
completed two new physical simulation trials: 35,000 physics samples, 672 RGB-D
frames, sustained motion, left/right turns and brakes, with no native guard
violations. Depth was rank two in every motion pair, including the first.
However, the intended full-body floor coverage was not achieved. Forward travel
was only 0.716/0.726 m; initial-view evaluator coverage peaked at 8/27 and 11/27
shapes. Both original shadow estimators stopped at 5.1 s after 36 accepted frames,
before any causal initial-view body-footprint coverage. Preserve these failures.

Next design a new fixed longer collection using fitting-only measured response,
not command integration: a 40-s forward segment at the unchanged 0.12-m/s command
is a candidate, subject to explicit scene/domain/resource checks and new trial
identities. The fitting-only virtual initial-posture query first covers all shapes
at a sampled 1.30-m offset; this is design evidence, not executed clearance.
Freeze a new relative RGB-D landmark/keyframe estimator comparison and prospective
action-response fitting rule before using new validation outcomes. Separate
local body/surface error from retained global/return history; never simply reset
or raise the failed estimator's 80-mm budget. V1's development-validation outcomes
are now known and are not an untouched test for later adaptive selection.

Then complete the execution and JEPA/multistep/memory/layout/seed/hardware stages
below. No intermediate data, geometry assay or regression substitutes for them.

## Historical stage notes (their “next” actions record the sequence, not the current queue)

Latest evidence: [finite RGB-D motion errors](go2_finite_rgbd_motion_errors_result_2026-09-06.md)
completed 16 fixed members across the three separately recorded appearance arms,
with exact nominal replay and all terminal failures retained. RGB blanking exactly
reproduced neutral fusion/failure in every arm; this establishes a limited causal
RGB state-estimation contribution, not JEPA navigation. All 2,071 tests passed.
Gyro bias produced materially larger errors; independent noise changed some
rank/stop decisions. All 124 actual-body surface queries had zero observed floor
coverage, so these short recordings cannot validate underbody clearance.

Immediate next action: freeze and collect new supervised development motion with
sustained travel into previously observed floor, turn/brake segments and measured
stopping tails, plus an initially weak-depth condition. Separate fitting and
validation trials, preserve native safety supervision and failures, and score
native references only outside prediction. Validate directional relative-body/
surface error and prospective action response; do not fit scalar clearance
multipliers to the short recordings or treat finite-member maxima as bounds.

Preceding evidence: [bounded depth-surface result](go2_bounded_depth_surface_result_2026-09-06.md)
implements explicit range intervals, a fixed-reference surface tube and robust
triangle orientation. All four previously rejected noisy members now have
conditional surface evidence. All 2,063 tests passed and extended-precision
endpoint checks covered every admitted cell. Original body-point allowances were
retained: front-left lower-leg clearance remains unresolved, and front-right's
nominal 0.011-mm margin changes sign under one noise pattern. No motion occurred.

The immediate next work is joint finite-error replay on the separately audited
shadow-motion development trials, including weak depth and original nominal budget
failure. Use actual RGB-D estimation, consistent shared errors and independent
represented sensor perturbations; preserve each member's terminal failure rather
than restarting it or discarding the ensemble. Score against native references
only after causal sensor prediction. Then validate a common physical floor/non-floor
consumer; the new tube alone cannot exempt obstacles or permit future gait.

Preceding evidence: [fixed multi-pixel plane diagnostic](go2_multipixel_floor_plane_result_2026-09-06.md)
implemented and audited a longer-baseline fit plus an explicit physical-footprint
query. It reduced the initial view's maximum gain-factor step discrepancy about
58-fold, but all four 0.1-mm independent-noise patterns failed the retained local
triangle-normal gate. Full regression: 2,042 tests passed. No controller change,
new mission or uncertainty calibration occurred. Next develop a depth-error-aware
surface model with per-pixel consistency, conditioning-based plane uncertainty
and preserved obstacle/missing-data checks; validate it before integration. Do
not simply widen the raw triangle gate or substitute fit residual for sensor noise.

Preceding evidence: [joint RGB-D pose/plane diagnostic](go2_joint_rgbd_pose_plane_result_2026-09-06.md)
implemented the actual shared-error estimator, but found 68–71% step dependence
in the two front-leg gap-factor vectors, dominated by range quantization in the
measured plane normal. Do not install those finite-difference covariances in
clearance. The immediate next work is finite-amplitude, quantization-aware error
propagation and separately validated observed-plane estimation, including noisy
independent development motion. All subsequent execution/scientific stages below
remain necessary.

Starting evidence: [audited fresh-maze failure](go2_fresh_fused_maze_audited_result_2026-09-06.md).
This replaces the immediate execution step of the
[previous plan](go2_fused_navigation_fresh_mission_next_steps_2026-09-06.md), not
its final scientific requirements. No predecessor run, source, threshold or
negative result may be rewritten. No sealed material is needed.

Progress update: the new RGB-D physical-configuration adapter and its recorded
seven-yaw/reference diagnostic are complete; see the
[implementation result](go2_rgbd_physical_configuration_result_2026-09-06.md).
All 27 primitives have conditional non-floor clearance, but two front lower-calf
shapes still lack demonstrated floor separation and four feet remain contact
candidates only. No motion was authorized. The immediate next work is step 2
with the actual RGB-D estimator, followed by the still-required step 3 execution
validation. Step 1 below records the implemented design and retained safeguards.

## 1. Replace the conflated clearance question in a new consumer

The 267 conflicts were floor returns admitted by historical pose envelopes, not
observed walls. Implement a new RGB-D-owned physical-configuration evidence
adapter before changing control. Keep one existing sensor-state owner; consume
its retained immutable observations and poses without another integration,
observer reset, simulator pose, supplied maze map or persistent setup prism.

Reuse the already tested physical support functions and prepared depth evidence,
not the height disks as physical collision identities. Existing starting points
are `primitive_floor_relation_development`, `primitive_floor_observation_development`,
`primitive_obstacle_memory_development`, and `factored_configuration_evidence_development`.
The latter's old owner/setup-region assumptions require an explicit new adapter;
do not globally patch or impersonate that owner.

For every actual primitive and supplied candidate configuration, retain distinct
answers for non-floor obstacle evidence, unpadded physical plane gap, observed
floor footprint coverage and exact foot-contact candidates. Keep padding and
uncertainty explicit. A measured upward plane is a hypothesis, not known ground;
missing floor and incompatible planes remain unresolved. Only the four exact
foot spheres can be contact candidates, never generic calf groups or height bands.
No candidate label itself permits contact or navigation. Preserve non-floor
conflicts even when another view is clear, and preserve covered physical
penetration or incompatible ground witnesses.

First exercise the new adapter on synthetic elevated floors, low non-floor
obstacles, invalid/border depth, partial views, ambiguous planes, exact feet versus
calves and deliberately penetrated configurations. Then perform a separately
bound saved-sensor diagnostic at the failed terminal configuration and candidate
yaws. Report residual unknowns, physical floor gaps and non-floor conflicts;
do not convert the old mission into a success. Static/posture queries alone must
not authorize future turning.

## 2. Derive relative error without deleting global uncertainty

The old sum of global endpoint scales can count a shared error twice. Derive
relative-pose errors from the actual estimator's shared variables and increments:
shared-error Jacobian differences must be transported in the correct frames.
An identical current view cancels; an arbitrary historical view does not. Shared
translation, shared rotation, velocity-prior sensitivity, depth/point noise,
gyro bias, calibration error and rejected correspondences have different effects.

Do not simply subtract scalar variances, assume independent RGB/depth estimates,
turn uncalibrated scales into confidence bounds, fit a radius to this run's small
observed error, or reset history at a junction. Maintain the global history needed
for return reasoning while bounding local relative queries. Add cancellation,
coordinate-invariance, bias accumulation, dropout, aliasing and false-match tests;
validate error coverage on independent development motion with injected sensor
and calibration faults before relying on the new bounds for execution.

## 3. Bind action permission to actual execution, not static snapshots

The learned gait's future joints/body response and stopping envelope still need
measurement and validation. Preserve the earlier held-out motion-response failure.
Collect a separately declared forward/turn/brake development set under existing
external native supervision; split trials for fitting and validation. Include
posture and recent command dependence, turning sway, lag, slip and stopping tails.
Use deployment-valid sensing for the predictor and native poses/contacts only for
scoring. A future command needs positive observed clearance for its prospective
body/foot envelope and an explicit stopping response; a static floor exemption
does not provide that. If calibration fails, retain the failure and its condition.

Address full-loop latency in parallel with this engineering work. The recorded
CPU loop is 2.0–3.5 times slower than a 10 Hz deadline even with ideal sensing.
Measure capture, state estimation, evidence queries, planning and actuation
separately as well as end-to-end. Optimizations must preserve reference decisions;
changing the control rate or asynchronous sensor handling is a new timed protocol,
not an accounting trick that makes old deadline misses disappear.

## 4. Complete development discovery-and-return with honest interventions

After the evidence/action interface passes its declared validation, freeze a new
controller and fresh mission protocol. Require genuine turns, wrong-branch
recovery, discovery of an initially hidden marker and physically verified home
return with a measured zero tail. Memory must support visited-branch and return
decisions while rejecting false loop closures. Report false marker/home claims,
contacts/falls, uncertainty stops, intervention/reset counts, path/time and latency.
Do not silently resume this failed attempt or retune on a protected final test.

## 5. Establish the JEPA claim, not just the navigation stack

Once a shared executable baseline completes development tasks, freeze sensors,
low-level gait, safety interface, datasets and compute budgets. Compare geometry-
only, matched supervised predictive models and JEPA predictive training. Separately
factor no/one/genuine multistep action-sequence rollout and memory off/on. Check
that the learned predictions actually change executed decisions; identical traces
cannot demonstrate a planning benefit. Treat independent layouts and training
seeds as experimental units, not repeated windows from the same route.

Measure success/return, safety, efficiency and full-loop cost with paired layouts,
uncertainty intervals and retained failures. Include appearance, depth, inertial,
occlusion and action-response shifts. An auxiliary loss improvement, a successful
single trajectory or a learned low-level gait is not the claimed JEPA navigation
result. Final evaluation requires independent external custody; legacy V4 and all
other protected sealed material remain ineligible/inaccessible.

Finally validate real-sensor timing/calibration, actual robot/model discrepancies,
deployment compute and bounded physical-platform operation when hardware access
exists. No present simulation result meets that requirement. Keep the ultimate
goal active until the full matched science and applicable hardware evidence exist.
