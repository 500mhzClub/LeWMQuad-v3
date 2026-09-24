# Longer observed-floor motion V1: fresh physical collection

This is a new supervised simulation experiment, not an extension, retry or
replacement of sustained-motion V1. Its purpose is to obtain actual full-body
travel over previously observed floor and subsequent turn/brake response. V1's
coverage failure remains unchanged. No navigation or hardware actuation occurs.

## Fixed design and rationale

Use the V1 wide-wall geometry, body/capture domain (-8,8) squared, native floor,
separate 32-m visual floor, camera, learned gait, checkpoint gains and physical /
policy / command clocks of 2 / 20 / 100 ms. New trial identities are fit: physics
2026090633, appearance 2026090635, spawn [-.5,-.3,0]; validation: physics
2026090634, appearance 2026090636, spawn [-.5,.3,.04]. Body spawn z is .375 m.
This remains ONE geometry, not an independent-maze generalization experiment.

After fifteen settling ticks, hold ten ticks, command forward [.12,0,0] for
400 ticks (40 s), brake ten, turn left [0,0,.25] for fifty, brake ten, forward
[.10,0,0] for thirty, brake ten, turn right [0,0,-.25] for fifty, brake ten,
then five extra zero ticks. Expected per complete trial: 580 stimulus ticks,
5 tail ticks, 30,000 physics samples, 586 RGB-D observations, 60 s of physics.
No adaptive length, stronger command, early success stop, restart or recovery.

The 40-s duration was selected from V1 FIT-only actual forward travel:
.716114 m / 15 s, not integration of the .12-m/s command. Constant-response
extrapolation suggests 1.91 m, exceeding the V1 fitting-only sampled 1.30-m
initial-posture coverage offset. Neither extrapolation is an execution envelope
or a guarantee of full-body coverage during gait. Actual travel and coverage
must be measured; insufficient coverage stays a negative result.

## Supervision and data ownership

Use the explicit LongerPhysicalInit/LongerMotionSession classes; no mutation of
predecessor modules, builders, specifications, clocks, thresholds or outputs.
Preserve inherited native contact/fall/tilt guards, exact-foot ground attribution,
base-speed <=.3 m/s and capture-domain checks on every active physical sample.
A physical stop terminates that trial immediately, without a recovery tail.
Native supervision is an external data-collection guard, not a deployment policy.
Native states and known geometry cannot enter pose prediction or navigation.

The unchanged original ShadowObserver runs for chronology only and never selects
commands. Its terminal latch is retained; stimulus can continue under external
supervision, not under failed navigation permission. Record RGB/depth/native
depth, body and fast-gyro histories, commands, all physical states/contacts,
camera/actuator/geometry identities, setup admission, stop causes and timing.

Freeze protocol, focused tests, source/import closure and inherited inputs before
launch. Verify before both trials and at completion. Require at least 10 GiB free
before launch and each trial; retain partial outputs on failure. Exclusive root:
`.generated/go2_longer_observed_floor_motion_development_v1_attempt_001`.
No sealed access, whole-tree export, GPU training, source editing after launch,
or silent retry is authorized by this protocol.

## Reserved evaluation before validation exposure

First independently audit acquisition, native guards, commands and actual motion.
Then replay the already frozen RigidRGBDKeyframePose joint and gyro modes on the
nominal recordings of BOTH trials, with no parameter fitting or model selection.
Construct every prediction before native scoring. Keep terminal failures and
compare on common admitted frames as well as report each full survival duration.
Report position/rotation errors and keyframe continuity; empirical maxima are
not calibrated uncertainty bounds. Do not retune from validation outcomes.

Evaluate initial-frame measured-floor footprint coverage at each camera sample
using actual joint configurations: native poses as a separately labeled evaluator
diagnostic and frozen sensor-estimated poses as a separate causal diagnostic.
Report first/full coverage counts, motion/turn/brake phases and all 27 shapes.
Zero additional shape error is only an observability diagnostic; it cannot give
motion permission. No union-of-history or future-view information is implied.

No action-response coefficients or uncertainty-fitting rule are specified here;
do not fit them after inspecting this validation and call that independent
validation. Those require a separately frozen procedure and fresh validation.
This experiment establishes whether the required physical observations exist
and whether the fixed estimator transfers to new motion/appearance trials.

Afterward derive and independently validate relative body/surface uncertainty and
prospective gait/braking, integrate full-body floor/non-floor control and online
memory, then complete genuine exploration/backtracking/marker/home missions.
Retain matched JEPA/supervised/geometry, multi-step rollout and memory factors,
independent layouts/seeds, real-time and bounded hardware evidence as unfinished
requirements of the ultimate goal.
