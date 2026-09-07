# Startup sensing: obtain missing evidence, then return to learned navigation

## Current execution update

The next diagnostic is now implemented and audited:
[ideal foot-load reconstruction result](go2_ideal_foot_load_reconstruction_result_2026-09-06.md).
It supplies an explicitly hypothetical three-axis load stream for all30000fit
samples, plus a separate uncalibrated raw-count transport type. It is not a
hardware-equivalent or closed-loop result. Follow the
[causal support/kinematic integration plan](go2_support_kinematic_integration_next_steps_2026-09-06.md)
next; never turn resultant loading into ground/continuous-floor permission.

The six-view self-visible study is now executed; see the
[result and retained ray-audit failures](go2_startup_self_visible_camera_result_2026-09-06.md).
Step 1 below is no longer an unexecuted plan. No view is qualified: the
outboard pair has incomplete sampled coverage, front mounts intersect visual
geometry before clipping, and two sparse-ray mismatches remain unexplained
after a separately recorded back-face-culling diagnosis.

Proceed with steps 2–4 as a separately typed support diagnostic. Explicitly
separate current loaded support, prospective foot landing and body/leg/braking
clearance; neither require complete imagery of every underbody point as an end
in itself nor treat gaps as safe. Optical/aperture validation remains necessary
before any proposed camera is used by a controller. Keep the execution and JEPA
scientific milestones below intact.

## Evidence and scope

The [bounded numerical adapter result](go2_bounded_rotation_geometry_adapter_result_2026-09-06.md)
removes the recorded rotation-interface rejection without changing poses or
relaxing geometry thresholds. It does not solve startup. In both recorded initial
postures, ALL 27 conditional body-floor footprint enclosures lie entirely before
the actual camera's .2-m minimum optical-depth plane. Their maximum optical depth
is only .019897 m; much of the footprint is behind the forward-facing camera.
No JEPA loss, more training on the same view or wider FOV alone can recover an
observation that this fixed optical-depth/range contract excludes.

The current sensor boundary contains only RGB, depth, gyro, specific force,
joints and applied commands. Native support-contact and floor-identity checks
are evaluator-only. They must not be renamed into deployment observations.
The current scene builder also forces robot visualization off. A newly downward
camera would therefore get falsely unobstructed underbody rays if that builder
were reused unchanged. This is a limitation to address, not evidence that the
existing forward-facing images have been independently shown self-occlusion-correct.

The SAINTS progression document places conventional locomotion beneath the
research contribution and centers predictive representation, action-conditioned
rollout, memory, hierarchy and assurance. Keep the gait fixed unless measured
execution evidence requires otherwise. Its uses of “flat” largely mean
nonhierarchical JEPA, NOT a guaranteed continuous planar floor. No blanket
unseen-floor prior is being introduced here to make startup pass. Outdated BEV
requirements and legacy final-test references do not override current custody.

## Next bounded implementation

1. Implement a NEW render-only startup sensor study before another walking run.
   Compare the existing front camera, the same mount pitched downward, and an
   explicitly declared additional downward/outboard view. Fix candidate poses,
   calibrations and selection metrics before reading outcomes. Use the actual
   robot visual geometry with calibrated joint configuration; account separately
   for physical collision and visual meshes. Do not globally patch the old
   hidden-robot builder or reuse its depth calibration under a different mount.
   Report range/frustum exclusion, robot self-occlusion, observed floor and missing
   regions separately. A static study selects a sensing design, not a safe gait.
2. Add a separately named candidate support-sensor diagnostic. Unitree's official
   Go2 message includes foot_force and foot_force_est arrays; its example also
   reads motor tau_est. Crucially, the example labels the foot-force readings as
   integers rather than true force values. Those fields justify investigating a
   modality, not treating their numbers as calibrated Newtons or trusting them
   on an uninspected robot/firmware. [Official LowState definition](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/idl/go2/LowState_.hpp),
   [official Go2 low-state reader](https://github.com/unitreerobotics/unitree_ros2/blob/master/example/src/src/read_low_state.cpp).
   A hardware adapter must retain raw values, timestamps, identity, missingness,
   channel ordering and calibration status; no control or automatic unit conversion
   until verified. With no hardware recording available, label simulated load
   measurements explicitly as an ideal sensor hypothesis, not vendor equivalence.
3. For a simulator support stream, use a new, audited acquisition/conversion
   protocol: measure all loads on each identified foot, without ground/wall/object
   labels or filtering by evaluator ground IDs. Keep privileged contact identity
   solely for scoring. Test missing/stale samples, offsets, saturation, foot-wall
   contact, self-contact, slip, uneven supports and sensor-axis errors. Existing
   tapes can support an explicitly new offline diagnostic; they cannot retroactively
   become closed-loop runs with that modality. Do not fabricate a passing support
   packet from the existing evaluator's Boolean guard result.
4. Fuse only the evidence those modalities actually provide. Foot load indicates
   local loading, not a continuous floor, unobstructed leg sweep or absence of a
   hole between feet. Three loaded feet plus kinematics can support a conditional
   local-plane hypothesis; they do not establish terrain between/beyond contacts.
   Use visual/depth observations for newly explored regions and retain missing
   regions as unknown. Reject incompatible support heights and uncertain contacts.
   Any continuous-flat-floor baseline must be a separately declared assumption
   arm with appropriately limited claims, not a substitute for the original goal.

## Return to execution and the scientific comparisons

Use the chosen sensor configuration consistently across geometric, matched
supervised and JEPA baselines. Diagnose task-relevant body/surface errors and
fit prospective forward/turn/brake response on the fitting role only; freeze
the procedure before collecting fresh reserved validation. Explicitly test the
observed forward yaw drift, stopping tails and joint/body sweep, not command
integration. None of the currently exposed development tapes is fresh validation
for a new fitting rule.

The next execution milestone is a short sensor-only closed-loop start/forward/
turn/brake task that needs no privileged startup-motion prelude. Native safety
supervision may stop an experiment but must not choose navigation actions.
Then integrate persistent place/branch memory and complete genuine exploration,
wrong-branch recovery, hidden-goal discovery and home return. Establish that
JEPA training and genuine multistep rollout change decisions and outcomes against
matched baselines, across independent layouts and training seeds. Preserve the
later hierarchy, real-time, robustness and bounded hardware requirements.

These are engineering prerequisites for the scientific tests, not replacements
for them. Do not claim the learned navigation objective complete from a better
camera, a support sensor, passing unit tests or one successful motion tape.
