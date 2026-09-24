# Next implementation plan after actuator correction

The [gain comparison](go2_actuator_gain_pair_development_v1_result_2026-09-05.md)
changes the priority: the integrated gait was configured incorrectly, and a
single measured correction makes the current two-crossing task work. Preserve
the corrected oracle execution reference. Do not mistake that result for RGB
navigation or treat further model complexity as the default next intervention.

Progress update: the corrected-gain command characterization and live ideal
simulated sensing/strict policy-reader package are now
[complete and audited](go2_causal_rgb_body_capture_development_v1_result_2026-09-05.md).
Steps 1 and 3 have this concrete development implementation; hardware validity,
odometry and realistic delay/noise remain open. The next execution package is
step 2 with this recorder attached, followed by action-diverse data and step 4.
Do not repeat the simple command arena as a substitute for broader task evidence.

## 1. Pin the effective execution contract

Use a new development session with explicit, checkpoint-derived actuator gains
and native readback before stepping. Record resolved joint names/indices, gait
and configuration hashes, effective gains, actuator latency, physics/policy/
command periods, clipping and previous applied command. Refuse silent defaults.
Keep the old studies and tracked frozen callers unchanged.

Add actual command characterization for zero, forward, reverse, left/right yaw,
mixed forward/yaw and stopping. Compare measured body-frame motion and latency
with requested and applied commands. Include yaw/velocity excitation and the
observed right-turn residual heading. Do not claim simulator-truth velocity is
a deployed measurement. Numerical task requirements and acceptance limits must
be specified before the next collection, not adjusted until an assay passes.

## 2. Expand physical scope before claiming reusable local execution

Generate fresh **development** cases, keeping all protected benchmarks closed:
vary spawn yaw/lateral offset and clearance; include a genuinely consecutive
second junction, offset turns and dead-end exit/reversal. Bind the case list and
budgets before execution. Include every generated start in the denominator;
do not condition the population on oracle success.

Retain the corrected baseline and compare a heading-aware arrival controller
only if the next junction makes that residual heading consequential. Fix its
parameters prospectively. Measure actual route continuation as well as the
arrival proxy, contact, falls, timeouts and distance. A failure should identify
whether the problem is gait tracking, body clearance, target semantics or
action coverage before any representation change.

## 3. Connect observations to the existing causal sensor buffer

Capture fixed-mount RGB at the actual decision boundary, with calibrated
intrinsics, exposure/measurement time and availability time. Acquire named
joint position/velocity and body-frame gyro/accelerometer histories with units,
clock alignment, validity, calibration and explicit reset identity. Preserve
chronology; do not mean-pool away motion order by construction.

For simulator development, explicitly label ideal simulated sensors and their
noise/latency assumptions. An accelerometer model uses specific force, not
unlabeled world acceleration. Simulator pose/velocity may generate sensor
references or teacher labels, but must not appear as model inputs under a
deployment-valid claim. Keep control history in a separate channel from sensed
history. Placeholder zero foot-contact telemetry stays invalid; actual contact
availability on hardware is not assumed.

Separate observation packets and oracle label packets by schema and access
interface. Test that future/late samples, wrong reset/calibration IDs, hidden
node IDs, global pose, true geometry, candidate outcomes and future sensors
cannot enter the observation encoder. Exercise dropout, sensor delay, camera
occlusion and gait-phase changes. Hardware calibration/availability remains a
separate gate; ideal simulated sensing cannot close it.

## 4. Establish a strong causal visual/action baseline

Collect synchronized sequences and diverse executed action outcomes using the
corrected plant. Include matched alternative actions, failures and recovery;
keep sibling branches and all descendants of a scene in the same data role.
Bind independent development-training and development-validation scene families
before fitting. Do not reuse observed test-like panels as final evaluation.

Use spatial RGB features, ordered body history, applied-command history and
explicit local intent. Start with a direct history/action-conditioned outcome
scorer or policy. A teacher-provided port remains a conditional local-control
experiment. For unknown-maze navigation, local targets must instead be detected
in current observations or selected from memory built online, not supplied by
the hidden map. Include this distinction in every result.

## 5. Test the JEPA claim, then integrate memory

Cross predictive versus nonpredictive training with online rollout versus direct
action selection, matching observations, encoder capacity, data, actions, safety
layer and gait. Include persistence/kinematics and action-/sensor-shuffle controls.
Use executed regret, contact-free progress and actual continuation, not latent
cosine alone. A predictive-pretraining benefit and an online-planning benefit
are separate conclusions. A credible null completes the JEPA comparison too.

Integrate provisional visual place associations with the existing directed
observed-edge graph. Discover beacons during exploration, then return using
observed memory. Test false merges, missed revisits and failed-edge invalidation.
The final comparison crosses rollout and persistent memory with matched recent
history in every arm. Independently generated mazes and maze-level uncertainty,
appearance/dynamics shifts, latency and a bounded real-Go2 pilot are required
before claiming the ultimate goal. Real-platform work requires actual hardware
access and a safe sensor/control deployment boundary, not an assumption that the
simulation adapter is ready to drive a robot.

Each package ends with executable tests, raw evidence, failure accounting and a
separate result document. Continue autonomous implementation within this scope;
do not replace empirical progress with repeated documentation or parameter
searches whose stopping condition is merely a positive outcome.
