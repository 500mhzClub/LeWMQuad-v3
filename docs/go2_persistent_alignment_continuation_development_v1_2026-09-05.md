# Persistent initial-heading feedback: fixed continuous-task development protocol

The completed initial-alignment panel has2/4 task successes per method and2/4
timeouts before traversal. Its negative fixtures rotate toward the observed
target, then repeatedly leave the0.02 rad acceptance band when the controller
zeros yaw during the required0.3 s dwell. This successor tests a control-law
change, preserving the failed evidence and all acceptance and task endpoints.

## Intervention and fixed population

Sixteen trials: the same four1.2 m corner/tee fixtures and coupled lateral/heading
offsets, crossed with fixed-forward, direct prediction, supervised recurrent
prediction and JEPA recurrent prediction. Seeds2026093000–2026093003 are paired
across methods. These are reused development fixtures, not independent mazes.
All three learned ensembles remain frozen, with the same observations, action
bank and half-second decisions. No checkpoint fitting, selection or retry.

Initial observation still requires four actual RGB/body frames and a fresh
floor-extension bearing. The sole initial alignment change is to keep requesting
clip(1.5*heading_error,±0.35) rad/s while nonterminal, including inside the
acceptance band. No integral term, rate feedforward, minimum-turn command,
threshold sweep or altered acceptance logic is added. Both translation commands
remain zero. At completion or timeout, requested yaw is zero.

Acceptance remains error≤0.02 rad and measured heading rate≤0.1 rad/s continuously
for0.3 s, with12 s deadline. Unlike the predecessor, acceptance dwell now occurs
under active feedback: it is not a zero-command settling certificate. The
unchanged zero hold of at least1.5 s follows, and the first traversal must obtain
four new RGB/body observations. Measure true post-hold heading and translation
with the unchanged evaluation-only diagnostic. There is no hidden recheck/retry
that changes the declared acceptance rule after seeing the pose.

The controller inherits the predecessor predicate, clocks and deadline directly;
it overrides only the nonterminal command. The wrapper selects the successor
operator once, on the zero-command proposal frame before any alignment sample.
Later scan, coarse side-bearing alignment, holds, fresh branch reobservation,
traversal, provisional ledgers and fault/native-stop handling are unchanged.

One uninterrupted gyro reference spans all stages. Global80 s/801 decision
budget, final0.5 s release, CPU Genesis, frozen gait gains20/0.5, native immediate
contact/stability stops, ideal body sensing and live500 Hz virtual gyro remain
unchanged. Sensor failures latch; retain all failed and partial traces. These
are simulation mechanisms, not calibrated real-Go2 sensing or safety evidence.

## Tests, endpoints and interpretation

Before execution, test constant synthetic yaw disturbances of0.009 and0.02 rad/s
opposing target signs±0.087266 and±0.3 rad. Record commanded yaw separately from
disturbance-generated gyro; do not pretend the disturbance was a commanded
action. Unresponsive and stronger-disturbance cases must retain bounded failure.
Compare acceptance fields against the predecessor on identical observations;
test fresh model inputs, causal fault latching and exact inherited collector/
auditor function bodies. Synthetic disturbance is a simplified control test,
not a calibrated simulator/gait model or physical-success gate.

Primary endpoint: the unchanged two-leg integration outcome, including actual
crossing/release checks for each leg, completed scan, observed side selection,
completed controller and no contact/native stop/sensor fault. Initial timeout
is task failure. Do not report no contact as successful navigation when the
robot never traverses. Report all four fixtures per method, exact repeated
trajectories, command choices, true post-hold alignment, actual arrival pose,
scan drift/contact, duration and release outcomes. No confidence interval based
on duplicate dynamics, correlated frames or sixteen nominally different names.

Proportional feedback under constant disturbance has nonzero steady-state error;
this proposal is not guaranteed to meet the unchanged tolerance. Removing
deadband switching does not fix perception bias, lateral offset, post-hold
drift or swept-body clearance. Even successful initial alignment is only an
operator outcome; the intended evidence is improved continuous-task execution.
All learned methods share the hand-controlled operator, which remains outside
their frozen action bank. This panel does not isolate JEPA prediction of turns
or establish multi-step planning benefit.

## Evidence and continuation

Fresh root: `.generated/go2_persistent_alignment_continuation_development_v1_attempt_001`.
Bind six new source/test/protocol paths, recursive dependencies and all inherited
source/input/gait bindings. Require the exact completed predecessor launch,
result and full audit before execution. The collector records actual RGB,
sensor/fast streams and histories, commands, decisions, native contacts, physical
poses/joints and ledgers. Full audit replays all scientific fields and recomputes
unchanged physical metrics; only the two learned timing fields are excluded.
Preserve every outcome. No bound-source edit, empirical retry or criterion change.

Next, use measured arrival/scan outcomes to decide whether visual centering,
repositioning or additional declared sensor coverage is needed. Continue with
uncertain episodic place/branch hypotheses, actually observed hidden beacons
and directed return. Independent maze/task comparisons, full operator coverage,
robust perception/sensors and bounded real-platform evidence remain required.
This local control intervention does not redefine the final scientific goal.
