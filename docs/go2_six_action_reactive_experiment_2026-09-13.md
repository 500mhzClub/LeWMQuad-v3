# Six-action non-predictive reactive development comparison

The earlier reactive comparator could choose forward or an in-place turn,
while the world-model planner could also choose left/right arcs and hold.
This experiment gives the reactive controller the exact same six command
primitives, with shared sensing, pose tracking, persistent map, route proposal,
8000-tick budget, stopping/dwell rule and physical evaluator.

The rule uses current observed waypoint bearing and current measured geometry.
When the observed waypoint connector permits translation, desired forward
speed is `0.20 * max(0, cos(heading_error))` m/s. Otherwise it is zero. Desired
yaw rate is heading error times `0.45 / 0.10` per second, saturated at ±0.45
rad/s. The constants come from the existing primitive speeds and heading
tolerance. Among eligible primitives, choose the nearest desired command in
squared distance, normalized by 0.20 m/s and 0.45 rad/s, with the original bank
order breaking ties. No coefficients were tuned against native results.

Current geometry failure admits no action and requests zero. Translation is
excluded while scanning or when the current waypoint connector is blocked.
The existing unknown-connector policy is retained and remains explicit.
No candidate future pose, learned score, motion forecast or residual correction
is evaluated. Static observed-route construction remains present.

This is a whole-method comparison with a matched action bank. Forecast utility
and predictive feasibility are not matched, so it does not alone isolate the
value of learned prediction. The original three-action reactive failure,
including its strict visibility failure, is retained.

Seven focused selection tests pass (2.01 s). A recorded-selection check uses
the earlier failed reactive trajectory only to check schema compatibility and
describe changed decisions; it cannot establish a changed physical trajectory.

The prospective assignment is layout 00, mode `six_action_reactive`, no model,
one native attempt at `go2_stop_conditioned_independent_00_six_action_reactive_no_model_v1_attempt_001`.
It uses the original perception implementation, not the separate timing
optimization candidates. The prepared runner performs collection and the same
complete raw audit serially. All failures are preserved. This is development
evidence on a previously examined layout, not a blinded benchmark or hardware
qualification.
# Queue status

The prepared case is queued behind the existing serial comparison process
(PID 3196768, creation time 1789286098.74). Its waiter is PID 3203505,
creation time 1789289459.78, recorded in
`docs/go2_six_action_reactive_queue_2026-09-13.json`. It starts only after the
preceding routing-memory case completes operationally; scientific success is
not required. The preceding JEPA assignment identity was checked against the
completed original run. No six-action native result exists yet.
