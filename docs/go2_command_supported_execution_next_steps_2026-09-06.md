# Next: respect gait command support, measure transitions, then navigate

Update: [the four command-pulse collections and raw audit are complete](go2_command_pulse_response_result_2026-09-06.md).
All 64 events were acquired, with variable net effects and important stopping
transients. Follow the [pulse-feedback execution plan](go2_pulse_feedback_execution_next_steps_2026-09-06.md)
next; do not repeat the completed collection or call its completion navigation
success. The original plan below is retained as context.

The [goal-hold experiment](go2_goal_hold_and_gait_command_support_result_2026-09-06.md)
failed 0/2. More importantly, the saved gait training configuration uses discrete
commands, whereas all nonzero recent servo commands interpolate below its first
nonzero training magnitudes. The nominal controller additionally stalled against
an INTERNAL correction margin despite being inside actual final pose tolerances.
Do not treat either defect as a JEPA or maze-observability limit.

## Immediate executable work

1. Build a new bounded action-response collection with the unchanged gait.
   Compare actual supported hold/forward-slow/yaw-left/yaw-right commands with
   the small commands used by the failed servo. Use fixed, counterbalanced
   pulse durations and command order, explicit new physical initial states,
   braking tails, nominal and lower-friction arms. Record RGB-D, body/joint/IMU
   history and exact issued/applied command sequences. Native pose only scores
   responses and native contacts/speed/domain remain stop-only supervision.
   Freeze schedule, episode/time budgets and partial-failure retention first.
   The saved bank's command amplitudes are NOT a safety certificate.
2. The old trials capped commands at 0.12m/s and 0.25rad/s, excluding even the
   bank's first nonzero primitives. A new protocol must explicitly declare
   0.20m/s forward and ±0.45rad/s yaw if testing those primitives; these are
   within the existing platform command limits, but require fresh bounded
   physical checks. Keep native speed and contact stops. Do not silently change
   or restart a frozen old trial. Preserve slew transients: the ±0.35rad/s
   per-tick yaw slew means the executed sequence differs from requested onset.
3. Measure onset, action-dependent lateral drift, cessation and post-command
   pose changes over multiple horizons. Analyze distinct pulse events/episodes,
   not overlapping windows as independent evidence. Identify adequate motion
   resolution, braking dispersion and failure modes before selecting control.
   This is directly useful training data for an action-conditioned predictor,
   unlike another fitted retrospective course filter.
4. Implement a bounded primitive-duration/pulse-and-observe baseline where
   appropriate. Correction events must return to zero-command ACTUAL-goal
   verification after a fixed finite duration; the 0.015rad intermediate margin
   is a design choice, not the objective. Keep final 0.06m/0.05rad tolerances and
   quiet hold requirement. Do not retrospectively relabel the goal-hold V1
   failure as success. Check approach, turning and stopping as a whole sequence.
5. If supported primitives cannot provide usable bounded local control, either
   establish a continuously commanded gait trained/validated over the required
   low-speed and stopping domain or use an appropriately validated conventional
   controller. Changing locomotion is permissible engineering work, not itself
   the JEPA contribution. Freeze any replacement and use it identically across
   later scientific arms. Do not keep the current gait merely because it loads.

## Preserve the scientific objective

Fresh validation should separate declared nominal repeatability from unresolved
low-friction robustness. Once local execution is repeatable in the declared
condition, proceed to persistent place/branch memory and complete exploration,
backtracking, goal and home missions; do not make exhaustive low-friction
perfection or a tighter internal pose target a substitute for those aims.

Use the same sensor/action histories, gait, data and compute/controller budgets
for geometric/persistence, supervised predictive and JEPA-predictive arms.
Distinguish representation training benefit from genuine online multistep
rollout and memory benefit, with independent layouts and training seeds. Action
space and effective gait command support must agree in every arm. Native
geometry/state cannot leak into decisions. Preserve all failures and do not use
exposed development runs as sealed final evidence.

Camera aperture/raster, near-field/body-sweep evidence, asynchronous/full-loop
timing and physical sensor calibration remain prerequisites for deployment-valid
claims. Hardware trials only when actual access permits. The complete scientific
goal remains active; a passing local controller, command-bank inspection or
test suite does not complete it.
