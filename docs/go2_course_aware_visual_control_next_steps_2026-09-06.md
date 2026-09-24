# Next: course-aware feedback and measured action response

## Execution update

The course estimator, supervised response diagnostic and new physical controller
have been implemented and executed. The small regression fails cross-condition
transfer. Both new target sequences fail: nominal at final brake/yaw drift,
low friction at timeout. Raw acquisition and exact controller replay pass;
2,303 regression tests pass. See the
[combined result](go2_course_response_and_control_result_2026-09-06.md) and
[current stopping/goal-hold plan](go2_action_conditioned_stopping_next_steps_2026-09-06.md).
The original plan below is historical; do not retry the frozen experiments.

V1 executed real closed-loop commands but failed both target sequences. Preserve
the [audited failures](go2_bounded_visual_servo_result_2026-09-06.md). The immediate
problem is now action-response/tracking, not the availability of visual pose.
Do not repeat the unchanged observer study, remove the excursion stop or merely
extend the V1 timeout until it appears successful.

## Implement a distinct controller successor

1. Estimate planar course and speed from a strictly causal window of visual
   positions in the initial observation frame. Keep measurement timestamps,
   low-speed/short-baseline unobservability and finite differencing error explicit.
   Body yaw and velocity direction are distinct. Use sensor history only; native
   lateral velocity above was evaluation, not a future controller input.
2. Add course-error feedback or an identified body-motion correction to the
   waypoint controller. Retain a visual-only baseline, bounded commands,
   terminal perception handling and native stop-only supervision. Avoid a
   persistent minimum-speed command near a target without considering stopping
   and turn radius. Freeze the causal window, gains, low-speed handling and phase
   transitions using development data before new execution. Test large lateral
   drift, changing course, low speed, delayed frames and brake/turn transitions.
3. Use the existing executed traces to fit/check an explicit action-response
   hypothesis with predicted displacement/yaw and model residuals. Commanded
   velocity is not observed velocity, and response depends on body/gait state
   and friction. Do not feed simulator friction labels as deployment sensing.
   Correlated closed-loop actions do not identify all counterfactual effects:
   if the data lack excitation, declare that and collect a separately fixed,
   bounded development schedule rather than reporting an identified model.
4. Choose a prospective duration/resource bound that accounts for measured
   forward response, accumulated heading correction and braking. Any changed
   limit is part of a new registered experiment; the previous timed-out task
   stays failed. Do not infer a universal margin from these phase averages or
   observed maxima. Keep the same complete forward/settle/turn/settle target
   sequence as the immediate outcome, not just partial movement.
5. Freeze the successor, then execute with genuinely new physical initial
   conditions and recorded raw sensor/controller traces. Maintain the explicit
   controlled-floor/ideal-camera scope. Count all failures; score position,
   heading, settling and full-loop timing from separate native evaluation.
   Step-synchronous success is not asynchronous or hardware success.

## Connection to the JEPA scientific goal

An action-conditioned predictor should model executed consequences in the
observed scene/body state, rather than learn nominal command integration. This
failure supplies a concrete prediction target and a baseline weakness, not proof
that JEPA fixes it. Compare a simple supervised dynamics model and engineered
course feedback against JEPA predictive training with matched sensor inputs,
action histories, data and control budgets. Keep predictive-training benefit
separate from any benefit of online multistep planning.

Once local target sequences work reliably, integrate persistent place/branch
memory and full novel-maze exploration, wrong-branch recovery, hidden goal and
home return. Resolve the unqualified optical/near-field/body-sweep and sensing
assumptions before deployment-valid claims. Evaluate independent maze layouts
and training seeds, memory and rollout ablations, robustness, end-to-end timing
and bounded hardware when available. Local servo success remains a stage, never
the full objective.
