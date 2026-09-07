# Next: predict executed action and stopping consequences, then verify goal hold

Update: [goal-hold V1 is complete and failed 0/2](go2_goal_hold_and_gait_command_support_result_2026-09-06.md).
The saved gait's discrete command bank does not contain the servo's nonzero
commands. Follow the [command-supported execution refinement](go2_command_supported_execution_next_steps_2026-09-06.md)
before fitting another response model or changing servo gains. The original
steps below remain historical context, not a request to repeat V1.

The [course-aware trials](go2_course_response_and_control_result_2026-09-06.md)
remain 0/2 complete successes. Do not create another retrospective course filter
and assume it predicts the next command. Do not widen final tolerances or call
the nominal near miss a success. The supervised baseline also failed transfer.

## Immediate control work

1. Implement a distinct terminal goal-hold/correction procedure. A quiet robot
   can still drift out of its pose target during zero-command settling. Keep the
   final position/yaw tolerances unchanged and require them throughout a declared
   verification interval. Permit a bounded, prospectively specified correction
   phase if pose escapes, then reverify settling; this is a new algorithm, not
   resuming a failed old trial. Cap corrections and total time, retaining failures.
   Use sensor-only pose/history and report any prediction of stopping displacement
   as conditional, not a universal bound.
2. Distinguish observed motion under forward, turning and braking commands.
   Do not use turn-induced sideways motion as the predicted forward course.
   Add issued/applied action history and body/gait sensing to a simple supervised
   response baseline. First measure whether existing data support the desired
   action/phase transitions; rank alone does not establish identification.
   If the relevant transitions are missing, collect a fixed bounded development
   excitation schedule with declared command/stop budgets and new physical states.
3. Predict multi-step displacement/yaw INCLUDING the zero-command braking tail.
   Compare held-past-motion, command integration and supervised action-conditioned
   predictions. Keep sensor-generated targets and native evaluation separate;
   no simulator friction label is an online feature. Split fitting and subsequent
   physical validation before exposure. Neither fitting error nor cross-condition
   error on already inspected trajectories is final generalization evidence.
4. Freeze the new controller, stopping model and trial protocol before actual
   physical validation. Retain the same whole forward/settle/turn/settle task,
   observed excursion limits, native stop-only supervision, changed physical
   starts, exact source identities and all partial failures. Test nominal
   reliability and the known low-friction failure condition separately; nominal
   success must not be used to claim low-friction robustness. Keep controlled
   terrain/ideal-camera assumptions explicit and no real-time claims from paused
   simulator execution.

## Prevent local engineering from replacing the research

The role of this work is a reliable execution baseline and action-conditioned
prediction target. Once a declared operating condition supports repeatable local
execution, proceed to persistent place/branch memory and complete maze missions
while retaining other conditions as unresolved robustness limits. Do not silently
drop low-friction failures from reports or require arbitrary additional simulator
conditions as a substitute for progressing maze/memory science.

Compare the supervised predictor with a JEPA predictive objective using matched
RGB-plus-valid-sensor histories, executed actions, training data and controller
budgets. Separate predictive-training effects from genuine online multistep
rollout and memory effects. Native state may score outcomes but not leak into
policy inputs. Independent layouts/training seeds, complete exploration/backtrack/
goal/home results, full-loop latency, robustness and bounded hardware remain
required. Resolve camera aperture/raster, near-field/body-sweep and hardware
sensor calibration before deployment-valid claims. No stage result completes
the ultimate objective by itself.
