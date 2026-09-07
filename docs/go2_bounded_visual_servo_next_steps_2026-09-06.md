# Next implementation: bounded sensor-selected visual servo in simulation

## Execution update: V1 complete, both target sequences failed

The fixed visual-feedback experiment executed both friction arms and passed raw
acquisition/command replay. Nominal timed out during its turn; lower friction
hit the forward excursion stop. No native physical stop occurred, but task
success is 0/2. See the [result](go2_bounded_visual_servo_result_2026-09-06.md)
and [course-aware successor plan](go2_course_aware_visual_control_next_steps_2026-09-06.md).
The original implementation plan below is historical, not permission to retry
the frozen V1 collector. Advance action-response/course control with fresh
validation while preserving these failures and the full scientific objective.

The motion interface is implemented, tested and integrated on recordings.
The next useful evidence is an actual closed-loop command, not another
unchanged-estimator accuracy comparison or a new wrapper around this wrapper.

## Declare the stage honestly

Run a controlled-floor, ideal-camera simulation engineering stage: the environment
generator supplies a continuous level support surface within a declared bounded
domain. That is an explicit experimental condition, not a conclusion inferred
from contact loads or unseen image pixels. The existing forward hidden-robot
RGB-D channel may be used only as an explicitly idealized control-development
arm. Its aperture/clipping shortcomings remain unresolved and it cannot qualify
the physical camera, real terrain or full novel-maze sensor stack.

This refines the sequencing of the earlier plan: it permits testing sensor-based
feedback under declared controlled conditions while optical/surface obligations
remain open. It does not delete old negative results or reclassify their failed
terrain checks. No legacy launch or source is modified or resumed. Do not infer
general safe locomotion, formal continuous coverage or hardware readiness from
an engineering trial. The full goal is unchanged.

## Implement and freeze before execution

1. Add a separately named 10Hz visual-servo controller using only current
   `VisualLedMotion` observations and a goal in the initial observation frame.
   For a short sequence, use forward translation, zero-command settling, a turn
   and final settling. Select commands from observed pose error, not elapsed
   command integration or native world state. Keep the pretrained gait unchanged.
   Missing current visual pose or terminal failure requests zero command and
   terminates the stage; contact dropout alone does not erase visual pose.
2. Fix targets, gains, command limits, acceptance tolerances and timeout before
   new data. Record their status as engineering choices, not calibrated safety
   bounds. A suitable first bounded design is roughly 0.4m forward and a 0.3rad
   turn at no more than 0.1m/s and 0.25rad/s, with explicit brake/settle phases.
   Final values must be source/protocol-bound before launch. Start with a single
   predeclared estimator, retaining the other as a future matched baseline rather
   than selecting between them per outcome. No contact-weight search.
3. Use genuinely changed physical initial pose/heading, not merely a different
   seed. Record a new scene/episode identity; verify the new physical trajectory
   is not an exact prefix of old fitting data. Predeclare nominal and lower-
   friction conditions if both fit the bounded budget, and report every failure.
   These are initial control-development trials, not held-out generalization.
4. Bind the exact new sources, protocol, inputs and native configuration.
   Verify actual gains/materials/geometry at acquisition. Retain the existing
   native contact, tilt, speed, domain and duration stop supervision, solely to
   stop execution or score it—not to choose headings or certify visual arrival.
   Record raw sensor histories, every controller input/output, measured and
   available clocks, stage transitions, applied commands and partial outcomes.
5. Evaluate observed target completion against separate native pose/twist:
   positional/angular error, overshoot, brake drift, settling, stops and wall
   timing. Do not turn the achieved finite error into a universal margin. A
   step-synchronous simulator run that pauses physics during compute must be
   labeled as such, not real-time execution. Subsequently test sensor/compute
   delay and stale-observation stopping with a declared asynchronous clock model.

## Do not stop at the servo result

Use the trial to establish an engineered local-execution baseline and to identify
the minimum action/brake model needed for longer navigation. Resolve the existing
camera aperture/raster and near-field support/sweep gaps before claiming a
deployment-valid sensing solution. Distinguish empirical bounded-risk testing
from formal guarantee; neither merely failing a conservative guard nor passing
a flat-floor stage settles the broader navigation question.

Then integrate persistent place/branch memory and complete novel-maze exploration,
wrong-branch recovery, hidden-goal discovery and return home. Compare JEPA
predictive training and genuine online multistep rollout against matched
geometric/supervised/no-prediction/no-memory baselines with equal sensors, gait
and budgets across independent layouts and training seeds. Full-loop timing,
robustness and bounded hardware evidence remain explicit end-state requirements.
