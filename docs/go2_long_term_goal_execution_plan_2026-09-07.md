# Long-term scientific goal and active execution plan

User instruction: set the long-term goal and iterate towards it autonomously.
The full goal is **not achieved**. This is the current operational plan; older
frozen protocols and negative results remain unchanged. Obsolete BEV work is
outside scope.

## Completion means

Demonstrate RGB plus deployment-valid-sensor JEPA navigation on a Go2 in novel
mazes: reliable continuous motion, observation-grounded exploration and return,
useful persistent memory, meaningful predictive-training and online-planning
contributions against matched baselines, independent-layout generalization,
realistic sensing and timing, and bounded real-platform evidence when access
and physical supervision permit. A learned gait, predictor, replay result,
software test or engineered navigation demonstration alone is not completion.

JEPA superiority is a hypothesis, not a promised outcome. If sound experiments
reject it, retain that result and revise the scientific direction explicitly;
do not redefine the goal as passing an easier component test.

## Current evidence

- The twelve-layout collection and 36-fit study completed: 1,440 trials and
  43,200 optimizer updates. Full-JEPA development position error22.8mm is worse
  than action/time18.2mm and no-RGB JEPA18.6mm. The conditional benefit over
  supervised rollout does not establish RGB usefulness or maze navigation.
- The training diagnostic found a common collision-free action set in all120
  matched groups. Visual, goal-directed decisions were not required. Do not
  launch another large unchanged-task training sweep.
- Latest audited continuous room return remains0/3. Nominal tracking loss and
  lower-friction control error are different failure mechanisms.
- Tracking V1 finished all8 recording schedules,8base replays and88stress
  replays, then terminated during native audit with
  `ValueError('unit native quaternions required')`. No complete accuracy report,
  predecessor comparison or complete-result verification exists. The original
  attempt must not be restarted, resumed, overwritten or retrospectively passed.
- Original keeper session99647 is terminal exit1; its worker service is dead.
  Do not follow older checkpoint instructions to monitor or relaunch it.
- Subsequent read-only diagnosis completed: the maximum quaternion norm
  deviation is1.159e-7, with765 samples over the1e-7 gate. The separately labeled
  sensor-convention coverage readout finds0/8 tapes meeting all intended motion
  requirements. This is not a completed full physics audit or tracking result.
  See [the diagnostic and motion readout](go2_tracking_precision_failure_and_motion_readout_2026-09-07.md).
  Immediate next work is a distinct complete raw/accuracy analysis and measured-
  motion feedback, not another unchanged fixed-tape collection.

Failure identities:

- Inner failure: d0902842b1446882430093c4969ed78127a50256e0b66195207a40ce631a7486
- Outside terminal: 0cb3f84351b721ee4eb50db1ee837655c2a1acfc376410009893871dccd0c37d
- Complete stress phase: 671aa02549a748ca34af9258142fc698b920974baf529e245cd1dc691547e163
- Frozen definition: 223056ac7ddcb47b9d1a4b1b188028761fb15f56443f02fda28ecfa668326744

## Ordered work and exit evidence

1. **Diagnose the failed measurement audit.** Authenticate the complete sensor
   phase before native arrays. Quantify quaternion norms, precision, timing and
   sensitivity without altering the original data or scorer. Distinguish a
   numerical interface mismatch from invalid native evidence. A separate
   corrected analysis or successor needs explicit numerical justification and
   its own identity; no blind tolerance relaxation or repeat collection.
   Exit: trustworthy interpretable measurements or documented invalidity.
2. **Make continuous local execution reliable.** Validate translation, braking,
   both turn directions, large turns and return without pose resets. Current or
   stored sensor observations choose targets; evaluator pose/topology never
   chooses commands. Preserve sensor-loss stops and treat support-dependent
   dynamics separately. Derive tolerances from clearance and stopping needs,
   then freeze prospective acceptance and the complete population.
   Exit: repeated fresh closed-loop mission completion in a stated operating
   domain, not an assertion of all-terrain reliability.
3. **Collect informative visual decisions.** In parallel with execution work
   where feasible, develop balanced geometry-dependent progressing choices.
   Audit action/time, appearance, start and support shortcuts on training-only
   coverage. Execute actions far enough to encounter relevant geometry. Keep
   contact labels and observed pre-contact image/motion targets with separate
   denominators; never invent post-terminal futures. Split independent geometry
   before fitting and vary appearance separately.
   Exit: a fixed action preference cannot solve the intended progressing task,
   and the required observations/outcomes have actually been recorded.
4. **Test useful action-conditioned JEPA prediction.** Match direct prediction,
   supervised rollout, JEPA, action/time and persistence controls. Include RGB,
   depth when deployed, body-history and action ablations with correct input
   semantics. Inspect training curves, latent variability, target coverage and
   action dependence before large fits. Report update/data budgets plus actual
   compute and inference cost. Exit: prospective independent-development
   evidence supporting a specified benefit, or a retained negative result.
5. **Execute prediction-and-memory-guided maze missions.** Reuse existing
   WholeTaskNavigation, EpisodicRouteHypotheses and ObservedExploration event
   semantics. Observe branches, enter dead ends, physically backtrack, discover
   a target and return. Preserve uncertain place association in similar-looking
   corridors. Plan with candidate predictions, execute bounded commands and
   reobserve; graph transitions never substitute for physical travel.
   Exit: independently scored complete connected-maze missions.
6. **Separate causal contributions.** Compare JEPA versus supervised training;
   online rollout on/off with the same frozen model; memory on/off with the
   same executor and observations. Score mission success, collisions, incorrect
   place/home claims, time/path cost and all incomplete missions. Choose layout
   replication using a declared meaningful effect and development variability;
   optimization seeds and frames are not independent layouts. Freeze numerical
   acceptance criteria before evaluation. Keep final evaluation externally
   custodian-isolated; never use legacy V4 as final evaluation.
7. **Establish deployment validity.** Test realistic sensor visibility,
   calibration, synchronization, noise and dropout, then full-loop timing while
   physics advances. Report latency tails and deadline failure behavior, not
   only averages. Physical Go2 work requires actual access and supervision.
   Exit: bounded real-system evidence for the complete declared navigation
   stack and operating conditions.

## Execution discipline

Each iteration should produce a diagnosed cause, tested implementation,
completed experiment or substantive scientific interpretation. Do not substitute
ever-more preparation documents, integrity replacements or unchanged training
for progress towards physical navigation. Preserve frozen sources and attempts.
Assess CPU/affinity, RAM, GPU/VRAM, contention, storage and measured throughput
before substantial work; choose and test new concurrency before freezing it.
Use independent process workloads where appropriate, not maximum utilization
at the expense of timing validity or resource safety. No further deletion is
authorized by this plan. Existing benchmark custody remains in force.

## Goal tracker state

The existing product goal already states the full objective. On this resumption
it still reported historical `blocked`. An explicit create-goal request was
rejected because that goal is unfinished; available goal tools expose no resume
operation. Do not mark it complete to replace it. User-directed work continues,
with its actual state recorded in the local checkpoint. Product reactivation,
if needed for automatic continuation between turns, is a separate UI action.

## Supporting reports

- [Learning result and original next steps](go2_independent_pulse_parallel_result_and_next_steps_2026-09-07.md)
- [Training interaction diagnosis](go2_training_interaction_design_result_2026-09-07.md)
- [Latest audited room return](go2_inner_arrival_collection_result_2026-09-06.md)
- [Execution to online memory](go2_local_execution_to_online_memory_next_steps_2026-09-06.md)
- [Conditional independent heading follow-up](go2_independent_heading_followup_plan_2026-09-07.md)
- [Hardware utilization rule](hardware_utilisation_rule_2026-09-07.md)
