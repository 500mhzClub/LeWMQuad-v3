# Continuous observed branch and second traversal: fixed development protocol

The preceding20-trial integration established a first RGB-initiated traversal,
but every learned choice was forward and the arrival wrapper made errors in
the slower directional baseline. This successor advances continuous branch
observation and execution, adding the missing speed/cadence-matched non-learned
control. It is not an end-to-end learned policy, trusted place recognition,
multi-step JEPA planning, beacon return, independent-maze or hardware evidence.

## Fixed panel

Sixteen trials: the four preceding local1.2 m corner/tee fixtures, each with
fixed-forward, direct-only/direct, supervised recurrent and JEPA recurrent
prediction. Geometry and the coupled ±0.06 m lateral/±0.1 rad heading conditions
are reused deliberately; these are not new independent layouts. Seeds2026092800
through2026092803 are paired across methods. Source center connects east into
a corner/tee; that junction has a north corridor ending in another local space.
The expected second geometry is evaluation-only. Runtime never receives a north
command from the scene, cell identities, wall coordinates or a teacher route.

The frozen three-seed temporal ensembles and all first-traversal proposal,
command-progress, body-span, mask-change and quiet-arrival rules are unchanged.
No model fitting, seed selection, source edits to predecessors, tuning or retry.
Fixed-forward uses the same fresh four-frame history,0.3 m/s forward command,
500 ms choice/hold cadence and inherited traversal wrapper, without predictions
or a learned ranking cost. Its sensor history is validated even though the action
is constant. It is not credited with learned perception. The previous slower
directional baseline and all its failures remain recorded separately.

## One uninterrupted sensor-driven episode

CPU Genesis and the unchanged gait, kp=20/kv=0.5,1.5 s initial zero settling,
100 ms actual RGB/body observations and separate live ideal500 Hz gyro channel.
Maintain one high-rate relative orientation from the first post-settle frame
through all stages. Validate complete current slow histories and immutable
overlapping values/validity/availability throughout, including stationary holds.
Learned input tensors and fitted model parameters are unchanged.

1. FIRST: execute the unchanged first traversal wrapper. Its fourth fresh frame
   must independently propose an opening. Save the observed incoming direction
   in the uninterrupted initial-body reference. A failed child ends the task;
   ARRIVAL_CANDIDATE remains provisional and initiates a zero hold.
2. HOLD_SCAN: at least1.5 s of explicit zero requests, providing settled command
   history for the scan's independent ground estimator. State transitions occur
   after processing a hold frame; the next stage begins on the next100 ms frame,
   so each nominal1.5 s hold occupies1.6 s from its trigger to next-stage start.
3. SCAN: the unchanged FastGyroScan four-quarter-view-plus-return controller,
   with30 s deadline,0.35 rad/s cap,0.08 rad error/0.1 rad/s rate tolerance and
   0.3 s dwell. At initial and completed selected views, compute actual floor-
   extension proposals and transport their rays using the continuous gyro
   reference. No translation compensation or safe swept clearance is asserted.
4. Select only an observed side bearing45–135 degrees away from the observed
   incoming direction. Fixed exploration ordering is left before right, closest
   to perpendicular, then greater pixel support and newer observation. Forward
   and reverse candidates do not count as a side branch. No eligible observation
   gives FAILED_NO_SIDE_BRANCH. Accepted scan evidence is at most30 s old and
   always remains unqualified; it is only an alignment suggestion.
5. HOLD_ALIGN, then ALIGN: another zero hold followed by gyro alignment to the
   selected observed ray. Same0.35 rad/s cap,0.08 rad/0.1 rad/s tolerance and
   0.3 s dwell, with12 s deadline. No world yaw/pose enters alignment.
6. HOLD_SECOND, then SECOND: another zero hold and a fresh traversal instance.
   Four new RGB frames are required; a transported scan proposal cannot substitute
   for the fourth-frame forward observation. Its observed direction must agree
   with the selected ray within0.35 rad or the task fails reobservation. The new
   traversal follows the same proposal/progress/arrival rule as the first.
7. A second ARRIVAL_CANDIDATE ends as COMPLETE_PROVISIONAL; execute five zero
   release ticks. No place label or trusted graph edge is ever created. Separate
   ledgers preserve the two observation/attempt/arrival hypotheses.

The whole controller has an80 s budget, at most801 post-settle observations,
plus final0.5 s release. Native disallowed contact/body instability terminates
simulation immediately, with no more steps after that stop. Sensor-contract
faults latch failure and receive explicit bounded zero release unless native
stopping interrupts it. A later stop preserves earlier provisional records but
fails the continuous task. No teleport, state restore or oracle repositioning
is allowed between stages or methods; each trial starts from its declared spawn.

## Physical outcomes and limitations

For each started traversal, evaluate its candidate-time whole-body crossing and
destination membership, then the same final crossing/stability/release criteria
as the predecessor. The first leg ends its evaluation window at500 ms after its
child terminal, inside the actual zero hold, not at the end of the later scan or
second leg. The second leg uses final release. Truncate either window at a native
stop. Retag only the analysis copy's actual post-child rows as release and require
all250 requested-command rows to be zero; never mutate raw phases or invent
unexecuted rows. Independently report partial/missing stages and failed arrival
hypotheses. A later contact does not erase an earlier successful crossing, but
always fails the complete two-leg task.

Two-leg integration success requires both leg endpoints, completed scan, an
observed side-branch selection, COMPLETE_PROVISIONAL and no native stop/contact
or sensor fault. Report scan translation drift without calling the scan safe.
Preserve all sixteen outcomes, per-leg false/missed arrivals, stage failures,
commands, model predictions and provisional ledgers. Report method counts and
paired fixture differences, not confidence intervals from correlated frames or
deterministic repeated trajectories. All-forward equivalence remains a possible
and scientifically meaningful outcome of this integration test.

This remains a flat-wall/palette, wider-local-fixture development domain.
Stored scan rays have unmodeled translation drift; fresh reobservation only
partly addresses that uncertainty. Arbitrary grid cells are not directly sensed
place identities. Current URDF extent does not guarantee future gait clearance.
The existing native calf/foot ground-contact grouping is not foot-only support
certification. Ideal virtual sensing, simulation-paused inference and privileged
native emergency stops do not establish real-Go2 sensing, timing or safety.

## Fixed evidence and next decision

Fresh root: `.generated/go2_observed_continuation_development_v1_attempt_001`.
Before physical execution, bind recursive source/test/protocol closure, prior
launch/result/full audit, exact fitted checkpoints, Go2 URDF and gait. Preserve
raw physics/contact arrays, actual RGB, both sensor streams/histories, all
requests, decisions and ledgers. Pair initial settling physics/body histories
across the four methods. The full auditor reconstructs sensing/cameras and
replays every scientific controller field, excluding only the two nested learned
inference timing fields. It recomputes both actual leg windows and task outcomes.

Do not edit/retry the bound panel or adjust arrival thresholds from its outcomes.
Use failures to specify the next integrated observation/memory intervention.
The original scientific objective still requires robust place/exit/beacon
observations, online directed return, independent maze comparisons separating
predictive training from multi-step online rollout, and bounded hardware evidence.
