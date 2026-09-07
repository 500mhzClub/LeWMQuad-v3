# Task-directed evidence acquisition: fixed factorial and matched-model protocol

The completed persistent-feedback panel has2/4 two-leg successes per method,
one scan collision and one missed physical arrival. Its negative corner already
observes an eligible side branch1.04 s before continuing the full-circle scan
into contact. Its negative tee crosses successfully but fails a moving-image
arrival predicate; actual zero-release frames improve visual evidence, without
enough recorded quiet dwell to rescue that failed run. These observations
motivate two new acquisition decisions, not changed physical thresholds.

## Fixed28-trial population and contrasts

Use the same four1.2 m corner/tee development fixtures with coupled lateral and
heading offsets. Seeds2026100100–2026100103 are paired within each fixture.
For each fixture execute seven arms, in fixed order:

1. Fixed-forward, baseline acquisition.
2. Fixed-forward, stopped-arrival acquisition only.
3. Fixed-forward, task-directed scanning only.
4. Fixed-forward, both acquisition changes.
5. Direct-prediction controller, both changes.
6. Supervised recurrent predictor, both changes.
7. JEPA recurrent predictor, both changes.

This provides a2×2 acquisition-policy factorial under the same non-learned
movement primitive, plus a four-method comparison under the joint policy.
It does not measure learned-method interactions with each individual factor.
The fresh baseline component is part of this paired panel, not a restart or
replacement of the completed predecessor experiment. Old evidence is retained.
All geometries remain reused development material, not independent mazes.

The original frozen three-seed ensembles, available RGB/body inputs, five-action
bank,0.5 s choices, command limits/slew and gait are unchanged. There is no
training, checkpoint selection, parameter sweep or outcome-dependent retry.
Initial persistent-feedback alignment, post-alignment hold and fresh first-leg
warmup are shared across all seven arms.

## Factor A: stop to acquire arrival evidence

When the original traversal reaches its existing1.4 m applied-command progress
cap without three qualifying visual-change frames, request zero command and
enter BRAKING instead of declaring immediate FAILED_NO_VISUAL_CHANGE. Preserve
the earlier ordinary braking trigger. Progress is still a command-derived proxy,
not measured translation or an actual cell crossing.

Every existing arrival predicate stays unchanged: current articulated extent
plus margin, minimum progress, three current native-grid floor-change frames
≥0.10, measured body-quiet proxy and0.3 s quiet dwell, minimum0.5 s braking,
and2 s settling deadline. If stopped observations do not qualify, retain the
existing explicit failure. No unconditional candidate follows a distance/time
limit. Apply this acquisition change to both legs; preserve fresh per-leg RGB
history, provisional ledgers and the strict fault latch.

Source-comparison tests constrain the traversal-body change to that single cap
branch. New synthetic tests distinguish improved stopped observations from
persistent missing visual evidence and failure to settle. These tests do not
qualify command odometry, floor colour or physical arrival.

## Factor B: stop scanning after sufficient branch evidence

Continue the existing scan until a completed target view has a fresh eligible
side-opening proposal. Apply the unchanged side-branch selection rule to only
proposals observed at that same decision timestamp. Old stored observations
alone cannot trigger this early stop. No simulator target or wall position is
used. If no view provides an eligible branch, retain the original full-scan
completion/selection or failure behaviour and deadline.

Upon such a fresh proposal, request zero command before any further scan turn,
store the selected observed branch, and enter the existing HOLD_ALIGN state.
The nested scan's SCANNING status and proposed command are preserved as the
interrupted operator state; the outer command explicitly cancels that turn.
Record a separate scan_stop_evidence event, with actual view count and timestamps,
and full_circle_complete=false. Do not relabel an interrupted scan COMPLETE.

The existing zero hold, coarse bearing alignment, second zero hold and four
fresh second-leg RGB/body frames remain mandatory. A transported scan bearing
cannot replace fresh forward reobservation or become a trusted edge. This
policy gathers enough evidence for one local continuation, not all possible
exits or all beacons. Whole-task exploration must later decide when additional
views are worth acquiring; this is not full-environment observation coverage.

## Two distinct reported endpoints

Retain the entire predecessor reduction, including its completed_scan check and
two_leg_integration_success field. Those remain false for a successfully
interrupted, incomplete full-circle scan.

Declare task_two_leg_integration_success as this panel's primary navigation
endpoint. It retains every predecessor controller, per-leg physical crossing/
release, selected-branch, contact/native-stop and sensor-fault check. Only the
full-scan requirement is replaced by explicit evidence acquisition: either a
completed full scan, or a valid recorded fresh-branch interruption event. The
auditor replays that event from actual causal packets and validates the zero
command, acquired view, current proposal and unqualified identity flags.

This changes the acquisition policy and its task endpoint, not the physical
destination/contact/release thresholds. Compare like task endpoints across all
seven arms, and also report the original strict full-scan endpoint. No old
failed trial is retrospectively rescued. A partial scan without both actual
traversals/release and successful controller completion is a failed task.

Report all28 outcomes, fixture/method/policy identities, first/second physical
checks, false and missed arrivals, native contacts, stalls, durations, actually
chosen commands, interrupted/full scan counts and exact repeated trajectories.
Frames and deterministic repeated traces are not independent experimental units;
do not generate confidence intervals pretending these are28 independent mazes.
The learned arms use hand-designed acquisition rules outside their action bank;
no claim about JEPA predicting these operators or multi-step planning follows.

## Runtime, evidence and stopping scope

Fresh root: `.generated/go2_task_acquisition_continuation_development_v1_attempt_001`.
The exact completed predecessor launch/result/full-audit identities and all
inherited source/input/gait bindings must validate before launch. Bind eight
new source/test/protocol paths and their recursive ordinary-source dependencies.
All selected source discovery honors .ignore and protected benchmark exclusions.

Use CPU Genesis, checkpoint gait gains20/0.5, ideal50 Hz body sensing plus live
500 Hz virtual gyro,100 ms observation/control, unchanged global80 s/801
decision budget, and final0.5 s zero release. Native disallowed contact or body
stability limit stops physics immediately. Sensor failure latches, records its
actual time, and receives bounded zero release unless native stop interrupts it.
No hardware execution or calibrated sensor/safety claim is authorized by this
development protocol. Retain failures, partial traces and original model outputs.

Collector and auditor preserve actual RGB, causal histories/streams, commands,
native contacts, poses/joints, decisions and provisional ledgers. Verify paired
settling physics/body histories across all seven arms of each fixture. Audit all
28 trials, all actual decisions and both endpoint reductions. Exclude only the
two declared learned timing fields from exact decision equality. No source edit,
in-place retry, quiet-threshold change or population truncation after launch.

## Continuation toward the full goal

Use factorial outcomes to identify whether either acquisition change actually
improves continuous execution, including any new failure. If useful, integrate
the local executor with uncertain episodic place/branch hypotheses, physically
observed initially hidden beacons and directed return in a small connected maze.
Do not wait for perfect association certification, invent qualified edges from
proposals, or inject evaluation cell labels into runtime memory.

Independent mazes, appearance/sensor robustness, matched predictive-training
and online-rollout comparisons, full operator coverage, runtime costs and bounded
real-Go2 evidence remain required. This panel is local integration development,
not a narrowed definition of final-goal completion.
