# Pulse-timed models trained; useful predictive control not established

Nine fixed fits completed: direct, supervised-rollout and JEPA for each of
three initialization seeds,12 optimizer updates each (108 total). Final model
and optimizer checkpoints are saved, reloaded and identity-checked. This is
actual training of the pulse-timed RGB/body predictor interface, but not a
learned navigation policy, independent-maze result or deployment-ready model.

The substantive negative result is that **every learned head has worse planar
error than zero motion** on this development corpus after the short budget.
Falling training/latent loss is not sufficient evidence of useful dynamics.

## Fixed-budget scores, not generalization

Each fit uses the same72 scheduled draws covering62 distinct pulse windows.
All185 windows (917 valid targets) are scored before/after, and all belong to
the same room-layout train role. Unsampled but overlapping train windows do
not become an independent evaluation set. Reported planar error below is the
mean over valid targets; there is no independent-layout confidence estimate.

| Trained primary head | Seed0721 | Seed0722 | Seed0723 |
| --- | ---: | ---: | ---: |
| Direct |6.50cm|4.03cm|5.92cm|
| Supervised recursive rollout |6.75cm|6.99cm|8.21cm|
| JEPA recursive rollout |4.55cm|4.93cm|4.10cm|
| Zero motion |2.23cm|2.23cm|2.23cm|

Full seeds are2026090721–2026090723. Auxiliary direct heads of the rollout/JEPA
arms also fail to beat zero motion in planar error; all values are retained in
the [output integrity/metric record](go2_pulse_training_pilot_integrity_2026-09-06.json).
JEPA's recursive head is better than supervised rollout in this tiny matched
pilot, but worse than the trivial positional control. That is not evidence
for JEPA navigation benefit. The12-update budget also cannot establish that
the architecture would fail after adequate training on adequate data.

JEPA recursive yaw errors are.12198/.12347/.12511rad; zero-yaw error is.12860rad.
The small resubstitution difference is not a generalization or safety claim.
Every contact target is negative, so100% threshold accuracy is uninformative
about hazard discrimination; the no-contact control achieves the same accuracy
without learning. The supervised recursive head violates cumulative-contact
monotonicity at every scored adjacent horizon pair. Other head-specific
violations and Brier scores are retained; no risk output is qualified for use.

## What is now implemented and verified

The CPU runner uses matched initial weights by seed, the frozen dataset and
partial-time objectives, AdamW(.001,no decay), gradient clipping1 and EMA.99
strictly after updates. Only intended active modules are optimized; EMA is
gradient-free. Direct, supervised-rollout and JEPA get the same online encoder
image exposure and sample schedule, not identical wall time or active parameter
count. Evaluators use the actual2.2/2.5s targets, preserve masks, restore model
mode and never supply future/native targets to inference.

Checkpoint state contains the model, optimizer moments/steps, seed, condition,
hyperparameters, completed update count and schedule identity. The final scores
come from reloaded weights. Output verification checks all9 fit records, all108
individual update files against the aggregate logs/schedule, all checkpoints'
tensor identities and every active optimizer parameter's step count12, plus
matching initial model hashes across arms for each seed. It verifies stored
metric identities but does not claim a full independent replay of gradients.

Eight runner tests pass, including actual recorded scoring. Full207-file
regression passes **2,619 tests in206.86s**; the additional end-to-end persistence
test passes separately in2.75s. These are208 distinct test files/2,620 tests
across the two runs, not a claim of one combined208-file invocation.

## Preserved infrastructure failure and corrected attempt

Original V1 (handle22530) stopped on an exclusive-writer conflict. Update1 was
persisted; update2 executed but its attempted overwrite of updates.json failed.
No complete fit or final checkpoint resulted. The original sources, log and
failure remain untouched; the lost in-memory state was not resumed.
Original launch SHA256:
`0123e2c16588abfcf8c6c3cdf5f77161c75da3b2f2bc77d104887ff3933904bd`.
Failure SHA256:
`68f4848b18a8569f9e811f71251addf131f78ae716fc59ac812fa8d817119070`.

Distinct V2 changes persistence only: immutable per-update files and one final
aggregate log. Its actual-writer test executes12 steps and refuses output reuse.
Models, seeds, data, schedules, budget and final-only selection policy remain
unchanged. The [V2 protocol](go2_pulse_training_pilot_v2_2026-09-06.md) and source
closure were frozen before fresh initialization.

V2 `.generated/go2_pulse_training_pilot_v2_attempt_001` completes with exit0,
handle15493. Launch SHA256:
`5059f97d20817d8781f1ecdabe10d811ad5e0741a2ca96861985a58a84e21984`.
Result SHA256:
`ecfbd5b84ac65821134c28b7983230c8f9196dce92325c1e4b2f0c683e15e032`.
Final verification handle28349 exits0, including complete source/input bindings
and153 output-file bindings. No source export, sealed access, data deletion,
GPU training, robot command or checkpoint promotion occurred.

## Scientific next steps

1. Before interpreting neural improvements, add a simple action/time-conditioned
   empirical predictor fitted to the same scheduled training labels. Diagnose
   motion/angle/contact/latent loss and gradient scales on development data.
   The current outcome loss mixes metre displacement with unit sin/cos and
   binary contact terms: centimetre accuracy can contribute little to total
   loss. This is a testable optimization concern, not a proven explanation
   of this short-budget result. Any normalization/budget change needs a new
   prospectively fixed comparison, not extension of this completed attempt.
2. Prioritize independent connected-maze geometry and state/action/support
   coverage, including the scarce short-forward and missing low-friction
   positive-turn cells, and properly censored obstacle/contact outcomes.
   Freeze actual layout roles and multiple seeds before scientific fitting;
   repeated balanced draws cannot repair missing independent information.
3. Establish useful task-relevant predictions against simple and neural matched
   controls, then RGB/history/action ablations. Integrate predictions into
   online selection only after demonstrating value; isolate predictive-training
   benefit from online rollout and memory benefit.
4. Continue sensor-support/uncertainty validation and state-dependent local
   execution after the paired0/3 room-return result. Complete observed branching,
   useful persistent memory and physical backtracking, independent-maze studies,
   realistic sensing/deadlines/body sweep and bounded hardware evidence.

The goal remains active and unachieved. There are now fitted pulse predictors,
but no demonstrated learned novel-maze navigation policy or JEPA advantage.
