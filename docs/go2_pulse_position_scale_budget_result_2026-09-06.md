# Scaling helps recursive position fitting, but the empirical baseline still wins

The fixed comparison completed all18 fresh fits,2,160 optimizer updates and36
saved/reloaded snapshots. It isolates raw versus6cm-scaled XY loss at12 and120
updates, holding architecture, seeds, data exposure and other losses fixed.
All nine raw12 snapshots exactly reproduce the predecessor's model tensor hash.

## Full120-update primary-head comparison

Mean planar error in centimetres on the same185 train-role windows/917targets:

| Condition and objective | Seed0721 | Seed0722 | Seed0723 |
| --- | ---: | ---: | ---: |
| Direct, raw |2.77|2.00|2.17|
| Direct, scaled XY |4.60|4.03|3.04|
| Supervised recursive rollout, raw |6.88|2.29|4.43|
| Supervised recursive rollout, scaled XY |2.43|2.04|1.33|
| JEPA recursive rollout, raw |2.14|3.34|2.73|
| JEPA recursive rollout, scaled XY |1.07|1.95|1.67|
| Action/time empirical mean |0.859|0.859|0.859|
| Zero motion |2.233|2.233|2.233|

Full seeds are2026090721–2026090723. All auxiliary direct heads are retained in
the [complete metric/integrity record](go2_pulse_position_scale_budget_integrity_2026-09-06.json).
Every one of the30 trained heads at120updates remains worse than the empirical
control. Do not replace a failed primary rollout comparison with a favorable
auxiliary direct head or a selected seed.

At120updates, scaling improves recursive position prediction in every supervised
and JEPA seed, but worsens standalone direct prediction in all three seeds.
Scaled JEPA beats scaled supervised rollout in only two of three seeds. This is
a real, bounded optimization result, not a consistent JEPA advantage or a general
endorsement of the scaled objective.

Budget matters: scaled JEPA at12updates scores8.54/9.97/8.53cm, worse than raw
JEPA's4.55/4.93/4.10cm. At120updates it scores1.07/1.95/1.67cm. Longer raw JEPA
training also improves all three seeds. The earlier12-step negative result was
insufficient to conclude that the architecture cannot fit useful motion. Conversely,
these longer fits still do not establish value beyond the simple empirical control.

## Tradeoffs and remaining weaknesses

Scaled JEPA's120-update yaw errors are0.1110/0.1108/0.1170rad, worse than matched
raw JEPA's0.0807/0.0919/0.0933rad and much worse than the empirical control's
0.00795rad. Improving XY does not make this a uniformly better dynamics predictor.
Scaled JEPA's low-friction planar errors are3.72/2.83/2.70cm; the corresponding
zero-motion score is2.99cm. Low-friction improvement over zero is not consistent
across seeds. No friction coefficient was supplied to the model.

All contact targets remain negative. At120updates the scaled JEPA cumulative-
contact output violates nondecreasing probability on96.2% of adjacent known
horizon pairs under the frozen1e-6 test, even though all-negative classification
accuracy is1. These outputs are not calibrated collision risk or a deployable
safety mechanism. The planned next data stage needs actual hazard outcomes and
an appropriate cumulative-event contract, not another all-negative success claim.

This is deliberately one-room resubstitution: the longer schedule repeats the
same72 draws ten times (720draws,62distinct windows per fit). The other123 scored
windows are overlapping train-role data, not independent evaluation. Three model
seeds do not supply three independent mazes. Neither RGB utility, online rollout
benefit, memory benefit nor novel-maze completion was tested here.

## Verified artifacts and scope

Output: `.generated/go2_pulse_position_scale_budget_v1_attempt_001`.
Training handle38766 is terminal exit0; no process remains to resume.
Launch SHA256:
`321a82e7163c1e19d88f153a2552af10d468a4b21a3ce2345320448de80d79ef`.
Result SHA256:
`43e502d49702857079e304945597b07518704c7abe23892eb1dcfb5a75e5a5b9`.

Independent verifier91246 exits0. It checks all2,307 explicitly enumerated output
files (266,032,636bytes),36 checkpoint tensor/optimizer identities, every active
optimizer step count,2,160 per-update records against their aggregates/schedules,
and all nine raw12 predecessor identities. It independently reconstructs target
arrays, exact partial times, masks and all reported neural position/yaw/contact
scores by condition/action/time from saved predictions. Source/input bindings
are verified before and after. This is not a full gradient-training replay or
an independent rerun of all checkpoint forward passes; scoring at collection
time was performed after loading the saved weights by the tested frozen scorer.

The executed verification program is preserved as
`scripts/audit_go2_pulse_position_scale_budget_completed_v1.py`, SHA256
`3c0dcf72153f5a5f0cb40ae408b384397810d57acc7a2727499321e9b191ad77`.
It is posthoc read-only analysis, not part of the pre-fit training source closure.

The explicitly enumerated212-file regression completes with **2,654passed in
216.12s**, handle35311, exit0. Sixteen new learning/runner tests passed before
launch, including unit-scale equality to the frozen loss/gradients, identical
raw updates, matched observation populations, physical-scale arithmetic,
partial-time scoring and actual exclusive snapshot persistence. Tests are
implementation evidence, not additional physical or navigation outcomes.

No old experiment, source, checkpoint, failure or dataset was overwritten.
No sealed access, source export, GPU training, new physics, robot control or
navigation promotion occurred. The latest room-return result remains0/3.

## Next: leave this same-room optimization stage

The [independent-context collection design](go2_independent_pulse_context_collection_design_2026-09-06.md)
is now the next implementation task. Build and audit a bounded training-context
constructor/pairing pilot, then collect the prospectively split connected-layout
inventory with all action-duration cells, recent/quiet body histories, support
variation and correctly censored obstacle/contact outcomes. Do not keep searching
scales, increasing this run's budget or installing its apparent winner.

Key source-level finding: the old fixed excitation selector gates commands on
successful visual pose registration. A new simulation-only collector must avoid
that data-selection dependency while preserving external native stops and honest
missing-data accounting. Key observability constraint: friction may be unknowable
at rest; provide recent causal response evidence or retain uncertainty/belief,
not privileged friction labels. Preserve matched physical/observation prefixes
across candidate actions and hold entire layout families within one role.

Then test sensor/history/action utility on independent development layouts before
matched predictive-training, online-rollout and memory studies. Reliable local
execution, observed branches, physically executed backtracking, realistic sensors/
deadlines/body sweep and bounded hardware remain required. The original full
scientific goal is active and unachieved.
